#!/usr/bin/env python3
"""Model-agnostic LoRA SFT for the design-review dataset (Path B, generalized).

This generalizes `sft_qwen36.py` (which is preserved as the exact record of the
27B run) so a *different* base model can be trained with identical everything
else — the whole point being an apples-to-apples model-size comparison. The only
model-specific decision is the loader class, which is auto-detected:

  - Multimodal bases (e.g. Qwen3.6-27B = Qwen3_5ForConditionalGeneration) load
    via AutoModelForImageTextToText.
  - Text-only bases load via AutoModelForCausalLM.

Override with --model-class {auto,causal,image-text} if the heuristic is wrong.

Keep the hyperparameters identical across models (rank, lr, seq, effective batch,
epochs) so the comparison isolates model size. Each run writes its own
baseline_metrics.json (wall-clock, peak mem, config) for side-by-side reading.

Run on the GB10 (128 GB unified) — bf16, no quantization. See train/README.md.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from trl import SFTConfig, SFTTrainer


class LengthGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer that batches similar-length examples together (batch>1 only).

    transformers 5.x dropped the `group_by_length` config flag; the underlying
    `LengthGroupedSampler` still exists. With our skewed lengths (median 377
    tokens, tail to ~16k) grouping cuts padding waste when batching. At batch=1
    there is no intra-batch padding, so fall back to the default shuffled sampler.
    """

    def _get_train_sampler(self, train_dataset=None):
        if self.args.per_device_train_batch_size <= 1:
            return super()._get_train_sampler(train_dataset)
        ds = train_dataset if train_dataset is not None else self.train_dataset
        try:
            lengths = [len(x) for x in ds["input_ids"]]
        except Exception:
            return super()._get_train_sampler(train_dataset)
        mb = self.args.per_device_train_batch_size * max(1, self.args.gradient_accumulation_steps)
        g = torch.Generator()
        g.manual_seed(self.args.seed)
        return LengthGroupedSampler(batch_size=mb, dataset=ds, lengths=lengths, generator=g)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="bf16 safetensors base (NOT a -GGUF repo — that's inference-only)")
    ap.add_argument("--model-class", default="auto", choices=["auto", "causal", "image-text"],
                    help="loader class; 'auto' inspects the config (multimodal -> image-text)")
    ap.add_argument("--train", default="../data/prepared/rust-0.4.jsonl")
    ap.add_argument("--eval", default="../data/prepared/cookbook-0.4.jsonl")
    ap.add_argument("--out", default="out/reviewer-9b")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device batch size (kept at 1 to match the 27B run; measure before raising)")
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps for a throughput probe (0 = full run)")
    return ap.parse_args()


def is_multimodal(model: str) -> bool:
    """Best-effort: does this base need the image-text-to-text loader?"""
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    if any(tok in archs for tok in ("ConditionalGeneration", "ImageText", "VL", "Vision")):
        return True
    # Multimodal configs nest a vision/visual sub-config.
    return any(hasattr(cfg, attr) for attr in ("vision_config", "visual_config", "vision_tower"))


def load_model(args):
    if args.model_class == "auto":
        mm = is_multimodal(args.model)
        print(f"model-class auto -> {'image-text' if mm else 'causal'} for {args.model}")
    else:
        mm = args.model_class == "image-text"
    # device_map={"": 0} forces the whole model onto GPU 0. Do NOT use "auto":
    # on GB10 unified memory accelerate misreads GPU capacity and offloads to the
    # meta/CPU device, breaking the backward pass. sdpa is the safe attention impl
    # for the Gated DeltaNet path (flash-attn2 may not cover it on sm_121).
    loader = AutoModelForImageTextToText if mm else AutoModelForCausalLM
    return loader.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def messages_only(ds):
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args)

    train_ds = messages_only(load_dataset("json", data_files=args.train, split="train"))
    eval_ds = messages_only(load_dataset("json", data_files=args.eval, split="train"))
    print(f"train: {len(train_ds):,} examples | eval: {len(eval_ds):,}")

    peft_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    sft_cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.seq,
        packing=False,  # sdpa: packing leaks attention across samples -> off
        assistant_only_loss=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_total_limit=3,
        report_to="none",
    )

    trainer = LengthGroupedSFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_cfg,
        processing_class=tok,
    )

    t0 = time.time()
    result = trainer.train()
    wall = time.time() - t0

    trainer.save_model(args.out)
    metrics = dict(result.metrics)
    metrics["wall_clock_seconds"] = round(wall, 1)
    if torch.cuda.is_available():
        metrics["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    metrics["config"] = vars(args)
    Path(args.out, "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"done in {wall/3600:.2f} h -> {args.out}")
    print(f"metrics -> {args.out}/baseline_metrics.json")


if __name__ == "__main__":
    main()
