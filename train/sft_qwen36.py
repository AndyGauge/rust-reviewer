#!/usr/bin/env python3
"""Path B: bf16 LoRA SFT of Qwen3.6-27B on the design-review dataset.

This is the *pragmatic baseline* (see docs/training-path-b.md). It exists to (a)
produce a working reviewer adapter and (b) generate reference metrics that a
future all-Rust (candle) implementation — Path A — can be compared against.

Consumes the chat JSONL produced by `reviewer-prepare` directly: each line is
{"messages": [...], "meta": {...}}. TRL applies the chat template; `meta` is
dropped before training.

Run on the GB10 (128 GB unified) — bf16, no quantization needed. See
train/README.md for environment setup and the metrics-capture protocol.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from trl import SFTConfig, SFTTrainer


class LengthGroupedSFTTrainer(SFTTrainer):
    """SFTTrainer that batches similar-length examples together.

    transformers 5.x dropped the `group_by_length` config flag, but the
    `LengthGroupedSampler` it drove still exists. With our heavily skewed lengths
    (median 377 tokens, tail to ~16k), grouping cuts padding waste dramatically —
    naive batch>1 was measured ~5x slower than batch=1 without it. Lengths are
    read from the processed dataset's own `input_ids`, so they align exactly with
    the indices the sampler returns.
    """

    def _get_train_sampler(self, train_dataset=None):
        # Length grouping only pays off when batching (batch>1). At batch=1 there
        # is no intra-batch padding, and grouping would just bias ordering, so
        # fall back to the default shuffled sampler. (Measured: on this GB10,
        # batch=1 is fastest anyway — the model is compute-bound, so batching
        # adds padding waste without throughput gain.)
        if self.args.per_device_train_batch_size <= 1:
            return super()._get_train_sampler(train_dataset)
        ds = train_dataset if train_dataset is not None else self.train_dataset
        try:
            lengths = [len(x) for x in ds["input_ids"]]
        except Exception:
            return super()._get_train_sampler(train_dataset)
        # Megabatch = per-device batch * grad-accum, matching HF's old behavior.
        mb = self.args.per_device_train_batch_size * max(1, self.args.gradient_accumulation_steps)
        g = torch.Generator()
        g.manual_seed(self.args.seed)
        return LengthGroupedSampler(batch_size=mb, dataset=ds, lengths=lengths, generator=g)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B",
                    help="bf16 safetensors base (NOT the -GGUF repo — that's inference-only)")
    ap.add_argument("--train", default="../data/prepared/rust-0.4.jsonl")
    ap.add_argument("--eval", default="../data/prepared/cookbook-0.4.jsonl")
    ap.add_argument("--out", default="out/reviewer-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1,
                    help="per-device batch size; measured fastest at 1 on GB10 (compute-bound)")
    ap.add_argument("--grad-accum", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="cap optimizer steps for a throughput probe (0 = full run)")
    return ap.parse_args()


def messages_only(ds):
    """TRL wants just the chat column; drop our provenance `meta` block."""
    return ds.remove_columns([c for c in ds.column_names if c != "messages"])


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # NOTE: Qwen3.6's Gated DeltaNet / hybrid attention is a brand-new arch.
    #   - `trust_remote_code=True` may be required until transformers ships it natively.
    #   - flash_attention_2 may not cover the DeltaNet path; fall back to "sdpa"
    #     or "eager" if you hit a kernel error on load.
    # Verified on the GB10 (transformers 5.12.1): the arch loads natively as
    # model_type=qwen3_5, class Qwen3_5ForConditionalGeneration — it's MULTIMODAL
    # (text+image+video), so we load via AutoModelForImageTextToText, NOT
    # AutoModelForCausalLM. No trust_remote_code needed. The vision tower loads
    # but is unused (we only feed text); narrowing target_modules (below) keeps
    # LoRA off the vision linears.
    # device_map={"": 0} forces the whole model onto GPU 0. Do NOT use "auto":
    # on GB10 unified memory accelerate misreads GPU capacity and offloads layers
    # to the meta/CPU device, which breaks the backward pass.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",  # flash-attn2 may not cover DeltaNet; sdpa is safe
    )

    train_ds = messages_only(load_dataset("json", data_files=args.train, split="train"))
    eval_ds = messages_only(load_dataset("json", data_files=args.eval, split="train"))
    print(f"train: {len(train_ds):,} examples | eval: {len(eval_ds):,}")

    # `all-linear` is the safe target for a novel arch where module names aren't
    # known yet. Refine to explicit q/k/v/o + FFN names once you inspect the model.
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
        # packing=False: with sdpa (flash-attn not built for sm_121 yet), packing
        # lets samples attend across boundaries -> cross-contamination. Off until
        # flash-attn is available. Costs some throughput (padding), buys clean loss.
        packing=False,
        # Train only on the assistant turn (mask the diff/prompt tokens). Requires
        # the chat template to emit assistant masks; if TRL warns it can't, the run
        # falls back to full-sequence loss — note which happened for the writeup.
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

    # --- metrics capture for the eventual Path A (Rust) comparison ---
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
