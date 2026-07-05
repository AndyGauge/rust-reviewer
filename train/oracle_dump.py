#!/usr/bin/env python3
"""Stage-1 oracle: run the reference (transformers) model once on a fixed input
and save the exact numbers the candle port must reproduce.

This is the ground truth for the architecture port — the Rust forward pass is
"correct" when it matches these tensors. We save per-layer hidden states (not
just final logits) so a mismatch localizes to a specific layer instead of
"something's wrong somewhere in 32 layers."

    python oracle_dump.py --model Qwen/Qwen3.5-9B --out oracle_9b.safetensors

Output (safetensors, fp32, CPU): input_ids, hidden_0..hidden_N (embedding +
each decoder layer output), logits_last, argmax. Load it in candle and diff.
"""
import argparse

import torch
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)

# Fixed, deterministic input — a scrap of Rust so the tokens are in-distribution.
PROMPT = "fn main() {\n    let x = "


def is_multimodal(model: str) -> bool:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    if any(t in archs for t in ("ConditionalGeneration", "ImageText", "VL", "Vision")):
        return True
    return any(hasattr(cfg, a) for a in ("vision_config", "visual_config", "vision_tower"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", default="oracle_9b.safetensors")
    ap.add_argument("--layers", type=int, default=6, help="how many leading layer states to save")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loader = AutoModelForImageTextToText if is_multimodal(args.model) else AutoModelForCausalLM
    print(f"loading {args.model} ({loader.__name__}) ...")
    model = loader.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
    )
    model.eval()

    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    print(f"input_ids: {input_ids.tolist()}")

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states: tuple (embedding_output, layer_0_output, ..., layer_{L-1}_output)
    hs = out.hidden_states
    logits = out.logits  # [1, seq, vocab]

    tensors = {
        "input_ids": input_ids.to("cpu", torch.int64),
        "logits_last": logits[0, -1, :].to("cpu", torch.float32),
        "argmax": logits[0, :, :].argmax(-1).to("cpu", torch.int64),
    }
    n = min(args.layers + 1, len(hs))  # +1 for the embedding output at index 0
    for i in range(n):
        tensors[f"hidden_{i}"] = hs[i][0].to("cpu", torch.float32)  # [seq, hidden]

    save_file(tensors, args.out)
    print(f"saved {len(tensors)} tensors -> {args.out}")
    print(f"  seq_len={input_ids.shape[1]}  hidden={hs[0].shape[-1]}  vocab={logits.shape[-1]}")
    print(f"  layers saved: hidden_0..hidden_{n-1}")
    print(f"  next-token argmax (last pos): {tensors['argmax'][-1].item()} "
          f"-> {tok.decode(tensors['argmax'][-1])!r}")


if __name__ == "__main__":
    main()
