#!/usr/bin/env python3
"""Greedy-generation oracle for Path A Stage 4b: run the real base(+LoRA) with
transformers `generate(do_sample=False)` on the same rendered (system, user)
fixture the Rust no-KV-cache loop uses, so the two can be diffed token for
token.

    python greedy_oracle.py --fixture chat_fixture.json \
        --adapter out/keep/checkpoint-1000-epoch1 --max-new-tokens 32 \
        --out oracle_greedy.safetensors
"""
import argparse
import json

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", required=True, help="JSON {system, user} from `reviewer-train dump-chat-fixture`")
ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
ap.add_argument("--adapter", default=None, help="PEFT LoRA adapter dir")
ap.add_argument("--max-new-tokens", type=int, default=32)
ap.add_argument("--out", default="oracle_greedy.safetensors")
args = ap.parse_args()

fixture = json.load(open(args.fixture))
messages = [
    {"role": "system", "content": fixture["system"]},
    {"role": "user", "content": fixture["user"]},
]

print(f"loading tokenizer + {args.base} (bf16) ...")
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
if args.adapter:
    from peft import PeftModel

    print(f"attaching adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

inputs = tok.apply_chat_template(
    messages, enable_thinking=False, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
prompt_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    out = model.generate(
        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
    )

ids = out[0].cpu().tolist()
print(f"prompt_len={prompt_len} total_len={len(ids)}")
print("generated:", repr(tok.decode(ids[prompt_len:], skip_special_tokens=True)))

save_file(
    {"ids": np.array(ids, dtype=np.int64), "prompt_len": np.array([prompt_len], dtype=np.int64)},
    args.out,
)
print(f"saved -> {args.out}")
