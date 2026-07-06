#!/usr/bin/env python3
"""Chat-template oracle for Path A Stage 4a: apply the real tokenizer's chat
template (thinking disabled) to the same (system, user) fixture the Rust side
renders by hand, and dump the resulting token ids for a byte-exact diff.

    python chat_template_oracle.py --fixture chat_fixture.json \
        --tokenizer .models/qwen3.6-27b --out oracle_chat.safetensors
"""
import argparse
import json

import numpy as np
from safetensors.numpy import save_file
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", required=True, help="JSON {system, user} from `reviewer-train dump-chat-fixture`")
ap.add_argument("--tokenizer", default="Qwen/Qwen3.6-27B", help="tokenizer dir or HF repo id")
ap.add_argument("--out", default="oracle_chat.safetensors")
args = ap.parse_args()

fixture = json.load(open(args.fixture))
messages = [
    {"role": "system", "content": fixture["system"]},
    {"role": "user", "content": fixture["user"]},
]

tok = AutoTokenizer.from_pretrained(args.tokenizer)
ids = tok.apply_chat_template(messages, enable_thinking=False, add_generation_prompt=True)
if hasattr(ids, "keys"):  # some transformers versions return a BatchEncoding/dict
    ids = ids["input_ids"]

text = tok.decode(ids)
print(f"rendered ({len(text)} chars):\n{text}")
print(f"ids ({len(ids)}): {ids}")

save_file({"input_ids": np.array(ids, dtype=np.int64)}, args.out)
print(f"saved {len(ids)} ids -> {args.out}")
