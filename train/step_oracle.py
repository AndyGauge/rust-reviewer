#!/usr/bin/env python3
"""Single-step teacher-forced oracle for an arbitrary input_ids sequence, with
the real rotary embedding — same tensor shape as oracle_full_f32.py's dump
(input_ids, cos, sin, logits, argmax), so `reviewer-train verify-model` can
consume it directly. Used to isolate whether a generation mismatch comes from
reviewer-train's own RoPE table (`rope.rs`) or from bf16 kernel drift between
candle and torch: if verify-model MATCHes using *this* oracle's cos/sin (real
rotary_emb) but reviewer-train's own generation loop (self-computed cos/sin)
diverges, the bug is in rope.rs, not the forward pass.

    python step_oracle.py --base .../Qwen3.6-27B --adapter <dir> \
        --ids '[1,2,3,...]' --out oracle_step.safetensors
"""
import argparse
import json

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
ap.add_argument("--adapter", default=None)
ap.add_argument("--ids", required=True, help="JSON list of input ids, or a path to a JSON file")
ap.add_argument("--out", default="oracle_step.safetensors")
args = ap.parse_args()

try:
    ids = json.loads(args.ids)
except json.JSONDecodeError:
    ids = json.load(open(args.ids))

model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map={"": 0})
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
s = input_ids.shape[1]
pos = torch.arange(s, device=model.device).unsqueeze(0)

rotary = None
for name, mod in model.named_modules():
    if name.endswith("rotary_emb") and "visual" not in name:
        rotary = mod

with torch.no_grad():
    emb = model.get_input_embeddings()(input_ids)
    cos, sin = rotary(emb, pos)
    out = model(input_ids=input_ids, use_cache=False)

logits = out.logits[0].float().cpu().contiguous()
tensors = {
    "input_ids": input_ids.cpu().long(),
    "cos": cos.float().cpu().contiguous(),
    "sin": sin.float().cpu().contiguous(),
    "logits": logits,
    "argmax": logits.argmax(-1).cpu().long(),
}
save_file(tensors, args.out)
top5 = logits[-1].topk(5)
print(f"seq_len={s}")
print("top5 @ last position:", list(zip(top5.indices.tolist(), [round(v, 3) for v in top5.values.tolist()])))
print(f"saved -> {args.out}")
