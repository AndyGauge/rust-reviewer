#!/usr/bin/env python3
"""Whole-model f32 oracle: the reference 9B forward on a fixed input, with the
recurrent form + pure gated norm forced across every DeltaNet layer (so it
matches the candle port), saved in f32 for a clean numerical comparison.

Saves input_ids, RoPE cos/sin, per-layer hidden states, and the final logits
(+ argmax). The candle full model loads the real weights and must reproduce
these — especially the argmax at every position.
"""
import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--out", default="oracle_full_f32.safetensors")
ap.add_argument("--bf16", action="store_true", help="run in bf16 (for the 27B)")
args = ap.parse_args()

MODEL = args.model
PROMPT = "fn main() {\n    let x = "
DTYPE = torch.bfloat16 if args.bf16 else torch.float32

model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=DTYPE, device_map={"": 0})


class PureGatedNorm(torch.nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, h, gate):
        dt = h.dtype
        h = h.float()
        v = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(v + self.eps)
        h = self.weight.float() * h
        h = h * torch.nn.functional.silu(gate.float())
        return h.to(dt)


def rec_wrap(q, k, v, g, beta, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    return M.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
    )


rotary = None
for name, mod in model.named_modules():
    if "GatedDeltaNet" in mod.__class__.__name__:
        mod.norm = PureGatedNorm(mod.norm.weight.data, mod.layer_norm_epsilon).to("cuda")
        mod.chunk_gated_delta_rule = rec_wrap
    if name.endswith("rotary_emb") and "visual" not in name:
        rotary = mod

tok = AutoTokenizer.from_pretrained(MODEL)  # noqa
input_ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
s = input_ids.shape[1]
pos = torch.arange(s, device="cuda").unsqueeze(0)

with torch.no_grad():
    emb = model.get_input_embeddings()(input_ids)
    cos, sin = rotary(emb, pos)
    out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

hs, logits = out.hidden_states, out.logits
tensors = {
    "input_ids": input_ids.cpu().long(),
    "cos": cos.float().cpu().contiguous(),
    "sin": sin.float().cpu().contiguous(),
    "logits": logits[0].float().cpu().contiguous(),
    "argmax": logits[0].argmax(-1).cpu().long(),
}
for i in range(min(9, len(hs))):
    tensors[f"hidden_{i}"] = hs[i][0].float().cpu().contiguous()

save_file(tensors, args.out)
print(f"saved. seq={s} vocab={logits.shape[-1]} rope_dim={cos.shape[-1]}")
print("argmax:", tensors["argmax"].tolist(), "->", repr(tok.decode(tensors["argmax"][-1])))
