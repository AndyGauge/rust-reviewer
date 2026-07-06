#!/usr/bin/env python3
"""Oracle for a full-attention decoder layer (layer 3 of the 9B). Dumps weights +
f32 I/O + the RoPE cos/sin + uses an explicit causal mask with eager attention,
so the candle port can be verified given cos/sin (RoPE *generation* is verified
separately at the full-model level).
"""
import copy

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map={"": 0}
)

layer, rotary = None, None
for name, mod in model.named_modules():
    if name.endswith("language_model.layers.3"):
        layer = mod
    if name.endswith("rotary_emb") and "visual" not in name:
        rotary = mod
print("layer 3 type:", layer.layer_type)

layer = copy.deepcopy(layer).to("cuda", torch.float32).eval()
layer.self_attn.config._attn_implementation = "eager"

torch.manual_seed(3)
s, h = 6, layer.hidden_size
x = torch.randn(1, s, h, dtype=torch.float32, device="cuda")
pos = torch.arange(s, device="cuda").unsqueeze(0)
cos, sin = rotary(x, pos)
cos, sin = cos.float(), sin.float()

# Explicit causal additive mask [1,1,s,s]: 0 on/below diagonal, -inf above.
mask = torch.full((s, s), float("-inf"), device="cuda").triu(1).view(1, 1, s, s).float()

with torch.no_grad():
    out = layer(x, position_embeddings=(cos, sin), attention_mask=mask)
    if isinstance(out, tuple):
        out = out[0]

tensors = {
    "input": x.cpu(),
    "output": out.float().cpu().contiguous(),
    "cos": cos.cpu().contiguous(),
    "sin": sin.cpu().contiguous(),
}
for k, val in layer.state_dict().items():
    tensors[k] = val.float().cpu().contiguous()
save_file(tensors, "attn_synth.safetensors")
print("saved attn_synth.safetensors  (cos/sin:", list(cos.shape), ")")
for k in tensors:
    if k not in ("input", "output", "cos", "sin"):
        print(f"  {k:40} {list(tensors[k].shape)}")
