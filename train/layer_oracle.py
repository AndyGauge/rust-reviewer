#!/usr/bin/env python3
"""Oracle for a full DeltaNet *decoder layer* (layer 0 of the 9B): input_layernorm
-> mixer -> residual -> post_attention_layernorm -> MLP -> residual. Dumps the
layer weights + f32 I/O so the candle decoder layer can be verified as a unit.
Forces the recurrent form + pure gated norm inside the mixer (as in mixer_oracle).
"""
import copy

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map={"": 0}
)

layer = None
for name, mod in model.named_modules():
    if name.endswith("language_model.layers.0"):
        layer = mod
        print("found layer 0, type:", mod.layer_type)
        break
layer = copy.deepcopy(layer).to("cuda", torch.float32).eval()


class PureGatedNorm(torch.nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.eps = eps

    def forward(self, h, gate):
        h = h.float()
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + self.eps)
        h = self.weight * h
        return h * torch.nn.functional.silu(gate.float())


layer.linear_attn.norm = PureGatedNorm(
    layer.linear_attn.norm.weight.data, layer.linear_attn.layer_norm_epsilon
).cuda()


def rec_wrap(q, k, v, g, beta, initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    return M.torch_recurrent_gated_delta_rule(
        q, k, v, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
    )


layer.linear_attn.chunk_gated_delta_rule = rec_wrap

torch.manual_seed(2)
s, h = 6, layer.hidden_size
x = torch.randn(1, s, h, dtype=torch.float32, device="cuda")
pe = (torch.zeros(1, s, 256, device="cuda"), torch.zeros(1, s, 256, device="cuda"))
with torch.no_grad():
    out = layer(x, position_embeddings=pe)
    if isinstance(out, tuple):
        out = out[0]

tensors = {"input": x.cpu(), "output": out.float().cpu().contiguous()}
for k, val in layer.state_dict().items():
    tensors[k] = val.float().cpu().contiguous()
save_file(tensors, "layer_synth.safetensors")
print("saved layer_synth.safetensors")
for k in tensors:
    if k not in ("input", "output"):
        print(f"  {k:36} {list(tensors[k].shape)}")
