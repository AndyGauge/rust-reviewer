#!/usr/bin/env python3
"""Oracle for the full Gated DeltaNet *mixer* (layer 0 of the 9B): dump the real
layer weights + a random input + the reference output, so the candle mixer can be
verified end to end. Forces the recurrent form (matching the ported kernel) and
runs in fp32 on CPU so the comparison tests the implementation, not bf16 rounding.
"""
import copy

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-9B", dtype=torch.bfloat16, device_map={"": 0}
)

delta = None
for name, mod in model.named_modules():
    if "GatedDeltaNet" in mod.__class__.__name__:
        print("found DeltaNet at:", name, mod.__class__.__name__)
        delta = mod
        break
delta = copy.deepcopy(delta).to("cuda", torch.float32).eval()

# Force the recurrent form so it matches the candle port (drop chunk-only kwargs).
def rec_wrap(query, key, value, g, beta, initial_state=None,
             output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None):
    return M.torch_recurrent_gated_delta_rule(
        query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel
    )

delta.chunk_gated_delta_rule = rec_wrap


# Replace the fused-Triton gated norm with a pure-torch one matching the candle
# port exactly (normalize -> weight -> * silu(gate)), so the comparison is clean.
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


delta.norm = PureGatedNorm(delta.norm.weight.data, delta.layer_norm_epsilon).to("cuda")

torch.manual_seed(1)
x = torch.randn(1, 6, delta.hidden_size, dtype=torch.float32, device="cuda")
with torch.no_grad():
    out = delta(x)
    if isinstance(out, tuple):
        out = out[0]

tensors = {"input": x.cpu(), "output": out.to(torch.float32).cpu().contiguous()}
for k, val in delta.state_dict().items():
    tensors[k] = val.to(torch.float32).cpu().contiguous()

save_file(tensors, "mixer_synth.safetensors")
print("dims: nv={} nk={} dk={} dv={} conv_k={} hidden={}".format(
    delta.num_v_heads, delta.num_k_heads, delta.head_k_dim, delta.head_v_dim,
    delta.conv_kernel_size, delta.hidden_size))
for k, v in tensors.items():
    print(f"  {k:22} {list(v.shape)}")
