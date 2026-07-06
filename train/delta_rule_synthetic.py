#!/usr/bin/env python3
"""Synthetic oracle for JUST the gated-delta-rule recurrence — the novel crux.

Calls the *actual* transformers reference function on small random inputs and
saves inputs+output, so the candle port can be verified in isolation, before any
weight loading / projections / conv are wired up. Small dims so a human can
eyeball it and a mismatch is cheap to debug.
"""
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_recurrent_gated_delta_rule,
)

torch.manual_seed(0)
B, S, H, Dk, Dv = 1, 5, 2, 4, 4  # batch, seq, heads, key-dim, value-dim

q = torch.randn(B, S, H, Dk)
k = torch.randn(B, S, H, Dk)
v = torch.randn(B, S, H, Dv)
g = -torch.rand(B, S, H)  # gate is negative in the model (exp(g) in (0,1] = decay)
beta = torch.rand(B, S, H)

out, _ = torch_recurrent_gated_delta_rule(
    q, k, v, g, beta, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True,
)

save_file(
    {
        "q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous(),
        "g": g.contiguous(), "beta": beta.contiguous(),
        "out": out.contiguous().to(torch.float32),
    },
    "delta_synth.safetensors",
)
print(f"saved: q/k{list(q.shape)} v{list(v.shape)} g/beta{list(g.shape)} -> out{list(out.shape)}")
