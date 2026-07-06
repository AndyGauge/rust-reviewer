//! The full Gated DeltaNet mixer in candle — the layer that wraps the verified
//! recurrence with projections, a causal depthwise conv, gating, and a gated
//! RMSNorm. Faithful to transformers' `Qwen3_5GatedDeltaNet.forward`.
//!
//! Dims are hardcoded for the 9B here (verification); they become config-driven
//! when this folds into the full model.

use std::collections::HashMap;

use candle_core::{D, Result, Tensor};

use crate::config::Config;
use crate::delta::recurrent_gated_delta_rule;

/// Look up `{prefix}{name}` in the weight map.
fn wt<'a>(w: &'a HashMap<String, Tensor>, prefix: &str, name: &str) -> &'a Tensor {
    &w[&format!("{prefix}{name}")]
}

/// `x @ w.T` for a no-bias Linear (`w` is `[out, in]`).
pub(crate) fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let inl = *dims.last().unwrap();
    let out = w.dim(0)?;
    let n: usize = dims[..dims.len() - 1].iter().product();
    let y = x.contiguous()?.reshape((n, inl))?.matmul(&w.t()?)?;
    let mut nd = dims[..dims.len() - 1].to_vec();
    nd.push(out);
    y.reshape(nd)
}

pub(crate) fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

pub(crate) fn silu(x: &Tensor) -> Result<Tensor> {
    x.mul(&sigmoid(x)?)
}

/// `softplus(x) = relu(x) + log(1 + exp(-|x|))` (numerically stable).
fn softplus(x: &Tensor) -> Result<Tensor> {
    let stable = x.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    x.relu()?.add(&stable)
}

/// Gated RMSNorm: normalize over last dim, scale by `weight`, then `* silu(gate)`.
fn gated_rmsnorm(h: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = h.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = h.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    normed.broadcast_mul(weight)?.mul(&silu(gate)?)
}

/// Repeat each head `rep` times along dim 2 (matches torch `repeat_interleave`).
fn repeat_interleave2(x: &Tensor, rep: usize) -> Result<Tensor> {
    let (b, s, nh, d) = x.dims4()?;
    x.reshape((b, s, nh, 1, d))?
        .broadcast_as((b, s, nh, rep, d))?
        .contiguous()?
        .reshape((b, s, nh * rep, d))
}

/// Full DeltaNet mixer forward. `x`: `[B, S, hidden]` → `[B, S, hidden]`.
/// `prefix` locates the mixer's weights (e.g. `""` standalone, `"linear_attn."`
/// inside a decoder layer). Thin wrapper over [`mixer_forward_prefill`] that
/// drops the cache it captures — kept so the existing oracle-verified callers
/// (`verify-mixer`/`verify-layer`/`verify-model`) don't change shape.
pub fn mixer_forward(w: &HashMap<String, Tensor>, x: &Tensor, prefix: &str, cfg: &Config) -> Result<Tensor> {
    mixer_forward_prefill(w, x, prefix, cfg, None).map(|(out, _state, _conv_tail)| out)
}

/// Same forward as [`mixer_forward`], but also returns the pieces a decode
/// step needs to pick up where this left off: the final recurrent state
/// `S [B,H,Dk,Dv]`, and the causal conv's last `kernel-1` *pre-conv* input
/// columns (`conv_tail`, `[B,conv_dim,kernel-1]`) — the causal conv's own
/// "history" a single new token's conv step needs.
///
/// `pad_mask` (`[B,S]`, 1.0 real / 0.0 padding) zeroes padded rows *before*
/// any projection — matching the reference's `apply_mask_to_padding_states`.
/// A zeroed row produces an exactly-zero `mixed`/`z`/`b`/`a` (the in-projections
/// are bias-free), so a left-padded prefix's causal conv sees an all-zero
/// window (output exactly zero) and the recurrence decays/updates a state
/// that started at zero by exactly zero — no separate padding-aware logic
/// needed inside the conv or the recurrence loop itself. `None` (the only
/// case before batching) skips the multiply entirely, byte-identical to the
/// pre-batching behavior.
pub fn mixer_forward_prefill(
    w: &HashMap<String, Tensor>,
    x: &Tensor,
    prefix: &str,
    cfg: &Config,
    pad_mask: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (nv, nk, dk, dv, kernel, eps) = (cfg.nv, cfg.nk, cfg.dk, cfg.dv, cfg.conv_kernel, cfg.eps);
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let conv_dim = cfg.conv_dim();
    let (b, s, _) = x.dims3()?;

    let masked;
    let x = match pad_mask {
        Some(m) => {
            masked = x.broadcast_mul(&m.unsqueeze(2)?.to_dtype(x.dtype())?)?;
            &masked
        }
        None => x,
    };

    let mixed = linear(x, wt(w, prefix, "in_proj_qkv.weight"))?; // [b,s,8192]
    let z = linear(x, wt(w, prefix, "in_proj_z.weight"))?; // [b,s,4096]
    let bb = linear(x, wt(w, prefix, "in_proj_b.weight"))?; // [b,s,32]
    let aa = linear(x, wt(w, prefix, "in_proj_a.weight"))?; // [b,s,32]

    // Causal depthwise conv over the qkv channels, then silu.
    let mt = mixed.transpose(1, 2)?.contiguous()?; // [b,8192,s]
    let conv = mt.conv1d(wt(w, prefix, "conv1d.weight"), kernel - 1, 1, 1, conv_dim)?;
    let conv = silu(&conv.narrow(2, 0, s)?)?; // causal slice [b,8192,s]
    let qkv = conv.transpose(1, 2)?.contiguous()?; // [b,s,8192]

    let query = qkv.narrow(2, 0, key_dim)?.contiguous()?.reshape((b, s, nk, dk))?;
    let key = qkv.narrow(2, key_dim, key_dim)?.contiguous()?.reshape((b, s, nk, dk))?;
    let value = qkv.narrow(2, key_dim * 2, value_dim)?.contiguous()?.reshape((b, s, nv, dv))?;

    let beta = sigmoid(&bb)?; // [b,s,32]
    // g = -exp(A_log) * softplus(a + dt_bias)
    let g = softplus(&aa.broadcast_add(wt(w, prefix, "dt_bias"))?)?;
    let g = g.broadcast_mul(&wt(w, prefix, "A_log").exp()?.neg()?)?; // [b,s,32]

    let rep = nv / nk;
    let query = repeat_interleave2(&query, rep)?; // [b,s,32,128]
    let key = repeat_interleave2(&key, rep)?;

    let (core, final_state) = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true, None)?; // [b,s,32,128]

    // Gated norm over each (·, head_v_dim) row, then out-projection.
    let core2 = core.reshape((b * s * nv, dv))?;
    let z2 = z.reshape((b * s * nv, dv))?;
    let normed = gated_rmsnorm(&core2, &z2, wt(w, prefix, "norm.weight"), eps)?;
    let core = normed.reshape((b, s, value_dim))?;

    let out = linear(&core, wt(w, prefix, "out_proj.weight"))?;
    let conv_tail = mt.narrow(2, s - (kernel - 1), kernel - 1)?.contiguous()?;
    Ok((out, final_state, conv_tail))
}

/// One-token DeltaNet decode: `x1` is `[B,1,hidden]`. `state`/`conv_tail` are
/// this layer's cache from the previous step (seeded by
/// [`mixer_forward_prefill`], advanced by this function every call after).
/// Does one recurrent step from `state` instead of replaying the sequence,
/// and a single causal-conv step over `[conv_tail, new_col]` — a plain
/// dot product, since that window is exactly one valid (unpadded) conv step.
pub fn mixer_decode(
    w: &HashMap<String, Tensor>,
    x1: &Tensor,
    prefix: &str,
    cfg: &Config,
    state: &Tensor,
    conv_tail: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (nv, nk, dk, dv, kernel, eps) = (cfg.nv, cfg.nk, cfg.dk, cfg.dv, cfg.conv_kernel, cfg.eps);
    let key_dim = cfg.key_dim();
    let value_dim = cfg.value_dim();
    let b = x1.dim(0)?;

    let mixed = linear(x1, wt(w, prefix, "in_proj_qkv.weight"))?; // [b,1,conv_dim]
    let z = linear(x1, wt(w, prefix, "in_proj_z.weight"))?;
    let bb = linear(x1, wt(w, prefix, "in_proj_b.weight"))?;
    let aa = linear(x1, wt(w, prefix, "in_proj_a.weight"))?;

    // window = [conv_tail (oldest..newest-1), new_col (newest)], chronological
    // order matching `conv1d.weight`'s own [oldest..newest] tap order.
    let new_col = mixed.transpose(1, 2)?.contiguous()?; // [b,conv_dim,1]
    let window = Tensor::cat(&[conv_tail, &new_col], 2)?; // [b,conv_dim,kernel]
    let weight = wt(w, prefix, "conv1d.weight").squeeze(1)?.unsqueeze(0)?; // [1,conv_dim,kernel]
    let conv_out = window.broadcast_mul(&weight)?.sum(2)?; // [b,conv_dim]
    let qkv = silu(&conv_out)?.unsqueeze(1)?; // [b,1,conv_dim]

    let query = qkv.narrow(2, 0, key_dim)?.contiguous()?.reshape((b, 1, nk, dk))?;
    let key = qkv.narrow(2, key_dim, key_dim)?.contiguous()?.reshape((b, 1, nk, dk))?;
    let value = qkv.narrow(2, key_dim * 2, value_dim)?.contiguous()?.reshape((b, 1, nv, dv))?;

    let beta = sigmoid(&bb)?;
    let g = softplus(&aa.broadcast_add(wt(w, prefix, "dt_bias"))?)?;
    let g = g.broadcast_mul(&wt(w, prefix, "A_log").exp()?.neg()?)?;

    let rep = nv / nk;
    let query = repeat_interleave2(&query, rep)?;
    let key = repeat_interleave2(&key, rep)?;

    let (core, new_state) = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true, Some(state))?; // [b,1,32,128]

    let core2 = core.reshape((b * nv, dv))?;
    let z2 = z.reshape((b * nv, dv))?;
    let normed = gated_rmsnorm(&core2, &z2, wt(w, prefix, "norm.weight"), eps)?;
    let core = normed.reshape((b, 1, value_dim))?;
    let out = linear(&core, wt(w, prefix, "out_proj.weight"))?;

    let new_conv_tail = window.narrow(2, 1, kernel - 1)?.contiguous()?; // drop oldest
    Ok((out, new_state, new_conv_tail))
}
