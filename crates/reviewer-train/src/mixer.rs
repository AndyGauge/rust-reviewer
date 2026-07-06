//! The full Gated DeltaNet mixer in candle — the layer that wraps the verified
//! recurrence with projections, a causal depthwise conv, gating, and a gated
//! RMSNorm. Faithful to transformers' `Qwen3_5GatedDeltaNet.forward`.
//!
//! Dims are hardcoded for the 9B here (verification); they become config-driven
//! when this folds into the full model.

use std::collections::HashMap;

use candle_core::{D, Result, Tensor};

use crate::delta::recurrent_gated_delta_rule;

/// `x @ w.T` for a no-bias Linear (`w` is `[out, in]`).
fn linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let inl = *dims.last().unwrap();
    let out = w.dim(0)?;
    let n: usize = dims[..dims.len() - 1].iter().product();
    let y = x.contiguous()?.reshape((n, inl))?.matmul(&w.t()?)?;
    let mut nd = dims[..dims.len() - 1].to_vec();
    nd.push(out);
    y.reshape(nd)
}

fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.neg()?.exp()?.affine(1.0, 1.0)?.recip()
}

fn silu(x: &Tensor) -> Result<Tensor> {
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
pub fn mixer_forward(w: &HashMap<String, Tensor>, x: &Tensor) -> Result<Tensor> {
    let (nv, nk, dk, dv, kernel, eps) = (32usize, 16usize, 128usize, 128usize, 4usize, 1e-6f64);
    let key_dim = nk * dk; // 2048
    let value_dim = nv * dv; // 4096
    let conv_dim = key_dim * 2 + value_dim; // 8192
    let (b, s, _) = x.dims3()?;

    let mixed = linear(x, &w["in_proj_qkv.weight"])?; // [b,s,8192]
    let z = linear(x, &w["in_proj_z.weight"])?; // [b,s,4096]
    let bb = linear(x, &w["in_proj_b.weight"])?; // [b,s,32]
    let aa = linear(x, &w["in_proj_a.weight"])?; // [b,s,32]

    // Causal depthwise conv over the qkv channels, then silu.
    let mt = mixed.transpose(1, 2)?.contiguous()?; // [b,8192,s]
    let conv = mt.conv1d(&w["conv1d.weight"], kernel - 1, 1, 1, conv_dim)?;
    let conv = silu(&conv.narrow(2, 0, s)?)?; // causal slice [b,8192,s]
    let qkv = conv.transpose(1, 2)?.contiguous()?; // [b,s,8192]

    let query = qkv.narrow(2, 0, key_dim)?.contiguous()?.reshape((b, s, nk, dk))?;
    let key = qkv.narrow(2, key_dim, key_dim)?.contiguous()?.reshape((b, s, nk, dk))?;
    let value = qkv.narrow(2, key_dim * 2, value_dim)?.contiguous()?.reshape((b, s, nv, dv))?;

    let beta = sigmoid(&bb)?; // [b,s,32]
    // g = -exp(A_log) * softplus(a + dt_bias)
    let g = softplus(&aa.broadcast_add(&w["dt_bias"])?)?;
    let g = g.broadcast_mul(&w["A_log"].exp()?.neg()?)?; // [b,s,32]

    let rep = nv / nk;
    let query = repeat_interleave2(&query, rep)?; // [b,s,32,128]
    let key = repeat_interleave2(&key, rep)?;

    let core = recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true)?; // [b,s,32,128]

    // Gated norm over each (·, head_v_dim) row, then out-projection.
    let core2 = core.reshape((b * s * nv, dv))?;
    let z2 = z.reshape((b * s * nv, dv))?;
    let normed = gated_rmsnorm(&core2, &z2, &w["norm.weight"], eps)?;
    let core = normed.reshape((b, s, value_dim))?;

    linear(&core, &w["out_proj.weight"])
}
