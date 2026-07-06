//! The Gated DeltaNet recurrence, ported to candle — the novel crux of the port.
//!
//! Faithful translation of transformers' `torch_recurrent_gated_delta_rule`.
//! Per timestep, per head, with a matrix state `S[k_dim, v_dim]`:
//!   1. decay:  S *= exp(g_t)
//!   2. read:   kv_mem = Σ_k S[k,·] · k_t[k]
//!   3. delta:  δ = (v_t − kv_mem) · β_t
//!   4. update: S[k,v] += k_t[k] · δ[v]           (outer product)
//!   5. output: out_t = Σ_k S[k,·] · q_t[k]
//! with q,k optionally L2-normalized and q scaled by 1/√d_k.
//!
//! This is the recurrent (sequential) form — correctness first. A chunked
//! parallel form comes later for training throughput.

use candle_core::{D, Result, Tensor};

/// `x · rsqrt(Σx² + eps)` over the last dim (matches FLA / the reference l2norm).
fn l2norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let sumsq = x.sqr()?.sum_keepdim(D::Minus1)?;
    let denom = sumsq.affine(1.0, eps)?.sqrt()?;
    x.broadcast_div(&denom)
}

/// Recurrent gated delta rule.
/// `q,k`: `[B,S,H,Dk]`  `v`: `[B,S,H,Dv]`  `g,beta`: `[B,S,H]`  →  `[B,S,H,Dv]`.
pub fn recurrent_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    qk_l2norm: bool,
) -> Result<Tensor> {
    let (q, k) = if qk_l2norm {
        (l2norm(q, 1e-6)?, l2norm(k, 1e-6)?)
    } else {
        (q.clone(), k.clone())
    };
    // [B,S,H,D] -> [B,H,S,D]
    let q = q.transpose(1, 2)?.contiguous()?;
    let k = k.transpose(1, 2)?.contiguous()?;
    let v = v.transpose(1, 2)?.contiguous()?;
    let g = g.transpose(1, 2)?.contiguous()?; // [B,H,S]
    let beta = beta.transpose(1, 2)?.contiguous()?; // [B,H,S]

    let (b, h, s, dk) = q.dims4()?;
    let dv = v.dim(D::Minus1)?;
    let scale = 1.0 / (dk as f64).sqrt();
    let q = q.affine(scale, 0.0)?;

    let mut state = Tensor::zeros((b, h, dk, dv), q.dtype(), q.device())?; // S[k,v]
    let mut outs = Vec::with_capacity(s);
    for i in 0..s {
        let q_t = q.narrow(2, i, 1)?.squeeze(2)?; // [B,H,Dk]
        let k_t = k.narrow(2, i, 1)?.squeeze(2)?; // [B,H,Dk]
        let v_t = v.narrow(2, i, 1)?.squeeze(2)?; // [B,H,Dv]
        let g_t = g.narrow(2, i, 1)?.squeeze(2)?; // [B,H]
        let beta_t = beta.narrow(2, i, 1)?.squeeze(2)?; // [B,H]

        // 1. decay
        let decay = g_t.exp()?.reshape((b, h, 1, 1))?;
        state = state.broadcast_mul(&decay)?;
        // 2. read with key: kv_mem[v] = Σ_k S[k,v]·k[k]
        let k_col = k_t.reshape((b, h, dk, 1))?;
        let kv_mem = state.broadcast_mul(&k_col)?.sum(2)?; // [B,H,Dv]
        // 3. delta
        let beta_col = beta_t.reshape((b, h, 1))?;
        let delta = v_t.sub(&kv_mem)?.broadcast_mul(&beta_col)?; // [B,H,Dv]
        // 4. update: outer product k ⊗ delta
        let delta_row = delta.reshape((b, h, 1, dv))?;
        state = state.add(&k_col.broadcast_mul(&delta_row)?)?; // [B,H,Dk,Dv]
        // 5. read with query
        let q_col = q_t.reshape((b, h, dk, 1))?;
        let out_i = state.broadcast_mul(&q_col)?.sum(2)?; // [B,H,Dv]
        outs.push(out_i.unsqueeze(2)?); // [B,H,1,Dv]
    }
    let out = Tensor::cat(&outs, 2)?; // [B,H,S,Dv]
    out.transpose(1, 2)?.contiguous() // [B,S,H,Dv]
}
