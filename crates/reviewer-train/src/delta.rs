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
/// `q,k`: `[B,S,H,Dk]`  `v`: `[B,S,H,Dv]`  `g,beta`: `[B,S,H]`  →
/// `([B,S,H,Dv] out, [B,H,Dk,Dv] final_state)`.
///
/// `initial_state` seeds the recurrence instead of starting from zero — this
/// is what makes a KV-cache decode step just "one more step of the same loop":
/// prefill runs the full sequence from a zero state and keeps the final state;
/// decode runs a single new timestep (`S == 1`) starting from that kept state.
pub fn recurrent_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    qk_l2norm: bool,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
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

    let mut state = match initial_state {
        Some(s) => s.clone(),
        None => Tensor::zeros((b, h, dk, dv), q.dtype(), q.device())?, // S[k,v]
    };
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
    Ok((out.transpose(1, 2)?.contiguous()?, state)) // ([B,S,H,Dv], [B,H,Dk,Dv])
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    /// The load-bearing question for training: does candle's autograd actually
    /// backprop through the full recurrence — the l2norm, the per-timestep loop,
    /// the `narrow`/`cat` seam — to a trainable parameter the inputs depend on?
    /// If a `Var` folded into `q` gets a non-zero grad from an output-derived
    /// loss, LoRA training through this op is sound.
    #[test]
    fn grads_flow_through_recurrence() -> Result<()> {
        let dev = Device::Cpu;
        let (b, s, h, dk, dv) = (1usize, 4usize, 2usize, 3usize, 3usize);
        // A trainable per-(head,key) scale that q depends on — grad must reach it.
        let w = Var::from_tensor(&Tensor::rand(0.5f32, 1.5f32, (h, dk), &dev)?)?;
        let base = Tensor::rand(0f32, 1f32, (b, s, h, dk), &dev)?;
        let q = base.broadcast_mul(&w.as_tensor().reshape((1, 1, h, dk))?)?;
        let k = Tensor::rand(0f32, 1f32, (b, s, h, dk), &dev)?;
        let v = Tensor::rand(0f32, 1f32, (b, s, h, dv), &dev)?;
        let g = Tensor::rand(-1f32, 0f32, (b, s, h), &dev)?; // decay in (0,1)
        let beta = Tensor::rand(0f32, 1f32, (b, s, h), &dev)?;

        let (out, _) = recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, true, None)?;
        let loss = out.sqr()?.sum_all()?;
        let grads = loss.backward()?;

        let gw = grads.get(w.as_tensor()).expect("no gradient reached the Var");
        let gnorm = gw.sqr()?.sum_all()?.to_scalar::<f32>()?;
        assert!(gnorm > 0.0, "gradient through the recurrence was zero");
        Ok(())
    }
}
