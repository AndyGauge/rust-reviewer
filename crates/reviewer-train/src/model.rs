//! Higher-level model pieces in candle: RMSNorm, the SwiGLU MLP, and the DeltaNet
//! decoder layer that assembles them around the verified mixer. Standard
//! transformer machinery — the risk is in wiring, which the layer oracle checks.

use std::collections::HashMap;

use candle_core::{D, Result, Tensor};

use crate::mixer::{linear, mixer_forward, silu};

const EPS: f64 = 1e-6;

/// RMSNorm with Qwen3.5's `(1 + weight)` scale (weight is stored zero-centered,
/// unlike the mixer's gated norm which uses `weight` directly).
pub(crate) fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    normed.broadcast_mul(&weight.affine(1.0, 1.0)?) // (1 + weight)
}

/// SwiGLU MLP: `down(silu(gate(x)) * up(x))`.
pub(crate) fn mlp(w: &HashMap<String, Tensor>, prefix: &str, x: &Tensor) -> Result<Tensor> {
    let gate = silu(&linear(x, &w[&format!("{prefix}gate_proj.weight")])?)?;
    let up = linear(x, &w[&format!("{prefix}up_proj.weight")])?;
    linear(&gate.mul(&up)?, &w[&format!("{prefix}down_proj.weight")])
}

/// A DeltaNet (linear-attention) decoder layer: pre-norm mixer + pre-norm MLP,
/// each with a residual. `prefix` locates this layer's weights.
pub fn decoder_layer_linear(w: &HashMap<String, Tensor>, x: &Tensor, prefix: &str) -> Result<Tensor> {
    let h = rmsnorm(x, &w[&format!("{prefix}input_layernorm.weight")], EPS)?;
    let h = mixer_forward(w, &h, &format!("{prefix}linear_attn."))?;
    let x = x.add(&h)?;

    let h = rmsnorm(&x, &w[&format!("{prefix}post_attention_layernorm.weight")], EPS)?;
    let h = mlp(w, &format!("{prefix}mlp."), &h)?;
    x.add(&h)
}
