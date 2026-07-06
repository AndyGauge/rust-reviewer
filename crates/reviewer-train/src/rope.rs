//! Text RoPE for Qwen3.5/3.6. The real rotary embedding is "interleaved
//! M-RoPE" (separate temporal/height/width position streams, for image/video
//! tokens) — but for text-only input all three streams carry the same
//! position ids, which makes the interleave a no-op: it reduces to plain RoPE
//! over the first `rotary_dim()` dims of each head. `model::apply_rope`
//! already expects exactly that (partial rotary, `cos`/`sin` narrower than
//! `head_dim`), so this just has to produce the right table for arbitrary
//! sequence lengths (the oracle only ever supplied one fixed prompt's table).
use candle_core::{Device, Result, Tensor};

use crate::config::Config;

/// cos/sin tables for positions `0..seq_len`, shape `[1, seq_len, rotary_dim]`.
pub fn rope_cos_sin(cfg: &Config, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let half = dim / 2;
    let inv_freq: Vec<f64> = (0..half)
        .map(|i| cfg.rope_theta.powf(-((2 * i) as f64) / dim as f64))
        .collect();

    let mut cos = vec![0f32; seq_len * dim];
    let mut sin = vec![0f32; seq_len * dim];
    for p in 0..seq_len {
        for (i, f) in inv_freq.iter().enumerate() {
            let angle = p as f64 * f;
            let (s, c) = (angle.sin() as f32, angle.cos() as f32);
            cos[p * dim + i] = c;
            cos[p * dim + half + i] = c;
            sin[p * dim + i] = s;
            sin[p * dim + half + i] = s;
        }
    }
    let cos = Tensor::from_vec(cos, (1, seq_len, dim), device)?;
    let sin = Tensor::from_vec(sin, (1, seq_len, dim), device)?;
    Ok((cos, sin))
}
