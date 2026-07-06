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

fn inv_freq(cfg: &Config) -> Vec<f64> {
    let dim = cfg.rotary_dim();
    (0..dim / 2).map(|i| cfg.rope_theta.powf(-((2 * i) as f64) / dim as f64)).collect()
}

/// One position's cos/sin row (length `rotary_dim`) — shared by the
/// whole-sequence table (prefill) and the single-position table (decode).
fn cos_sin_row(dim: usize, inv_freq: &[f64], pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let mut cos = vec![0f32; dim];
    let mut sin = vec![0f32; dim];
    for (i, f) in inv_freq.iter().enumerate() {
        let angle = pos as f64 * f;
        let (s, c) = (angle.sin() as f32, angle.cos() as f32);
        cos[i] = c;
        cos[half + i] = c;
        sin[i] = s;
        sin[half + i] = s;
    }
    (cos, sin)
}

/// cos/sin tables for positions `0..seq_len`, shape `[1, seq_len, rotary_dim]`.
/// Used for prefill, where every position from 0 needs a row.
pub fn rope_cos_sin(cfg: &Config, seq_len: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let freq = inv_freq(cfg);
    let mut cos = Vec::with_capacity(seq_len * dim);
    let mut sin = Vec::with_capacity(seq_len * dim);
    for p in 0..seq_len {
        let (c, s) = cos_sin_row(dim, &freq, p);
        cos.extend(c);
        sin.extend(s);
    }
    let cos = Tensor::from_vec(cos, (1, seq_len, dim), device)?;
    let sin = Tensor::from_vec(sin, (1, seq_len, dim), device)?;
    Ok((cos, sin))
}

/// cos/sin for a single absolute position, shape `[1, 1, rotary_dim]`. Used
/// for decode: the new token's position is `cache.len`, not 0, so the table
/// can't just be "the first row of `rope_cos_sin`" — it has to be computed at
/// the right absolute offset.
pub fn rope_cos_sin_at(cfg: &Config, pos: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let (cos, sin) = cos_sin_row(dim, &inv_freq(cfg), pos);
    let cos = Tensor::from_vec(cos, (1, 1, dim), device)?;
    let sin = Tensor::from_vec(sin, (1, 1, dim), device)?;
    Ok((cos, sin))
}
