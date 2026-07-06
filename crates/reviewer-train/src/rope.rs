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

/// Batched prefill positions: one independent position sequence per row,
/// shape `[b, max_len, rotary_dim]`. Left-padded batches need this instead of
/// [`rope_cos_sin`] because position is *row-relative to that row's own real
/// content*, not the shared column index — the reference computes it as
/// `cumsum(attention_mask) - 1` per row (0,1,2,... starting at that row's
/// first real token, whatever column it lands on); padded columns get an
/// arbitrary placeholder here since the attention/DeltaNet padding masks zero
/// their contribution regardless of what RoPE did to them.
pub fn rope_cos_sin_rows(cfg: &Config, positions: &[Vec<usize>], device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let freq = inv_freq(cfg);
    let b = positions.len();
    let max_len = positions.first().map_or(0, |row| row.len());
    let mut cos = Vec::with_capacity(b * max_len * dim);
    let mut sin = Vec::with_capacity(b * max_len * dim);
    for row in positions {
        for &p in row {
            let (c, s) = cos_sin_row(dim, &freq, p);
            cos.extend(c);
            sin.extend(s);
        }
    }
    let cos = Tensor::from_vec(cos, (b, max_len, dim), device)?;
    let sin = Tensor::from_vec(sin, (b, max_len, dim), device)?;
    Ok((cos, sin))
}

/// Batched decode positions: one absolute position per row (rows generally
/// disagree, since ragged prompts finished prefill at different row-relative
/// lengths), shape `[b, 1, rotary_dim]`.
pub fn rope_cos_sin_at_rows(cfg: &Config, positions: &[usize], device: &Device) -> Result<(Tensor, Tensor)> {
    let dim = cfg.rotary_dim();
    let freq = inv_freq(cfg);
    let b = positions.len();
    let mut cos = Vec::with_capacity(b * dim);
    let mut sin = Vec::with_capacity(b * dim);
    for &p in positions {
        let (c, s) = cos_sin_row(dim, &freq, p);
        cos.extend(c);
        sin.extend(s);
    }
    let cos = Tensor::from_vec(cos, (b, 1, dim), device)?;
    let sin = Tensor::from_vec(sin, (b, 1, dim), device)?;
    Ok((cos, sin))
}
