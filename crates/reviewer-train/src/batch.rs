//! Stage 5a: batched inference — left-padded prefill + a growing per-row
//! padding mask through decode. Reuses the exact same prefill/decode building
//! blocks as the single-sequence [`crate::cache`] (this module *is* the
//! `pad_mask`/`extra_mask` parameters those functions were built with —
//! passing `None` there gets back the pre-batching Stage 4c behavior exactly).
//!
//! Left-padding means every row's *last* column is always a real token, so
//! there's never any per-row indexing needed to find "the next-token
//! position" — it's always the last column, uniformly.
use std::collections::HashMap;

use candle_core::{D, DType, Device, Result, Tensor};

use crate::config::Config;
use crate::mixer::linear;
use crate::model::{
    decoder_layer_full_decode, decoder_layer_full_prefill, decoder_layer_linear_decode,
    decoder_layer_linear_prefill, rmsnorm,
};
use crate::rope::{rope_cos_sin_at_rows, rope_cos_sin_rows};

enum LayerCache {
    Attn { k: Tensor, v: Tensor },
    Delta { state: Tensor, conv_tail: Tensor },
}

pub struct BatchCache {
    layers: Vec<LayerCache>,
    /// Frozen additive mask (`[b,1,1,max_len]`, 0.0/-inf) for the *original*
    /// left-padded prefill columns — these stay masked for the life of
    /// generation, since those columns are never removed from the KV cache.
    key_pad_base: Tensor,
    /// Each row's real (non-padded) prompt length — the RoPE position base
    /// for that row's first generated token.
    real_lens: Vec<usize>,
    /// Decode steps taken so far (shared step counter; each row's actual
    /// RoPE position is `real_lens[i] + step`).
    step: usize,
}

/// Left-pad ragged `prompts` to a common length with `pad_id`, run the batched
/// prefill, and seed the cache. Returns logits `[b,1,vocab]` for each row's
/// last (always-real, thanks to left-padding) position.
pub fn prefill_batch(
    w: &HashMap<String, Tensor>,
    prompts: &[Vec<u32>],
    pad_id: u32,
    cfg: &Config,
    device: &Device,
) -> Result<(Tensor, BatchCache)> {
    let b = prompts.len();
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap_or(0);

    let mut ids = Vec::with_capacity(b * max_len);
    let mut key_pad = Vec::with_capacity(b * max_len);
    let mut positions = Vec::with_capacity(b);
    let mut pad_mask_data = Vec::with_capacity(b * max_len);
    for p in prompts {
        let pad_len = max_len - p.len();
        for _ in 0..pad_len {
            ids.push(pad_id);
            key_pad.push(f32::NEG_INFINITY);
            pad_mask_data.push(0f32);
        }
        ids.extend_from_slice(p);
        key_pad.extend(std::iter::repeat_n(0f32, p.len()));
        pad_mask_data.extend(std::iter::repeat_n(1f32, p.len()));
        let row_positions: Vec<usize> = (0..pad_len).map(|_| 0).chain(0..p.len()).collect();
        positions.push(row_positions);
    }

    let input_ids = Tensor::from_vec(ids, (b, max_len), device)?;
    let key_pad_base = Tensor::from_vec(key_pad, (b, 1, 1, max_len), device)?;
    let pad_mask = Tensor::from_vec(pad_mask_data, (b, max_len), device)?;

    // Prefill's own attention needs a *fuller* mask than `key_pad_base`: a
    // padded query row (one that's itself in the left-padded prefix) is
    // causally restricted to keys at or before its own column, which for a
    // padded row are *all* padding — a fully `-inf` row, and softmax of an
    // all-`-inf` row is 0/0 = NaN. Masking-by-multiply in the DeltaNet layers
    // doesn't clean that up (`NaN * 0 == NaN`), so it survives into the next
    // conv1d and contaminates nearby *real* tokens. Fix: every row keeps its
    // own diagonal unmasked (trivial self-attention) — harmless, since we
    // never read a padded position's output, and it guarantees at least one
    // finite entry per row.
    let mut full_mask = vec![0f32; b * max_len * max_len];
    for (row, p) in prompts.iter().enumerate() {
        let pad_len = max_len - p.len();
        for i in 0..max_len {
            for j in 0..pad_len {
                if j != i {
                    full_mask[row * max_len * max_len + i * max_len + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    let full_mask = Tensor::from_vec(full_mask, (b, 1, max_len, max_len), device)?;

    let embed = &w["model.language_model.embed_tokens.weight"];
    let flat_ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
    let mut x = embed.index_select(&flat_ids, 0)?.reshape((b, max_len, cfg.hidden))?;

    let (cos, sin) = rope_cos_sin_rows(cfg, &positions, device)?;
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;
    let key_pad_base = key_pad_base.to_dtype(x.dtype())?;
    let full_mask = full_mask.to_dtype(x.dtype())?;
    let pad_mask = pad_mask.to_dtype(x.dtype())?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("model.language_model.layers.{i}.");
        if cfg.is_full_attn(i) {
            let (nx, k, v) = decoder_layer_full_prefill(w, &x, &cos, &sin, &prefix, cfg, Some(&full_mask))?;
            x = nx;
            layers.push(LayerCache::Attn { k, v });
        } else {
            let (nx, state, conv_tail) = decoder_layer_linear_prefill(w, &x, &prefix, cfg, Some(&pad_mask))?;
            x = nx;
            layers.push(LayerCache::Delta { state, conv_tail });
        }
    }

    let x = rmsnorm(&x, &w["model.language_model.norm.weight"], cfg.eps)?;
    let last = x.narrow(1, max_len - 1, 1)?;
    let logits = linear(&last, &w["lm_head.weight"])?; // [b,1,vocab]

    let real_lens = prompts.iter().map(|p| p.len()).collect();
    Ok((logits, BatchCache { layers, key_pad_base, real_lens, step: 0 }))
}

/// Decode one new token per row. `next_ids[i]` feeds row `i`'s cache — every
/// row advances every step regardless of whether it's already hit EOS (the
/// caller just stops recording that row's output; the batched math doesn't
/// support skipping a subset of rows without re-batching, so it's simplest
/// to keep advancing and truncate the text afterward).
pub fn decode_step_batch(
    w: &HashMap<String, Tensor>,
    next_ids: &[u32],
    cache: &mut BatchCache,
    cfg: &Config,
    device: &Device,
) -> Result<Tensor> {
    let b = next_ids.len();
    let ids = Tensor::from_vec(next_ids.to_vec(), (b,), device)?;
    let embed = &w["model.language_model.embed_tokens.weight"];
    let mut x = embed.index_select(&ids, 0)?.reshape((b, 1, cfg.hidden))?;

    let positions: Vec<usize> = cache.real_lens.iter().map(|&rl| rl + cache.step).collect();
    let (cos1, sin1) = rope_cos_sin_at_rows(cfg, &positions, device)?;
    let cos1 = cos1.to_dtype(x.dtype())?;
    let sin1 = sin1.to_dtype(x.dtype())?;

    // The new token this call appends makes the cache one column longer than
    // `cache.step` reflects (that's only bumped at the end) — the mask has to
    // cover that new column too, or it's one short of the post-append K/V.
    let zeros = Tensor::zeros((b, 1, 1, cache.step + 1), cache.key_pad_base.dtype(), device)?;
    let mask = Tensor::cat(&[&cache.key_pad_base, &zeros], 3)?;

    for i in 0..cfg.n_layers {
        let prefix = format!("model.language_model.layers.{i}.");
        cache.layers[i] = match &cache.layers[i] {
            LayerCache::Attn { k, v } => {
                let (nx, nk, nv) = decoder_layer_full_decode(w, &x, &cos1, &sin1, &prefix, cfg, k, v, Some(&mask))?;
                x = nx;
                LayerCache::Attn { k: nk, v: nv }
            }
            LayerCache::Delta { state, conv_tail } => {
                let (nx, ns, nc) = decoder_layer_linear_decode(w, &x, &prefix, cfg, state, conv_tail)?;
                x = nx;
                LayerCache::Delta { state: ns, conv_tail: nc }
            }
        };
    }

    let x = rmsnorm(&x, &w["model.language_model.norm.weight"], cfg.eps)?;
    let logits = linear(&x, &w["lm_head.weight"])?; // [b,1,vocab]
    cache.step += 1;
    Ok(logits)
}

/// argmax of each row's logits, `[b,1,vocab]` -> `Vec<u32>` length `b`.
fn argmax_batch(logits: &Tensor) -> Result<Vec<u32>> {
    logits.squeeze(1)?.to_dtype(DType::F32)?.argmax(D::Minus1)?.to_vec1()
}

/// Greedy-decode a ragged batch of prompts, left-padded to a common length.
/// Each row stops recording once it hits an id in `eos_ids` (but keeps
/// decoding structurally, per [`decode_step_batch`]'s note). Returns one
/// generated-id sequence per row (prompt not included).
pub fn greedy_generate_batch(
    w: &HashMap<String, Tensor>,
    cfg: &Config,
    prompts: &[Vec<u32>],
    pad_id: u32,
    max_new_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
) -> Result<Vec<Vec<u32>>> {
    let b = prompts.len();
    let (mut logits, mut cache) = prefill_batch(w, prompts, pad_id, cfg, device)?;

    let mut outputs: Vec<Vec<u32>> = vec![Vec::new(); b];
    let mut finished = vec![false; b];
    for _ in 0..max_new_tokens {
        let next_ids = argmax_batch(&logits)?;
        for i in 0..b {
            if !finished[i] {
                outputs[i].push(next_ids[i]);
                if eos_ids.contains(&next_ids[i]) {
                    finished[i] = true;
                }
            }
        }
        if finished.iter().all(|f| *f) {
            break;
        }
        logits = decode_step_batch(w, &next_ids, &mut cache, cfg, device)?;
    }
    Ok(outputs)
}
