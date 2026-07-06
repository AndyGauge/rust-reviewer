//! Stage 4b: greedy generation with **no KV cache** — correctness first. Each
//! new token re-runs the whole verified `full_model_forward` over the entire
//! sequence so far (O(n^2), slow), rather than risk a subtly-wrong cache
//! before the no-cache path is proven. The cache comes in Stage 4c, diffed
//! against this loop token-for-token.
use std::collections::HashMap;

use candle_core::{D, Device, Result, Tensor};

use crate::cache;
use crate::config::Config;
use crate::model::full_model_forward;
use crate::rope::rope_cos_sin;

/// argmax of the logits at the last (only, for a `[1,1,vocab]` decode-step
/// tensor) position, as `u32`.
fn argmax_last(logits: &Tensor) -> Result<u32> {
    let s = logits.dim(1)?;
    logits
        .narrow(1, s - 1, 1)?
        .squeeze(1)?
        .squeeze(0)?
        .to_dtype(candle_core::DType::F32)?
        .argmax(D::Minus1)?
        .to_scalar()
}

/// Greedy-decode from `prompt_ids`, stopping at `max_new_tokens` or the first
/// id in `eos_ids`. Returns the full sequence (prompt + generated).
pub fn greedy_generate(
    w: &HashMap<String, Tensor>,
    cfg: &Config,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
) -> Result<Vec<u32>> {
    let mut ids = prompt_ids.to_vec();
    for _ in 0..max_new_tokens {
        let s = ids.len();
        let input = Tensor::from_vec(ids.clone(), (1, s), device)?;
        let (cos, sin) = rope_cos_sin(cfg, s, device)?;
        let logits = full_model_forward(w, &input, &cos, &sin, cfg)?; // [1, s, vocab]
        let last = logits
            .narrow(1, s - 1, 1)?
            .squeeze(1)?
            .squeeze(0)?
            .to_dtype(candle_core::DType::F32)?; // [vocab]
        let next: u32 = last.argmax(D::Minus1)?.to_scalar()?;
        ids.push(next);
        if eos_ids.contains(&next) {
            break;
        }
    }
    Ok(ids)
}

/// Same greedy decode as [`greedy_generate`], but through the Stage 4c KV /
/// recurrent-state cache: one `cache::prefill` instead of re-running the
/// whole sequence every step. Exists to be diffed token-for-token against
/// [`greedy_generate`] — the cache is only trustworthy once it reproduces the
/// no-cache path exactly.
pub fn greedy_generate_cached(
    w: &HashMap<String, Tensor>,
    cfg: &Config,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    eos_ids: &[u32],
    device: &Device,
) -> Result<Vec<u32>> {
    let mut ids = prompt_ids.to_vec();
    let input = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
    let (mut logits, mut cache) = cache::prefill(w, &input, cfg, device)?;

    for _ in 0..max_new_tokens {
        let next = argmax_last(&logits)?;
        ids.push(next);
        if eos_ids.contains(&next) {
            break;
        }
        logits = cache::decode_step(w, next, &mut cache, cfg, device)?;
    }
    Ok(ids)
}
