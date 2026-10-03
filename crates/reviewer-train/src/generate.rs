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
    let t0 = std::time::Instant::now();
    let (mut logits, mut cache) = cache::prefill(w, &input, cfg, device)?;

    let mut prefill = None;
    let mut prefill_delta_ms = 0.0;
    for _ in 0..max_new_tokens {
        let next = argmax_last(&logits)?; // reads the logits back, so the GPU is done here
        if prefill.is_none() {
            prefill = Some(t0.elapsed());
            prefill_delta_ms = crate::delta::take_delta_ms();
        }
        ids.push(next);
        if eos_ids.contains(&next) {
            break;
        }
        logits = cache::decode_step(w, next, &mut cache, cfg, device)?;
    }
    let total = t0.elapsed();
    let prefill = prefill.unwrap_or(total);
    let decoded = ids.len() - prompt_ids.len();
    let decode_s = (total - prefill).as_secs_f64();
    let decode_delta_ms = crate::delta::take_delta_ms();
    if decode_delta_ms + prefill_delta_ms > 0.0 {
        eprintln!("recurrence: prefill {prefill_delta_ms:.0} ms, decode {decode_delta_ms:.0} ms (of the times below)");
    }
    eprintln!(
        "timing: prefill {:.0} ms ({} prompt tokens, {:.0} tok/s) | decode {:.2} s for {} tokens ({:.2} tok/s)",
        prefill.as_secs_f64() * 1e3,
        prompt_ids.len(),
        prompt_ids.len() as f64 / prefill.as_secs_f64(),
        decode_s,
        decoded.saturating_sub(1),
        decoded.saturating_sub(1) as f64 / decode_s.max(1e-9),
    );
    Ok(ids)
}
