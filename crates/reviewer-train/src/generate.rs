//! Stage 4b: greedy generation with **no KV cache** — correctness first. Each
//! new token re-runs the whole verified `full_model_forward` over the entire
//! sequence so far (O(n^2), slow), rather than risk a subtly-wrong cache
//! before the no-cache path is proven. The cache comes in Stage 4c, diffed
//! against this loop token-for-token.
use std::collections::HashMap;

use candle_core::{D, Device, Result, Tensor};

use crate::config::Config;
use crate::model::full_model_forward;
use crate::rope::rope_cos_sin;

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
