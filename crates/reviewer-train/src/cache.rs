//! Stage 4c: the KV / recurrent-state cache. Two kinds of state, because this
//! is a hybrid architecture — attention layers cache K/V like any transformer;
//! DeltaNet layers cache the recurrent matrix state `S` and the causal conv's
//! last `kernel-1` inputs. `prefill` seeds both from the prompt; `decode_step`
//! advances them one token at a time, so generation stops re-running the
//! whole sequence (Stage 4b) and instead does O(1) work per new token.
use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use crate::config::Config;
use crate::mixer::linear;
use crate::model::{
    decoder_layer_full_decode, decoder_layer_full_prefill, decoder_layer_linear_decode,
    decoder_layer_linear_prefill, rmsnorm,
};
use crate::rope::{rope_cos_sin, rope_cos_sin_at};

/// Per-layer cached state — exactly one variant populated, matching
/// `cfg.is_full_attn(i)` for that layer index.
enum LayerCache {
    Attn { k: Tensor, v: Tensor },
    Delta { state: Tensor, conv_tail: Tensor },
}

pub struct Cache {
    layers: Vec<LayerCache>,
    /// Tokens already folded into the cache — the next decode step's RoPE
    /// position, and the count `decode_step` advances by one each call.
    pub len: usize,
}

/// Prefill the prompt, seeding the cache. Returns logits for the *last*
/// position only (all a greedy decoder needs from prefill).
pub fn prefill(w: &HashMap<String, Tensor>, input_ids: &Tensor, cfg: &Config, device: &Device) -> Result<(Tensor, Cache)> {
    let (b, s) = input_ids.dims2()?;
    let embed = &w["model.language_model.embed_tokens.weight"];
    let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
    let mut x = embed.index_select(&ids, 0)?.reshape((b, s, cfg.hidden))?;

    let (cos, sin) = rope_cos_sin(cfg, s, device)?;
    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let prefix = format!("model.language_model.layers.{i}.");
        if cfg.is_full_attn(i) {
            let (nx, k, v) = decoder_layer_full_prefill(w, &x, &cos, &sin, &prefix, cfg)?;
            x = nx;
            layers.push(LayerCache::Attn { k, v });
        } else {
            let (nx, state, conv_tail) = decoder_layer_linear_prefill(w, &x, &prefix, cfg)?;
            x = nx;
            layers.push(LayerCache::Delta { state, conv_tail });
        }
    }

    let x = rmsnorm(&x, &w["model.language_model.norm.weight"], cfg.eps)?;
    let last = x.narrow(1, s - 1, 1)?;
    let logits = linear(&last, &w["lm_head.weight"])?; // [b,1,vocab]
    Ok((logits, Cache { layers, len: s }))
}

/// Decode one new token, advancing the cache. Returns logits `[b,1,vocab]`.
pub fn decode_step(w: &HashMap<String, Tensor>, next_id: u32, cache: &mut Cache, cfg: &Config, device: &Device) -> Result<Tensor> {
    let embed = &w["model.language_model.embed_tokens.weight"];
    let ids = Tensor::from_vec(vec![next_id], (1,), device)?;
    let mut x = embed.index_select(&ids, 0)?.reshape((1, 1, cfg.hidden))?;

    let (cos1, sin1) = rope_cos_sin_at(cfg, cache.len, device)?;
    let cos1 = cos1.to_dtype(x.dtype())?;
    let sin1 = sin1.to_dtype(x.dtype())?;

    for i in 0..cfg.n_layers {
        let prefix = format!("model.language_model.layers.{i}.");
        cache.layers[i] = match &cache.layers[i] {
            LayerCache::Attn { k, v } => {
                let (nx, nk, nv) = decoder_layer_full_decode(w, &x, &cos1, &sin1, &prefix, cfg, k, v)?;
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
    cache.len += 1;
    Ok(logits)
}
