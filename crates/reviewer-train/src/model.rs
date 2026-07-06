//! Higher-level model pieces in candle: RMSNorm, the SwiGLU MLP, and the DeltaNet
//! decoder layer that assembles them around the verified mixer. Standard
//! transformer machinery — the risk is in wiring, which the layer oracle checks.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{D, DType, Device, Result, Tensor, safetensors};
use candle_nn::ops::softmax;

use crate::config::Config;
use crate::mixer::{linear, mixer_forward, sigmoid, silu};

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
pub fn decoder_layer_linear(w: &HashMap<String, Tensor>, x: &Tensor, prefix: &str, cfg: &Config) -> Result<Tensor> {
    let h = rmsnorm(x, &w[&format!("{prefix}input_layernorm.weight")], cfg.eps)?;
    let h = mixer_forward(w, &h, &format!("{prefix}linear_attn."), cfg)?;
    let x = x.add(&h)?;

    let h = rmsnorm(&x, &w[&format!("{prefix}post_attention_layernorm.weight")], cfg.eps)?;
    let h = mlp(w, &format!("{prefix}mlp."), &h)?;
    x.add(&h)
}

/// Causal additive mask `[s,s]`: 0 on/below the diagonal, -inf above.
fn causal_mask(s: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; s * s];
    for i in 0..s {
        for j in (i + 1)..s {
            data[i * s + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (s, s), device)
}

/// Partial rotary embedding: rotate the first `cos.dim(-1)` dims of each head,
/// pass the rest through. `x`: `[b,h,s,hd]`, `cos`/`sin`: `[1,s,rd]` (rd ≤ hd).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(1)?; // [1,1,s,rd]
    let sin = sin.unsqueeze(1)?;
    let rd = cos.dim(D::Minus1)?;
    let hd = x.dim(D::Minus1)?;
    let x_rot = x.narrow(D::Minus1, 0, rd)?;
    let x_pass = x.narrow(D::Minus1, rd, hd - rd)?.contiguous()?;
    let half = rd / 2;
    let x1 = x_rot.narrow(D::Minus1, 0, half)?;
    let x2 = x_rot.narrow(D::Minus1, half, rd - half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?; // rotate_half
    let rotated = x_rot.broadcast_mul(&cos)?.add(&rot.broadcast_mul(&sin)?)?;
    Tensor::cat(&[&rotated, &x_pass], D::Minus1)
}

/// Repeat each kv head `rep` times along the head dim (GQA expansion).
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (b, nkv, s, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .broadcast_as((b, nkv, rep, s, hd))?
        .contiguous()?
        .reshape((b, nkv * rep, s, hd))
}

/// Gated GQA attention: q/k per-head RMSNorm, partial RoPE, causal softmax, and
/// a `sigmoid` gate on the output (the `q_proj` emits query+gate).
fn attention(
    w: &HashMap<String, Tensor>,
    prefix: &str,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    cfg: &Config,
) -> Result<Tensor> {
    let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let (b, s, _) = x.dims3()?;

    let qg = linear(x, &w[&format!("{prefix}q_proj.weight")])?.reshape((b, s, nh, 2 * hd))?;
    let query = qg.narrow(3, 0, hd)?.contiguous()?; // [b,s,nh,hd]
    let gate = qg.narrow(3, hd, hd)?.contiguous()?.reshape((b, s, nh * hd))?;

    let query = rmsnorm(&query, &w[&format!("{prefix}q_norm.weight")], cfg.eps)?;
    let query = query.transpose(1, 2)?.contiguous()?; // [b,nh,s,hd]
    let key = linear(x, &w[&format!("{prefix}k_proj.weight")])?.reshape((b, s, nkv, hd))?;
    let key = rmsnorm(&key, &w[&format!("{prefix}k_norm.weight")], cfg.eps)?;
    let key = key.transpose(1, 2)?.contiguous()?; // [b,nkv,s,hd]
    let value = linear(x, &w[&format!("{prefix}v_proj.weight")])?
        .reshape((b, s, nkv, hd))?
        .transpose(1, 2)?
        .contiguous()?;

    let query = apply_rope(&query, cos, sin)?;
    let key = apply_rope(&key, cos, sin)?;
    let key = repeat_kv(&key, nh / nkv)?;
    let value = repeat_kv(&value, nh / nkv)?;

    let scale = (hd as f64).powf(-0.5);
    let attn = query.matmul(&key.transpose(2, 3)?.contiguous()?)?.affine(scale, 0.0)?; // [b,nh,s,s]
    let mask = causal_mask(s, x.device())?.to_dtype(attn.dtype())?;
    let attn = attn.broadcast_add(&mask)?;
    let attn = softmax(&attn, D::Minus1)?;
    let out = attn.matmul(&value)?; // [b,nh,s,hd]
    let out = out.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?; // [b,s,4096]
    let out = out.mul(&sigmoid(&gate)?)?; // gated
    linear(&out, &w[&format!("{prefix}o_proj.weight")])
}

/// A full-attention decoder layer (pre-norm attention + residual, pre-norm MLP +
/// residual). Needs the RoPE `cos`/`sin` for its position.
pub fn decoder_layer_full(
    w: &HashMap<String, Tensor>,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    prefix: &str,
    cfg: &Config,
) -> Result<Tensor> {
    let h = rmsnorm(x, &w[&format!("{prefix}input_layernorm.weight")], cfg.eps)?;
    let h = attention(w, &format!("{prefix}self_attn."), &h, cos, sin, cfg)?;
    let x = x.add(&h)?;
    let h = rmsnorm(&x, &w[&format!("{prefix}post_attention_layernorm.weight")], cfg.eps)?;
    let h = mlp(w, &format!("{prefix}mlp."), &h)?;
    x.add(&h)
}

/// Load the language-model + lm_head weights from a directory of sharded
/// safetensors onto `device`, cast to f32 (the vision tower is skipped).
pub fn load_weights(dir: &Path, device: &Device, dtype: DType) -> Result<HashMap<String, Tensor>> {
    let mut w = HashMap::new();
    for entry in std::fs::read_dir(dir).map_err(candle_core::Error::wrap)? {
        let path = entry.map_err(candle_core::Error::wrap)?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        for (k, v) in safetensors::load(&path, device)? {
            if k.starts_with("model.language_model.") || k.starts_with("lm_head.") {
                w.insert(k, v.to_dtype(dtype)?);
            }
        }
    }
    Ok(w)
}

/// Merge a PEFT LoRA adapter into the loaded base weights, in place. For each
/// base weight `W[out,in]` with adapter `lora_A[r,in]` and `lora_B[out,r]`:
/// `W += scale · (B @ A)`, `scale = alpha / r`. PEFT keys are
/// `base_model.model.{stem}.lora_{A,B}.weight`; weights without a pair (embed,
/// norms, lm_head) are left as the base.
pub fn apply_lora(
    w: &mut HashMap<String, Tensor>,
    adapter: &Path,
    scale: f64,
    dtype: DType,
) -> Result<()> {
    let dev = w.values().next().unwrap().device().clone();
    let a = safetensors::load(adapter, &dev)?;
    let base_keys: Vec<String> = w.keys().cloned().collect();
    let mut merged = 0usize;
    for k in base_keys {
        let Some(stem) = k.strip_suffix(".weight") else {
            continue;
        };
        let ka = format!("base_model.model.{stem}.lora_A.weight");
        let kb = format!("base_model.model.{stem}.lora_B.weight");
        if let (Some(la), Some(lb)) = (a.get(&ka), a.get(&kb)) {
            let la = la.to_dtype(dtype)?; // [r, in]
            let lb = lb.to_dtype(dtype)?; // [out, r]
            let delta = lb.matmul(&la)?.affine(scale, 0.0)?; // [out, in]
            let merged_w = w[&k].add(&delta)?;
            w.insert(k, merged_w);
            merged += 1;
        }
    }
    println!("  merged {merged} LoRA weights (scale {scale})");
    Ok(())
}

/// The full model: embed → 32 hybrid layers (3 DeltaNet : 1 attention) → final
/// norm → lm_head. Returns logits `[B, S, vocab]`.
pub fn full_model_forward(
    w: &HashMap<String, Tensor>,
    input_ids: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    cfg: &Config,
) -> Result<Tensor> {
    let (b, s) = input_ids.dims2()?;
    let embed = &w["model.language_model.embed_tokens.weight"]; // [vocab, hidden]
    let ids = input_ids.flatten_all()?.to_dtype(DType::U32)?;
    let mut x = embed.index_select(&ids, 0)?.reshape((b, s, cfg.hidden))?;

    // RoPE tables must match the model's dtype (bf16 for the 27B).
    let cos = &cos.to_dtype(x.dtype())?;
    let sin = &sin.to_dtype(x.dtype())?;

    for i in 0..cfg.n_layers {
        let prefix = format!("model.language_model.layers.{i}.");
        x = if cfg.is_full_attn(i) {
            decoder_layer_full(w, &x, cos, sin, &prefix, cfg)?
        } else {
            decoder_layer_linear(w, &x, &prefix, cfg)?
        };
    }

    let x = rmsnorm(&x, &w["model.language_model.norm.weight"], cfg.eps)?;
    linear(&x, &w["lm_head.weight"]) // [b, s, vocab]
}
