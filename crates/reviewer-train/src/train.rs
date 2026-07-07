//! Path A, the part that never got built until now: an all-Rust (candle) LoRA
//! SFT trainer for the design-review model. The forward pass is the same
//! verified `model::full_model_forward` used everywhere else; training adds three
//! things on top:
//!
//! 1. **LoRA as `Var`s.** Base weights stay frozen constant `Tensor`s. For each
//!    adapted linear we hold trainable `lora_A [r,in]` and `lora_B [out,r]`, and
//!    build the *effective* weight `W + (α/r)·(B·A)` as a graph node each step —
//!    so `loss.backward()` reaches only A and B, and we get to reuse the entire
//!    forward unchanged.
//! 2. **Masked SFT loss.** Standard next-token cross-entropy, supervised only on
//!    the assistant completion (the prompt is masked out).
//! 3. **AdamW.** Over the LoRA `Var`s.
//!
//! The one thing that had to be proven first — that candle's autograd backprops
//! through the sequential Gated DeltaNet recurrence — is verified by
//! `delta::tests::grads_flow_through_recurrence`.

use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var, D};
use candle_nn::{Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use serde::Deserialize;

use crate::chat;
use crate::config::Config;
use crate::model;
use crate::rope;

const IGNORE: i64 = -100;

/// Which linears to adapt. The DeltaNet mixer projections and the full-attention
/// q/k/v/o are the "attention-equivalent" matrices; `all` adds the (much larger)
/// MLP projections, at a real memory cost through the backward graph.
fn is_target(key: &str, all_linears: bool) -> bool {
    if !key.starts_with("model.language_model.layers.") || !key.ends_with(".weight") {
        return false;
    }
    let attn = [
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ];
    let mlp = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"];
    attn.iter().any(|m| key.contains(m)) || (all_linears && mlp.iter().any(|m| key.contains(m)))
}

/// One adapted linear: the frozen weight key plus its trainable low-rank pair.
struct LoraLayer {
    weight_key: String,
    a: Var, // [r, in]
    b: Var, // [out, r]
}

struct LoraSet {
    layers: Vec<LoraLayer>,
    scale: f64,
}

impl LoraSet {
    /// Initialise LoRA the standard way: `A` small-random, `B` zero — so the
    /// initial delta is exactly zero and training starts from the base model.
    fn init(w: &HashMap<String, Tensor>, all_linears: bool, r: usize, alpha: f64, dev: &Device) -> Result<Self> {
        let mut keys: Vec<&String> = w.keys().filter(|k| is_target(k, all_linears)).collect();
        keys.sort();
        let mut layers = Vec::new();
        for k in keys {
            let (out, inp) = w[k].dims2().with_context(|| format!("target {k} is not 2-D"))?;
            let a = Var::from_tensor(&Tensor::randn(0f32, 0.01f32, (r, inp), dev)?)?;
            let b = Var::from_tensor(&Tensor::zeros((out, r), DType::F32, dev)?)?;
            layers.push(LoraLayer { weight_key: k.clone(), a, b });
        }
        Ok(Self { layers, scale: alpha / r as f64 })
    }

    fn vars(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|l| [l.a.clone(), l.b.clone()]).collect()
    }

    /// Base map with every adapted weight replaced by `W + scale·(B·A)`, cast back
    /// to the base dtype. A fresh graph each call (backward consumes it).
    fn effective_weights(&self, base: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut w = base.clone(); // Arc clones; cheap
        for l in &self.layers {
            let delta = l.b.matmul(&l.a.as_tensor())?.affine(self.scale, 0.0)?; // [out,in] f32
            let base_w = &base[&l.weight_key];
            let eff = base_w.add(&delta.to_dtype(base_w.dtype())?)?;
            w.insert(l.weight_key.clone(), eff);
        }
        Ok(w)
    }
}

/// One tokenized SFT example: the full token stream and where the supervised
/// completion begins (everything before `prompt_len` is masked).
struct Example {
    ids: Vec<u32>,
    prompt_len: usize,
}

#[derive(Deserialize)]
struct RawRecord {
    messages: Vec<RawMsg>,
}
#[derive(Deserialize)]
struct RawMsg {
    role: String,
    content: String,
}

fn load_examples(paths: &[PathBuf], tok: &tokenizers::Tokenizer, max_seq: usize) -> Result<Vec<Example>> {
    let im_end = "<|im_end|>";
    let mut out = Vec::new();
    let (mut skipped, mut truncated) = (0usize, 0usize);
    for path in paths {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let rec: RawRecord = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(_) => { skipped += 1; continue; }
            };
            let find = |role: &str| rec.messages.iter().find(|m| m.role == role).map(|m| m.content.as_str());
            let (Some(user), Some(asst)) = (find("user"), find("assistant")) else { skipped += 1; continue; };
            let system = find("system").unwrap_or(reviewer_core::SYSTEM);

            let prompt = chat::render_prompt(system, user);
            let prompt_ids = chat::encode(tok, &prompt)?;
            let comp_ids = chat::encode(tok, &format!("{asst}{im_end}"))?;
            if comp_ids.is_empty() { skipped += 1; continue; }

            let mut prompt_ids = prompt_ids;
            // If the pair is too long, front-truncate the prompt (keep the hunk
            // tail nearest the answer + the whole completion).
            let total = prompt_ids.len() + comp_ids.len();
            if total > max_seq {
                let drop = total - max_seq;
                if drop >= prompt_ids.len() { skipped += 1; continue; } // completion alone too long
                prompt_ids.drain(0..drop);
                truncated += 1;
            }
            let prompt_len = prompt_ids.len();
            let mut ids = prompt_ids;
            ids.extend_from_slice(&comp_ids);
            out.push(Example { ids, prompt_len });
        }
    }
    eprintln!("  loaded {} examples ({} truncated, {} skipped)", out.len(), truncated, skipped);
    Ok(out)
}

/// Masked next-token cross-entropy for one example.
fn sft_loss(w: &HashMap<String, Tensor>, ex: &Example, cfg: &Config, dev: &Device) -> Result<Tensor> {
    let s = ex.ids.len();
    let ids = Tensor::from_slice(&ex.ids, (1, s), dev)?;
    let (cos, sin) = rope::rope_cos_sin(cfg, s, dev)?;
    let logits = model::full_model_forward(w, &ids, &cos, &sin, cfg)?; // [1,s,vocab]
    let logits = logits.to_dtype(DType::F32)?.squeeze(0)?; // [s,vocab] — f32 for a stable loss

    // Position t predicts token t+1; supervise only where t+1 is in the completion.
    let pred = logits.narrow(0, 0, s - 1)?; // [s-1, vocab]
    let valid: Vec<u32> = ((ex.prompt_len.saturating_sub(1))..(s - 1)).map(|t| t as u32).collect();
    if valid.is_empty() {
        anyhow::bail!("example has no supervised positions");
    }
    let vidx = Tensor::from_slice(&valid, valid.len(), dev)?;
    let pred_v = pred.index_select(&vidx, 0)?; // [m, vocab]
    let tgt: Vec<u32> = valid.iter().map(|&t| ex.ids[t as usize + 1]).collect();
    let tgt = Tensor::from_slice(&tgt, tgt.len(), dev)?;

    let logp = candle_nn::ops::log_softmax(&pred_v, D::Minus1)?;
    let picked = logp.gather(&tgt.unsqueeze(1)?, 1)?; // [m,1]
    let loss = picked.mean_all()?.neg()?;
    let _ = IGNORE; // masking is done by index selection above; sentinel kept for clarity
    Ok(loss)
}

/// Write the trained LoRA as a PEFT-format adapter (loadable by `apply_lora` and
/// by vLLM): `adapter_model.safetensors` + `adapter_config.json`.
fn save_adapter(dir: &Path, lora: &LoraSet, r: usize, alpha: f64, all_linears: bool) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut modules = std::collections::BTreeSet::new();
    for l in &lora.layers {
        let stem = l.weight_key.strip_suffix(".weight").unwrap();
        tensors.insert(format!("base_model.model.{stem}.lora_A.weight"), l.a.as_tensor().clone());
        tensors.insert(format!("base_model.model.{stem}.lora_B.weight"), l.b.as_tensor().clone());
        if let Some(m) = stem.rsplit('.').next() {
            modules.insert(m.to_string());
        }
    }
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    let cfg = serde_json::json!({
        "peft_type": "LORA",
        "auto_mapping": null,
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": modules.into_iter().collect::<Vec<_>>(),
        "inference_mode": true,
        "_note": format!("all_linears={all_linears}; trained by reviewer-train (candle)"),
    });
    std::fs::write(dir.join("adapter_config.json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    weights: &Path,
    tokenizer: &Path,
    data: &[PathBuf],
    out: &Path,
    config: Option<&Path>,
    rank: usize,
    alpha: f64,
    lr: f64,
    max_seq: usize,
    epochs: usize,
    limit: Option<usize>,
    all_linears: bool,
    bf16: bool,
    log_every: usize,
    save_every: usize,
) -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    eprintln!("device: {dev:?}");
    let cfg = match config {
        Some(p) => Config::from_json(p)?,
        None => {
            let j = weights.join("config.json");
            if j.exists() { Config::from_json(&j)? } else { Config::qwen9b() }
        }
    };
    let base_dtype = if bf16 { DType::BF16 } else { DType::F32 };

    eprintln!("loading base weights ({base_dtype:?}) …");
    let mut w = model::load_weights(weights, &dev, base_dtype)?;
    // Drop everything the text forward doesn't touch (vision tower, MTP head) to
    // free memory for the backward graph.
    w.retain(|k, _| k.starts_with("model.language_model.") || k == "lm_head.weight");
    eprintln!("  {} base tensors kept", w.len());

    let lora = LoraSet::init(&w, all_linears, rank, alpha, &dev)?;
    eprintln!("LoRA: {} adapted linears, r={rank}, alpha={alpha}, scale={:.3}", lora.layers.len(), lora.scale);

    let tok = chat::load_tokenizer(tokenizer)?;
    eprintln!("loading data …");
    let mut examples = load_examples(data, &tok, max_seq)?;
    if let Some(n) = limit {
        examples.truncate(n);
        eprintln!("  limited to {} examples", examples.len());
    }
    anyhow::ensure!(!examples.is_empty(), "no training examples");

    let mut opt = candle_nn::AdamW::new(
        lora.vars(),
        ParamsAdamW { lr, ..Default::default() },
    )?;

    let mut rng = rand::thread_rng();
    let mut step = 0usize;
    let mut running = 0f64;
    let t0 = std::time::Instant::now();
    let total_steps = examples.len() * epochs;
    for epoch in 0..epochs {
        examples.shuffle(&mut rng);
        for ex in &examples {
            let loss = sft_loss(&lora.effective_weights(&w)?, ex, &cfg, &dev)?;
            opt.backward_step(&loss)?;
            let l = loss.to_scalar::<f32>()? as f64;
            running += l;
            step += 1;
            if step % log_every == 0 {
                let avg = running / log_every as f64;
                running = 0.0;
                let sps = step as f64 / t0.elapsed().as_secs_f64();
                eprintln!(
                    "  epoch {epoch} · step {step}/{total_steps} · loss {avg:.4} · {sps:.2} steps/s",
                );
                std::io::stderr().flush().ok();
            }
            if save_every > 0 && step % save_every == 0 {
                save_adapter(out, &lora, rank, alpha, all_linears)?;
                eprintln!("  checkpoint -> {}", out.display());
            }
        }
    }
    save_adapter(out, &lora, rank, alpha, all_linears)?;
    eprintln!(
        "done: {step} steps in {:.1} min -> {}",
        t0.elapsed().as_secs_f64() / 60.0,
        out.display(),
    );
    Ok(())
}
