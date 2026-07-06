//! Path A: an all-Rust (candle) LoRA trainer for the design-review model.
//!
//! Stage 1 is an *architecture port*: candle has no Gated DeltaNet, so we
//! translate the reference (transformers `Qwen3_5MoeGatedDeltaNet`) into candle
//! and verify the forward pass matches, layer by layer, against an oracle dumped
//! by `oracle_dump.py`. Training (LoRA + SFT loop) comes only after the forward
//! pass is proven correct.
//!
//! This skeleton just loads and inspects the oracle — the first Rust code to
//! touch both candle and the ground-truth tensors. The model port builds on it.

mod batch;
mod cache;
mod chat;
mod config;
mod delta;
mod generate;
mod mixer;
mod model;
mod rope;

use anyhow::{Context, Result};
use config::Config;
use candle_core::{Device, safetensors};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Path A candle trainer — architecture port + LoRA SFT (WIP)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Load an oracle safetensors dump and print its tensors (Stage 1 sanity).
    Inspect {
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify the candle Gated DeltaNet recurrence against the synthetic oracle.
    VerifyDelta {
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify the full candle DeltaNet mixer against the layer-0 oracle.
    VerifyMixer {
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify the candle DeltaNet decoder layer against the layer oracle.
    VerifyLayer {
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify the candle full-attention decoder layer against its oracle.
    VerifyAttn {
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify the full candle model's logits against the whole-model oracle.
    VerifyModel {
        /// oracle safetensors (input_ids, cos/sin, logits, argmax).
        #[arg(long)]
        oracle: PathBuf,
        /// Directory of the model's sharded safetensors weights.
        #[arg(long)]
        weights: PathBuf,
        /// Model config.json (dims). Defaults to the 9B if omitted.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Load weights in bf16 (for the 27B, which doesn't fit in f32).
        #[arg(long)]
        bf16: bool,
        /// Merge a PEFT LoRA adapter_model.safetensors before running.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// LoRA scale (alpha/r); epoch-1 adapter is 64/32 = 2.0.
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
    },
    /// Write the sample (system, user) chat fixture as JSON — the single
    /// source of truth fed to both the Rust template renderer and the Python
    /// oracle script, so they tokenize identical strings.
    DumpChatFixture {
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify our hardcoded chat-template + tokenizer against a Python oracle
    /// (`train/chat_template_oracle.py`) built from the same fixture (Stage 4a).
    VerifyChatTemplate {
        /// The (system, user) JSON from `dump-chat-fixture`.
        #[arg(long)]
        fixture: PathBuf,
        /// `tokenizer.json` from the HF snapshot.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Oracle safetensors (one "input_ids" tensor) from the Python side.
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Stage 4b: greedy-generate one reviewer comment for a (system, user)
    /// fixture, no KV cache (re-runs the full forward every new token).
    Generate {
        /// The (system, user) JSON from `dump-chat-fixture`.
        #[arg(long)]
        fixture: PathBuf,
        /// `tokenizer.json` from the HF snapshot.
        #[arg(long)]
        tokenizer: PathBuf,
        /// Directory of the model's sharded safetensors weights.
        #[arg(long)]
        weights: PathBuf,
        /// Model config.json (dims). Defaults to the 9B if omitted.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Load weights in bf16 (for the 27B, which doesn't fit in f32).
        #[arg(long)]
        bf16: bool,
        /// Merge a PEFT LoRA adapter_model.safetensors before running.
        #[arg(long)]
        adapter: Option<PathBuf>,
        /// LoRA scale (alpha/r); epoch-1 adapter is 64/32 = 2.0.
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
        /// Use the Stage 4b no-cache loop (O(n^2), for comparison/debugging)
        /// instead of the Stage 4c KV/state cache (the default — this is the
        /// whole point of 4c: not re-running the full sequence every token).
        #[arg(long)]
        no_cache: bool,
    },
    /// Verify greedy generation (no KV cache) against a Python greedy-decode
    /// oracle (`train/greedy_oracle.py`, `do_sample=False`) on the same
    /// fixture — token-for-token, for as many tokens as the oracle generated.
    VerifyGenerate {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        bf16: bool,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
        /// Oracle safetensors: "ids" (prompt+generated) and "prompt_len".
        #[arg(long)]
        oracle: PathBuf,
    },
    /// Verify `rope::rope_cos_sin`'s self-computed table against the real
    /// `rotary_emb` module's cos/sin (from `train/step_oracle.py`), with no
    /// model load — isolates a RoPE bug from ordinary bf16 forward-pass drift.
    VerifyRope {
        /// Oracle safetensors with "cos"/"sin" (shape `[1,s,rd]` or `[s,rd]`).
        #[arg(long)]
        oracle: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Stage 4c: verify the KV/recurrent-state cache by diffing
    /// `generate::greedy_generate_cached` against the Stage 4b no-cache
    /// `greedy_generate` on the same fixture — both Rust, both candle, so
    /// unlike `verify-generate` this should match exactly, not "up to a bf16
    /// tie": any divergence here is the cache, not cross-framework drift.
    VerifyKvCache {
        #[arg(long)]
        fixture: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        bf16: bool,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
    },
    /// Stage 5a/5b: verify the batched (left-padded) path by diffing each
    /// row's output against that same prompt run alone through the Stage 4c
    /// single-sequence cache — the trusted baseline, since there's no Python
    /// batched-candle oracle to compare against.
    VerifyBatch {
        /// `reviewer-run review --dump-prompts` jsonl (one hunk per line).
        #[arg(long)]
        jsonl: PathBuf,
        /// Use only the first N prompts (omit for all of them).
        #[arg(long)]
        n: Option<usize>,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        bf16: bool,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
        #[arg(long, default_value_t = 32)]
        max_new_tokens: usize,
    },
    /// Stage 5b/5c: sequential (batch=1 x N) vs parallel (one batch=N)
    /// wall-clock comparison on the same prompts — tokens/sec and hunks/sec,
    /// the actual measurement behind blog 6's bandwidth-bound decode claim.
    Bench {
        #[arg(long)]
        jsonl: PathBuf,
        #[arg(long)]
        n: Option<usize>,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        weights: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        bf16: bool,
        #[arg(long)]
        adapter: Option<PathBuf>,
        #[arg(long, default_value_t = 2.0)]
        lora_scale: f64,
        #[arg(long, default_value_t = 128)]
        max_new_tokens: usize,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect { oracle } => inspect(&oracle),
        Cmd::VerifyDelta { oracle } => verify_delta(&oracle),
        Cmd::VerifyMixer { oracle } => verify_mixer(&oracle),
        Cmd::VerifyLayer { oracle } => verify_layer(&oracle),
        Cmd::VerifyAttn { oracle } => verify_attn(&oracle),
        Cmd::VerifyModel { oracle, weights, config, bf16, adapter, lora_scale } => {
            verify_model(&oracle, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale)
        }
        Cmd::DumpChatFixture { out } => chat::ChatFixture::sample().save(&out),
        Cmd::VerifyChatTemplate { fixture, tokenizer, oracle } => {
            verify_chat_template(&fixture, &tokenizer, &oracle)
        }
        Cmd::Generate { fixture, tokenizer, weights, config, bf16, adapter, lora_scale, max_new_tokens, no_cache } => {
            generate_cmd(&fixture, &tokenizer, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale, max_new_tokens, no_cache)
        }
        Cmd::VerifyGenerate { fixture, tokenizer, weights, config, bf16, adapter, lora_scale, oracle } => {
            verify_generate(&fixture, &tokenizer, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale, &oracle)
        }
        Cmd::VerifyRope { oracle, config } => verify_rope(&oracle, config.as_deref()),
        Cmd::VerifyKvCache { fixture, tokenizer, weights, config, bf16, adapter, lora_scale, max_new_tokens } => {
            verify_kv_cache(&fixture, &tokenizer, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale, max_new_tokens)
        }
        Cmd::VerifyBatch { jsonl, n, tokenizer, weights, config, bf16, adapter, lora_scale, max_new_tokens } => {
            verify_batch(&jsonl, n, &tokenizer, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale, max_new_tokens)
        }
        Cmd::Bench { jsonl, n, tokenizer, weights, config, bf16, adapter, lora_scale, max_new_tokens } => {
            bench(&jsonl, n, &tokenizer, &weights, config.as_deref(), bf16, adapter.as_deref(), lora_scale, max_new_tokens)
        }
    }
}

fn load_fixtures(jsonl: &PathBuf, n: Option<usize>) -> Result<Vec<chat::ChatFixture>> {
    let mut fx = chat::ChatFixture::load_jsonl(jsonl)?;
    if let Some(n) = n {
        fx.truncate(n);
    }
    Ok(fx)
}

fn prompt_ids_for(fixtures: &[chat::ChatFixture], tok: &tokenizers::Tokenizer) -> Result<Vec<Vec<u32>>> {
    fixtures.iter().map(|f| chat::encode(tok, &chat::render_prompt(&f.system, &f.user))).collect()
}

#[allow(clippy::too_many_arguments)]
fn verify_batch(
    jsonl: &PathBuf,
    n: Option<usize>,
    tokenizer: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    max_new_tokens: usize,
) -> Result<()> {
    let fixtures = load_fixtures(jsonl, n)?;
    anyhow::ensure!(fixtures.len() >= 2, "need at least 2 prompts to exercise batching (got {})", fixtures.len());
    let tok = chat::load_tokenizer(tokenizer)?;
    let prompts = prompt_ids_for(&fixtures, &tok)?;
    let eos = chat::eos_ids(&tok);
    let pad_id = chat::pad_id(&tok);
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    println!("{} prompts, lengths {lens:?}", prompts.len());

    let (cfg, dev, w) = load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;

    println!("running batched (left-padded, one forward per step)...");
    let batched = batch::greedy_generate_batch(&w, &cfg, &prompts, pad_id, max_new_tokens, &eos, &dev)?;

    println!("running the Stage 4c single-sequence baseline, one prompt at a time...");
    let mut all_match = true;
    for (i, p) in prompts.iter().enumerate() {
        let single = generate::greedy_generate_cached(&w, &cfg, p, max_new_tokens, &eos, &dev)?;
        let single_new = &single[p.len()..];
        let matches = single_new == batched[i].as_slice();
        println!(
            "row {i} (prompt {} tokens): batched {} tokens, single {} tokens — {}",
            p.len(),
            batched[i].len(),
            single_new.len(),
            if matches { "MATCH ✓" } else { "MISMATCH ✗" }
        );
        if !matches {
            all_match = false;
            let n = batched[i].len().min(single_new.len());
            if let Some(j) = (0..n).find(|&j| batched[i][j] != single_new[j]) {
                println!("    first mismatch at generated index {j}: batched={} single={}", batched[i][j], single_new[j]);
            }
        }
    }
    println!("{}", if all_match { "ALL ROWS MATCH ✓" } else { "SOME ROWS MISMATCH ✗" });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bench(
    jsonl: &PathBuf,
    n: Option<usize>,
    tokenizer: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    max_new_tokens: usize,
) -> Result<()> {
    let fixtures = load_fixtures(jsonl, n)?;
    let tok = chat::load_tokenizer(tokenizer)?;
    let prompts = prompt_ids_for(&fixtures, &tok)?;
    let eos = chat::eos_ids(&tok);
    let pad_id = chat::pad_id(&tok);
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    println!("{} prompts, lengths {lens:?}", prompts.len());

    let (cfg, dev, w) = load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;

    println!("--- sequential (batch=1 x {}) ---", prompts.len());
    let t0 = std::time::Instant::now();
    let mut seq_outputs = Vec::new();
    for p in &prompts {
        let full = generate::greedy_generate_cached(&w, &cfg, p, max_new_tokens, &eos, &dev)?;
        seq_outputs.push(full[p.len()..].to_vec());
    }
    let seq_elapsed = t0.elapsed();
    let seq_tokens: usize = seq_outputs.iter().map(|o| o.len()).sum();

    println!("--- parallel (batch = {}) ---", prompts.len());
    let t1 = std::time::Instant::now();
    let par_outputs = batch::greedy_generate_batch(&w, &cfg, &prompts, pad_id, max_new_tokens, &eos, &dev)?;
    let par_elapsed = t1.elapsed();
    let par_tokens: usize = par_outputs.iter().map(|o| o.len()).sum();

    let n_hunks = prompts.len() as f64;
    println!();
    println!(
        "sequential: {:.1}s, {seq_tokens} tokens, {:.2} tok/s, {:.2} hunks/s",
        seq_elapsed.as_secs_f64(),
        seq_tokens as f64 / seq_elapsed.as_secs_f64(),
        n_hunks / seq_elapsed.as_secs_f64()
    );
    println!(
        "parallel:   {:.1}s, {par_tokens} tokens, {:.2} tok/s, {:.2} hunks/s",
        par_elapsed.as_secs_f64(),
        par_tokens as f64 / par_elapsed.as_secs_f64(),
        n_hunks / par_elapsed.as_secs_f64()
    );
    println!("speedup: {:.2}x (wall clock)", seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64());

    for (i, (s, p)) in seq_outputs.iter().zip(&par_outputs).enumerate() {
        println!("row {i} sequential: {}", chat::decode(&tok, s)?);
        println!("row {i} parallel:   {}", chat::decode(&tok, p)?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_kv_cache(
    fixture: &PathBuf,
    tokenizer: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    max_new_tokens: usize,
) -> Result<()> {
    let fx = chat::ChatFixture::load(fixture)?;
    let text = chat::render_prompt(&fx.system, &fx.user);
    let tok = chat::load_tokenizer(tokenizer)?;
    let prompt_ids = chat::encode(&tok, &text)?;
    let eos = chat::eos_ids(&tok);
    println!("prompt: {} tokens, generating up to {max_new_tokens} tokens both ways", prompt_ids.len());

    let (cfg, dev, w) = load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;

    let no_cache = generate::greedy_generate(&w, &cfg, &prompt_ids, max_new_tokens, &eos, &dev)?;
    let cached = generate::greedy_generate_cached(&w, &cfg, &prompt_ids, max_new_tokens, &eos, &dev)?;

    println!("no-cache ({}): {no_cache:?}", no_cache.len());
    println!("cached   ({}): {cached:?}", cached.len());
    let n = no_cache.len().min(cached.len());
    let hits = no_cache.iter().zip(&cached).take(n).filter(|(a, b)| a == b).count();
    if let Some((i, (a, b))) = no_cache.iter().zip(&cached).enumerate().find(|(_, (a, b))| a != b) {
        println!("  first mismatch at index {i}: no-cache={a} cached={b}");
        // Is this a real disagreement or another instance of the bf16 tie from
        // Stage 4b? Recompute both paths' actual logits at this exact step —
        // if they're both near-tied between the two candidates, this is the
        // same known bf16-precision floor, not a cache bug.
        diagnose_mismatch(&w, &cfg, &dev, &no_cache[..i], *a, *b)?;
    }
    println!("  {hits}/{n} tokens match, lengths: no-cache={} cached={}", no_cache.len(), cached.len());
    println!("  {}", if hits == n && no_cache.len() == cached.len() { "MATCH ✓" } else { "MISMATCH ✗" });
    Ok(())
}

/// Recompute the disputed step's logits both ways — no-cache full recompute
/// on `prefix` (the agreed-upon context) vs. a fresh prefill-then-one-decode
/// through the cache — and print where candidates `a`/`b` land in each.
fn diagnose_mismatch(
    w: &std::collections::HashMap<String, candle_core::Tensor>,
    cfg: &Config,
    dev: &Device,
    prefix: &[u32],
    a: u32,
    b: u32,
) -> Result<()> {
    let full = candle_core::Tensor::from_vec(prefix.to_vec(), (1, prefix.len()), dev)?;
    let (cos, sin) = rope::rope_cos_sin(cfg, prefix.len(), dev)?;
    let logits_nc = model::full_model_forward(w, &full, &cos, &sin, cfg)?;
    let last_nc = logits_nc.narrow(1, prefix.len() - 1, 1)?.squeeze(1)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;

    let prefill_input =
        candle_core::Tensor::from_vec(prefix[..prefix.len() - 1].to_vec(), (1, prefix.len() - 1), dev)?;
    let (_prefill_logits, mut c) = cache::prefill(w, &prefill_input, cfg, dev)?;
    let logits_c = cache::decode_step(w, prefix[prefix.len() - 1], &mut c, cfg, dev)?;
    let last_c = logits_c.squeeze(1)?.squeeze(0)?.to_dtype(candle_core::DType::F32)?;

    let nc_vals: Vec<f32> = last_nc.to_vec1()?;
    let c_vals: Vec<f32> = last_c.to_vec1()?;
    let (va, vb) = (nc_vals[a as usize], nc_vals[b as usize]);
    let (va2, vb2) = (c_vals[a as usize], c_vals[b as usize]);
    println!("  no-cache logits: {a}={va:.4} {b}={vb:.4} (diff {:.4})", (va - vb).abs());
    println!("  cached   logits: {a}={va2:.4} {b}={vb2:.4} (diff {:.4})", (va2 - vb2).abs());
    Ok(())
}

fn verify_rope(oracle: &PathBuf, config: Option<&std::path::Path>) -> Result<()> {
    let cfg = match config {
        Some(p) => Config::from_json(p)?,
        None => Config::qwen9b(),
    };
    let o = safetensors::load(oracle, &Device::Cpu)
        .with_context(|| format!("loading {}", oracle.display()))?;
    let exp_cos = o["cos"].to_dtype(candle_core::DType::F32)?;
    let exp_sin = o["sin"].to_dtype(candle_core::DType::F32)?;
    // oracle tensors may be [1,s,rd] or [s,rd]; normalize to [1,s,rd].
    let exp_cos = if exp_cos.dims().len() == 2 { exp_cos.unsqueeze(0)? } else { exp_cos };
    let exp_sin = if exp_sin.dims().len() == 2 { exp_sin.unsqueeze(0)? } else { exp_sin };
    let s = exp_cos.dim(1)?;

    let (got_cos, got_sin) = rope::rope_cos_sin(&cfg, s, &Device::Cpu)?;
    println!("seq_len={s}, rotary_dim={}", cfg.rotary_dim());
    // The model casts cos/sin to bf16 before use (see model::full_model_forward),
    // and this oracle round-tripped through that same cast, so tolerance is one
    // bf16 ULP near 1.0 (~2^-7 ≈ 7.8e-3), not float32 precision.
    compare(&got_cos, &exp_cos, "rope cos table", 8e-3)?;
    compare(&got_sin, &exp_sin, "rope sin table", 8e-3)?;
    Ok(())
}

/// Shared setup for `Generate`/`VerifyGenerate`: config, device, weights (+LoRA).
fn load_model_for_generation(
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
) -> Result<(Config, candle_core::Device, std::collections::HashMap<String, candle_core::Tensor>)> {
    let cfg = match config {
        Some(p) => Config::from_json(p)?,
        None => Config::qwen9b(),
    };
    let dtype = if bf16 { candle_core::DType::BF16 } else { candle_core::DType::F32 };
    let dev = Device::cuda_if_available(0)?;
    println!("config: {} layers, hidden {}, dtype {dtype:?}, device {dev:?}", cfg.n_layers, cfg.hidden);
    println!("loading weights from {} …", weights.display());
    let mut w = model::load_weights(weights, &dev, dtype)?;
    println!("  loaded {} language-model tensors", w.len());
    if let Some(ad) = adapter {
        println!("merging LoRA adapter {} …", ad.display());
        model::apply_lora(&mut w, ad, lora_scale, dtype)?;
    }
    Ok((cfg, dev, w))
}

#[allow(clippy::too_many_arguments)]
fn generate_cmd(
    fixture: &PathBuf,
    tokenizer: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    max_new_tokens: usize,
    no_cache: bool,
) -> Result<()> {
    let fx = chat::ChatFixture::load(fixture)?;
    let text = chat::render_prompt(&fx.system, &fx.user);
    let tok = chat::load_tokenizer(tokenizer)?;
    let prompt_ids = chat::encode(&tok, &text)?;
    let eos = chat::eos_ids(&tok);
    println!("prompt: {} tokens", prompt_ids.len());

    let (cfg, dev, w) = load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;
    let ids = if no_cache {
        generate::greedy_generate(&w, &cfg, &prompt_ids, max_new_tokens, &eos, &dev)?
    } else {
        generate::greedy_generate_cached(&w, &cfg, &prompt_ids, max_new_tokens, &eos, &dev)?
    };
    let new_ids = &ids[prompt_ids.len()..];
    println!("generated {} new tokens: {new_ids:?}", new_ids.len());
    println!("comment:\n{}", chat::decode(&tok, new_ids)?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_generate(
    fixture: &PathBuf,
    tokenizer: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
    oracle: &PathBuf,
) -> Result<()> {
    let fx = chat::ChatFixture::load(fixture)?;
    let text = chat::render_prompt(&fx.system, &fx.user);
    let tok = chat::load_tokenizer(tokenizer)?;
    let prompt_ids = chat::encode(&tok, &text)?;
    let eos = chat::eos_ids(&tok);

    let o = safetensors::load(oracle, &Device::Cpu)
        .with_context(|| format!("loading {}", oracle.display()))?;
    let exp_ids: Vec<i64> = o["ids"].to_vec1()?;
    let prompt_len: Vec<i64> = o["prompt_len"].to_vec1()?;
    let prompt_len = prompt_len[0] as usize;
    anyhow::ensure!(
        prompt_len == prompt_ids.len(),
        "prompt length mismatch before generation even starts: rust={} python={prompt_len} \
         (Stage 4a template/tokenizer drift — re-run verify-chat-template)",
        prompt_ids.len()
    );
    let max_new = exp_ids.len() - prompt_len;
    println!("prompt: {prompt_len} tokens, oracle generated {max_new} new tokens");

    let (cfg, dev, w) = load_model_for_generation(weights, config, bf16, adapter, lora_scale)?;
    let got_ids = generate::greedy_generate(&w, &cfg, &prompt_ids, max_new, &eos, &dev)?;

    println!("rust   ({}): {got_ids:?}", got_ids.len());
    println!("python ({}): {exp_ids:?}", exp_ids.len());
    let n = got_ids.len().min(exp_ids.len());
    let hits = got_ids.iter().zip(&exp_ids).take(n).filter(|(a, b)| **a as i64 == **b).count();
    if let Some((i, (a, b))) = got_ids.iter().zip(&exp_ids).enumerate().find(|(_, (a, b))| **a as i64 != **b) {
        println!("  first mismatch at index {i} (position {} of the generated suffix): rust={a} python={b}", i as i64 - prompt_len as i64);
    }
    println!("  {hits}/{n} tokens match, lengths: rust={} python={}", got_ids.len(), exp_ids.len());
    println!("  {}", if hits == n && got_ids.len() == exp_ids.len() { "MATCH ✓" } else { "MISMATCH ✗" });
    Ok(())
}

fn verify_chat_template(fixture: &PathBuf, tokenizer: &PathBuf, oracle: &PathBuf) -> Result<()> {
    let fx = chat::ChatFixture::load(fixture)?;
    let text = chat::render_prompt(&fx.system, &fx.user);
    let tok = chat::load_tokenizer(tokenizer)?;
    let got = chat::encode(&tok, &text)?;

    let o = safetensors::load(oracle, &Device::Cpu)
        .with_context(|| format!("loading {}", oracle.display()))?;
    let exp: Vec<i64> = o["input_ids"].to_vec1()?;

    println!("rendered prompt ({} chars):\n{text}", text.len());
    println!("rust ids   ({}): {got:?}", got.len());
    println!("python ids ({}): {exp:?}", exp.len());

    let matches = got.len() == exp.len() && got.iter().zip(&exp).all(|(a, b)| *a as i64 == *b);
    if !matches {
        if let Some((i, (a, b))) = got.iter().zip(&exp).enumerate().find(|(_, (a, b))| **a as i64 != **b) {
            println!("  first mismatch at index {i}: rust={a} python={b}");
        } else if got.len() != exp.len() {
            println!("  length mismatch: rust={} python={}", got.len(), exp.len());
        }
    }
    println!("  {}", if matches { "MATCH ✓" } else { "MISMATCH ✗" });
    Ok(())
}

fn verify_model(
    oracle: &PathBuf,
    weights: &PathBuf,
    config: Option<&std::path::Path>,
    bf16: bool,
    adapter: Option<&std::path::Path>,
    lora_scale: f64,
) -> Result<()> {
    let cfg = match config {
        Some(p) => Config::from_json(p)?,
        None => Config::qwen9b(),
    };
    let dtype = if bf16 { candle_core::DType::BF16 } else { candle_core::DType::F32 };
    println!("config: {} layers, hidden {}, dtype {dtype:?}", cfg.n_layers, cfg.hidden);
    let dev = Device::cuda_if_available(0)?;
    println!("device: {dev:?}");
    let o = safetensors::load(oracle, &dev)
        .with_context(|| format!("loading {}", oracle.display()))?;
    println!("loading weights from {} …", weights.display());
    let mut w = model::load_weights(weights, &dev, dtype)?;
    println!("  loaded {} language-model tensors", w.len());
    if let Some(ad) = adapter {
        println!("merging LoRA adapter {} …", ad.display());
        model::apply_lora(&mut w, ad, lora_scale, dtype)?;
    }

    let logits = model::full_model_forward(&w, &o["input_ids"], &o["cos"], &o["sin"], &cfg)?;
    let got = logits.squeeze(0)?.to_dtype(candle_core::DType::F32)?; // [s, vocab]
    compare(&got, &o["logits"], "full model logits", 5e-1)?;

    // Argmax agreement — the meaningful "same predictions" check.
    let got_am: Vec<u32> = got.argmax(candle_core::D::Minus1)?.to_vec1()?;
    let exp_am: Vec<i64> = o["argmax"].to_vec1()?;
    let n = got_am.len();
    let hits = got_am.iter().zip(&exp_am).filter(|(a, b)| **a as i64 == **b).count();
    println!("  argmax agreement: {hits}/{n} positions");
    println!("  {}", if hits == n { "ARGMAX MATCH ✓" } else { "ARGMAX MISMATCH ✗" });
    Ok(())
}

fn verify_attn(path: &PathBuf) -> Result<()> {
    let w = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let got = model::decoder_layer_full(&w, &w["input"], &w["cos"], &w["sin"], "", &Config::qwen9b())?;
    compare(&got, &w["output"], "full-attention decoder layer", 1e-3)
}

/// Compare a computed tensor against the oracle's expected tensor.
fn compare(got: &candle_core::Tensor, exp: &candle_core::Tensor, what: &str, tol: f32) -> Result<()> {
    let diff = got.sub(exp)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    let mean = diff.mean_all()?.to_scalar::<f32>()?;
    println!("{what} vs reference:");
    println!("  max_abs_diff  = {max:.3e}");
    println!("  mean_abs_diff = {mean:.3e}");
    println!("  {}", if max < tol { "MATCH ✓" } else { "MISMATCH ✗" });
    Ok(())
}

fn verify_layer(path: &PathBuf) -> Result<()> {
    let w = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let got = model::decoder_layer_linear(&w, &w["input"], "", &Config::qwen9b())?;
    compare(&got, &w["output"], "DeltaNet decoder layer", 1e-3)
}

fn verify_mixer(path: &PathBuf) -> Result<()> {
    let w = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let got = mixer::mixer_forward(&w, &w["input"], "", &Config::qwen9b())?;
    compare(&got, &w["output"], "full DeltaNet mixer", 1e-3)
}

fn verify_delta(path: &PathBuf) -> Result<()> {
    let t = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let (got, _final_state) = delta::recurrent_gated_delta_rule(
        &t["q"], &t["k"], &t["v"], &t["g"], &t["beta"], true, None,
    )?;
    let exp = &t["out"];
    let diff = got.sub(exp)?.abs()?;
    let max = diff.max_all()?.to_scalar::<f32>()?;
    let mean = diff.mean_all()?.to_scalar::<f32>()?;
    println!("gated delta recurrence vs reference:");
    println!("  max_abs_diff  = {max:.3e}");
    println!("  mean_abs_diff = {mean:.3e}");
    println!("  {}", if max < 1e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    Ok(())
}

fn inspect(path: &PathBuf) -> Result<()> {
    let tensors = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading oracle {}", path.display()))?;
    let mut names: Vec<_> = tensors.keys().cloned().collect();
    names.sort();
    println!("oracle: {} tensors from {}", names.len(), path.display());
    for n in &names {
        let t = &tensors[n];
        println!("  {n:14} {:?}  {:?}", t.dtype(), t.dims());
    }
    Ok(())
}
