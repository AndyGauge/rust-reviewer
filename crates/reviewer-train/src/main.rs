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

mod delta;
mod mixer;
mod model;

use anyhow::{Context, Result};
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
        /// oracle_full_f32.safetensors (input_ids, cos/sin, logits, argmax).
        #[arg(long)]
        oracle: PathBuf,
        /// Directory of the 9B's sharded safetensors weights.
        #[arg(long)]
        weights: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect { oracle } => inspect(&oracle),
        Cmd::VerifyDelta { oracle } => verify_delta(&oracle),
        Cmd::VerifyMixer { oracle } => verify_mixer(&oracle),
        Cmd::VerifyLayer { oracle } => verify_layer(&oracle),
        Cmd::VerifyAttn { oracle } => verify_attn(&oracle),
        Cmd::VerifyModel { oracle, weights } => verify_model(&oracle, &weights),
    }
}

fn verify_model(oracle: &PathBuf, weights: &PathBuf) -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    println!("device: {dev:?}");
    let o = safetensors::load(oracle, &dev)
        .with_context(|| format!("loading {}", oracle.display()))?;
    println!("loading 9B weights from {} …", weights.display());
    let w = model::load_weights(weights, &dev)?;
    println!("  loaded {} language-model tensors", w.len());

    let logits = model::full_model_forward(&w, &o["input_ids"], &o["cos"], &o["sin"])?;
    let got = logits.squeeze(0)?; // [s, vocab]
    compare(&got, &o["logits"], "full model logits", 5e-2)?;

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
    let got = model::decoder_layer_full(&w, &w["input"], &w["cos"], &w["sin"], "")?;
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
    let got = model::decoder_layer_linear(&w, &w["input"], "")?;
    compare(&got, &w["output"], "DeltaNet decoder layer", 1e-3)
}

fn verify_mixer(path: &PathBuf) -> Result<()> {
    let w = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let got = mixer::mixer_forward(&w, &w["input"], "")?;
    compare(&got, &w["output"], "full DeltaNet mixer", 1e-3)
}

fn verify_delta(path: &PathBuf) -> Result<()> {
    let t = safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("loading {}", path.display()))?;
    let got = delta::recurrent_gated_delta_rule(
        &t["q"], &t["k"], &t["v"], &t["g"], &t["beta"], true,
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
