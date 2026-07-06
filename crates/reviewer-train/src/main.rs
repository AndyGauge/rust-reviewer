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
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect { oracle } => inspect(&oracle),
        Cmd::VerifyDelta { oracle } => verify_delta(&oracle),
    }
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
