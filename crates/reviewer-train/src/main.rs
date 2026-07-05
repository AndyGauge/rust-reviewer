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
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Inspect { oracle } => inspect(&oracle),
    }
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
