//! Candle-op baseline for the cuTile RMSNorm / softmax benchmarks.
//!
//! Same shapes as `cutile-benchmarks` (M=4096 rows, N=2^10..2^15, f16) and the
//! same throughput definition (bytes read + bytes written of the M x N tensor),
//! so the numbers are directly comparable. Prints one JSON object per line.
//!
//!   CUDA_COMPUTE_CAP=121 cargo run --release -p reviewer-train --features cuda --bin bench_ops

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::ops;
use std::time::Instant;

const M: usize = 4096;
const WARMUP: usize = 20;
const ITERS: usize = 100;
const BATCHES: usize = 7;
/// Distinct input tensors cycled through, so small shapes don't just re-read L2.
const ROTATE: usize = 8;

/// The `rmsnorm` from `reviewer-train/src/model.rs`: what the model actually runs
/// (op-by-op: sqr, mean, affine, sqrt, div, mul), with the `(1 + weight)` scale.
fn rmsnorm_naive(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = x.broadcast_div(&var.affine(1.0, eps)?.sqrt()?)?;
    Ok(normed.broadcast_mul(&weight.affine(1.0, 1.0)?)?)
}

/// Median per-call microseconds over `BATCHES` batches of `ITERS` calls, with the
/// device synchronized around each batch so we time the GPU, not the launch queue.
fn time_us(dev: &Device, mut f: impl FnMut(usize) -> Result<Tensor>) -> Result<f64> {
    for i in 0..WARMUP {
        f(i)?;
    }
    dev.synchronize()?;
    let mut per_call = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        let t = Instant::now();
        for i in 0..ITERS {
            f(i)?;
        }
        dev.synchronize()?;
        per_call.push(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64);
    }
    per_call.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(per_call[BATCHES / 2])
}

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    eprintln!("device: {:?}", dev);
    for p in 10..=15 {
        let n = 1usize << p;
        let xs = (0..ROTATE)
            .map(|_| Ok(Tensor::randn(0f32, 1f32, (M, n), &dev)?.to_dtype(DType::F16)?))
            .collect::<Result<Vec<_>>>()?;
        let w = Tensor::randn(0f32, 1f32, n, &dev)?.to_dtype(DType::F16)?;
        let bytes = (2 * M * n * 2) as f64; // read + write, f16

        let runs: Vec<(&str, f64)> = vec![
            ("rmsnorm_naive", time_us(&dev, |i| rmsnorm_naive(&xs[i % ROTATE], &w, 1e-5))?),
            ("rmsnorm_fused", time_us(&dev, |i| Ok(ops::rms_norm(&xs[i % ROTATE], &w, 1e-5)?))?),
            ("softmax_naive", time_us(&dev, |i| Ok(ops::softmax(&xs[i % ROTATE], D::Minus1)?))?),
            ("softmax_fused", time_us(&dev, |i| Ok(ops::softmax_last_dim(&xs[i % ROTATE])?))?),
        ];
        for (kernel, us) in runs {
            println!(
                "{}",
                serde_json::json!({
                    "kernel": kernel, "n": n, "us": us, "gb_s": bytes / (us * 1e-6) / 1e9
                })
            );
        }
    }
    Ok(())
}
