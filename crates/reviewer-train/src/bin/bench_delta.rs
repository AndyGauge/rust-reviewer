//! Candle baseline for the cuTile Gated DeltaNet recurrence
//! (`experiments/cutile-deltanet`): the *actual* `delta.rs` the trainer runs, on the
//! GPU, at Qwen3.6-27B shapes (48 value heads, Dk = Dv = 128, batch 1, f32).
//! Prints JSON lines matching the cuTile bench.
//!
//!   CUDA_COMPUTE_CAP=121 cargo run --release -p reviewer-train --features cuda --bin bench_delta

#[cfg(feature = "cutile")]
#[path = "../cutile_delta.rs"]
#[allow(dead_code)]
mod cutile_delta;

#[path = "../delta.rs"]
#[allow(dead_code)]
mod delta;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    eprintln!("device: {:?}", dev);
    let (b, h, dk, dv) = (1usize, 48usize, 128usize, 128usize);
    for &s in &[64usize, 256, 1024, 2048] {
        let q = Tensor::randn(0f32, 1f32, (b, s, h, dk), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, s, h, dk), &dev)?;
        let v = Tensor::randn(0f32, 1f32, (b, s, h, dv), &dev)?;
        let g = Tensor::rand(-1f32, 0f32, (b, s, h), &dev)?;
        let beta = Tensor::rand(0f32, 1f32, (b, s, h), &dev)?;
        assert_eq!(q.dtype(), DType::F32);
        let run = || delta::recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, true, None);
        run()?; // warmup
        dev.synchronize()?;
        let mut times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let (out, _state) = run()?;
            dev.synchronize()?;
            drop(out);
            times.push(t.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(r#"{{"impl":"candle","s":{s},"us":{:.1}}}"#, times[2]);
    }
    Ok(())
}
