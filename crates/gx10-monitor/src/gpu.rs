//! Queries `nvidia-smi` for temperature and utilization. GPU memory is
//! queried too, but on GB10's unified-memory design `nvidia-smi` reports it as
//! the literal string `[N/A]` — there's no discrete VRAM to report — so those
//! fields parse to `None` instead of failing the whole sample.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuSample {
    pub gpu_temp_c: Option<f32>,
    pub gpu_util_pct: Option<f32>,
    pub gpu_mem_used_mib: Option<u64>,
    pub gpu_mem_total_mib: Option<u64>,
}

pub async fn sample() -> Result<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .context("spawning nvidia-smi")?;

    if !output.status.success() {
        bail!(
            "nvidia-smi exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().context("empty nvidia-smi output")?;
    let fields: Vec<&str> = first_line.split(',').map(str::trim).collect();
    let [temp, util, mem_used, mem_total] = fields.as_slice() else {
        bail!("expected 4 fields from nvidia-smi, got: {first_line:?}");
    };

    Ok(GpuSample {
        gpu_temp_c: temp.parse().ok(),
        gpu_util_pct: util.parse().ok(),
        gpu_mem_used_mib: mem_used.parse().ok(),
        gpu_mem_total_mib: mem_total.parse().ok(),
    })
}
