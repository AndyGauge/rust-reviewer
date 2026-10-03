//! The unit of data collected each tick: GPU stats, system memory, and the
//! thread count of a matching process — combined and timestamped.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::gpu::{self, GpuSample};
use crate::mem::{self, MemSample};
use crate::threads;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sample {
    pub ts: String,
    #[serde(flatten)]
    pub gpu: GpuSample,
    #[serde(flatten)]
    pub mem: MemSample,
    pub process_count: u32,
    pub process_threads: u32,
}

pub async fn collect_sample(process_pattern: &str) -> Result<Sample> {
    let gpu = gpu::sample().await.context("nvidia-smi sample")?;
    let mem = mem::sample().await.context("/proc/meminfo sample")?;
    let (process_count, process_threads) = threads::sample(process_pattern)
        .await
        .context("process thread sample")?;

    Ok(Sample {
        ts: chrono::Utc::now().to_rfc3339(),
        gpu,
        mem,
        process_count,
        process_threads,
    })
}
