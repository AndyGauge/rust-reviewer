//! Reads system memory from `/proc/meminfo`. On GB10 this *is* GPU memory too
//! — CPU and GPU share one unified pool — so this is the number that actually
//! matters when `nvidia-smi`'s own memory fields report `N/A`.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemSample {
    pub sys_mem_total_gib: f64,
    pub sys_mem_available_gib: f64,
    pub sys_mem_used_gib: f64,
}

pub async fn sample() -> Result<MemSample> {
    let text = tokio::fs::read_to_string("/proc/meminfo")
        .await
        .context("reading /proc/meminfo")?;

    let mut fields = HashMap::new();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let kb: u64 = rest
            .trim()
            .split_whitespace()
            .next()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        fields.insert(key.to_string(), kb);
    }

    let total_kb = *fields.get("MemTotal").context("missing MemTotal")?;
    let available_kb = *fields
        .get("MemAvailable")
        .context("missing MemAvailable")?;

    let kb_to_gib = |kb: u64| kb as f64 / (1024.0 * 1024.0);

    Ok(MemSample {
        sys_mem_total_gib: kb_to_gib(total_kb),
        sys_mem_available_gib: kb_to_gib(available_kb),
        sys_mem_used_gib: kb_to_gib(total_kb.saturating_sub(available_kb)),
    })
}
