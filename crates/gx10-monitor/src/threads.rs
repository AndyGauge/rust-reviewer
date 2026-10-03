//! Counts OS threads across every process whose command line matches
//! `process_pattern` (via `pgrep -f`), by summing the `Threads:` field of
//! each matched PID's `/proc/<pid>/status`. `nvidia-smi` has no equivalent —
//! this is the thing it can't tell you.

use anyhow::{Context, Result, bail};
use tokio::process::Command;

pub async fn sample(process_pattern: &str) -> Result<(u32, u32)> {
    let output = Command::new("pgrep")
        .args(["-f", process_pattern])
        .output()
        .await
        .context("spawning pgrep")?;

    // pgrep exits 1 when nothing matches — that's zero processes, not an error.
    if !output.status.success() && output.status.code() != Some(1) {
        bail!(
            "pgrep exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let pids: Vec<u32> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|l| l.trim().parse().ok())
        .collect();

    let mut total_threads = 0u32;
    for pid in &pids {
        // A matched process can exit between the pgrep snapshot and this read;
        // skip it rather than failing the whole sample over that race.
        if let Some(n) = read_thread_count(*pid).await {
            total_threads += n;
        }
    }

    Ok((pids.len() as u32, total_threads))
}

async fn read_thread_count(pid: u32) -> Option<u32> {
    let text = tokio::fs::read_to_string(format!("/proc/{pid}/status"))
        .await
        .ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .and_then(|v| v.trim().parse().ok())
}
