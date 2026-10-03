//! Runs anywhere (not gx10): polls a `serve` instance's `/health` endpoint
//! over the network and appends every sample to a local file, so the
//! capacity-planning record survives independent of gx10's own uptime.

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::AsyncWriteExt;

use crate::sample::Sample;

pub struct WatchArgs {
    pub url: String,
    pub interval_secs: u64,
    pub out: PathBuf,
}

pub async fn run(args: WatchArgs) -> Result<()> {
    let client = reqwest::Client::new();
    let health_url = format!("{}/health", args.url.trim_end_matches('/'));

    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .await
        .with_context(|| format!("opening {:?}", args.out))?;

    let mut interval = tokio::time::interval(Duration::from_secs(args.interval_secs));
    loop {
        interval.tick().await;
        // A fetch failure (the exact kind of network disruption this tool
        // exists to catch) should not kill the recorder — log it and keep polling.
        match fetch(&client, &health_url).await {
            Ok(sample) => {
                let line = serde_json::to_string(&sample)?;
                println!("{line}");
                if let Err(e) = file.write_all(format!("{line}\n").as_bytes()).await {
                    eprintln!("gx10-monitor watch: failed writing {:?}: {e:#}", args.out);
                }
            }
            Err(e) => eprintln!("gx10-monitor watch: fetch failed: {e:#}"),
        }
    }
}

async fn fetch(client: &reqwest::Client, url: &str) -> Result<Sample> {
    let resp = client.get(url).send().await.context("GET /health")?;
    let resp = resp.error_for_status().context("non-2xx from /health")?;
    resp.json::<Sample>().await.context("parsing /health JSON")
}
