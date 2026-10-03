//! Network-queryable health monitor for gx10: GPU temperature/utilization
//! (`nvidia-smi`), system memory (`/proc/meminfo` — the number that matters on
//! GB10's unified-memory design, since `nvidia-smi`'s own memory fields report
//! `N/A`), and OS thread count for a matching process (`pgrep` +
//! `/proc/<pid>/status`) — nvidia-smi alone doesn't expose that last one.
//!
//! Two subcommands:
//!
//! ```text
//! # on gx10: sample every 10 minutes, serve latest + history over HTTP
//! gx10-monitor serve --process-pattern "vllm serve" --log-file ~/gx10-monitor.log
//!
//! # anywhere else: poll gx10 over the network and keep a durable local
//! # record, for capacity planning / spotting exhaustion trends independent
//! # of gx10's own uptime
//! gx10-monitor watch --url http://gx10-d903.local:9100 --out ~/gx10-history.jsonl
//! ```

mod gpu;
mod mem;
mod sample;
mod serve;
mod threads;
mod watch;

use std::net::SocketAddr;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(about = "Sample and serve gx10's GPU/memory/thread health over the network")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run on gx10: sample periodically, serve `/health` (latest) and
    /// `/samples` (recent history) over HTTP.
    Serve {
        /// Address to bind the HTTP server to.
        #[arg(long, default_value = "0.0.0.0:9100")]
        bind: SocketAddr,
        /// How often to sample, in seconds.
        #[arg(long, default_value_t = 600)]
        interval_secs: u64,
        /// `pgrep -f` pattern for the process whose thread count gets tracked.
        #[arg(long, default_value = "vllm serve")]
        process_pattern: String,
        /// Optional local file to append every sample to, for durability
        /// independent of the in-memory history buffer.
        #[arg(long)]
        log_file: Option<PathBuf>,
        /// Max samples kept in memory for `/samples` (default: 28 days at the
        /// default 10-minute interval).
        #[arg(long, default_value_t = 4032)]
        history_capacity: usize,
    },
    /// Run anywhere: poll a `serve` instance's `/health` endpoint over the
    /// network and append every sample to a local file.
    Watch {
        /// Base URL of a running `gx10-monitor serve`, e.g.
        /// http://gx10-d903.local:9100
        #[arg(long)]
        url: String,
        /// How often to poll, in seconds.
        #[arg(long, default_value_t = 600)]
        interval_secs: u64,
        /// File to append recorded samples to.
        #[arg(long)]
        out: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Serve {
            bind,
            interval_secs,
            process_pattern,
            log_file,
            history_capacity,
        } => {
            serve::run(serve::ServeArgs {
                bind,
                interval_secs,
                process_pattern,
                log_file,
                history_capacity,
            })
            .await
        }
        Cmd::Watch {
            url,
            interval_secs,
            out,
        } => watch::run(watch::WatchArgs { url, interval_secs, out }).await,
    }
}
