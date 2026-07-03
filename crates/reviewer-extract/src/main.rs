//! Pull every review comment from a GitHub repo into a local JSONL file.
//!
//! Uses `GET /repos/{owner}/{repo}/pulls/comments` (repo-wide review comments),
//! sorted by `updated` ascending so runs are resumable via a checkpoint.
//!
//! ```text
//! GITHUB_TOKEN=ghp_xxx cargo run -p reviewer-extract -- \
//!     --repo rust-lang/rust --out data/raw/rust.jsonl
//! ```
//!
//! Re-running continues from the last `updated_at` written to
//! `<out>.checkpoint`. Honors primary and secondary rate limits.

use std::io::Write as _;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use reviewer_core::ReviewComment;

#[derive(Parser)]
#[command(about = "Fetch GitHub PR review comments into JSONL")]
struct Args {
    /// `owner/name`, e.g. `rust-lang/rust` or `rust-lang/rust-cookbook`.
    #[arg(long)]
    repo: String,

    /// Output JSONL path. A sibling `<out>.checkpoint` tracks progress.
    #[arg(long)]
    out: PathBuf,

    /// GitHub token (classic or fine-grained; public-repo read is enough).
    #[arg(long, env = "GITHUB_TOKEN")]
    token: String,

    /// Only fetch comments updated at/after this ISO-8601 timestamp.
    /// Overrides the checkpoint when set. Required when --workers > 1.
    #[arg(long)]
    since: Option<String>,

    /// Upper bound (exclusive) for concurrent mode. Defaults to now.
    #[arg(long)]
    until: Option<String>,

    /// Number of concurrent time-sharded workers. 1 = sequential (resumable,
    /// checkpointed). >1 = split [since, until] into N windows crawled in
    /// parallel — far faster when latency-bound, as the GitHub API is here.
    /// NOTE: concurrent mode is NOT checkpointed; a crash loses the whole crawl.
    #[arg(long, default_value_t = 1)]
    workers: usize,

    /// Stop after roughly this many comments (debugging, sequential only).
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

const API: &str = "https://api.github.com";
const PER_PAGE: u32 = 100;

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.workers > 1 {
        run_concurrent(args).await
    } else {
        run_sequential(args).await
    }
}

async fn run_sequential(args: Args) -> Result<()> {
    let checkpoint_path = args.out.with_extension("checkpoint");
    let since = args
        .since
        .clone()
        .or_else(|| std::fs::read_to_string(&checkpoint_path).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());

    let client = reqwest::Client::builder()
        .user_agent("rustc-reviewer-extract/0.1")
        .build()?;

    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .with_context(|| format!("opening {}", args.out.display()))?;

    // First page URL. We page via the `Link: rel="next"` header thereafter.
    let mut url = {
        let mut u = format!(
            "{API}/repos/{}/pulls/comments?per_page={PER_PAGE}&sort=updated&direction=asc",
            args.repo
        );
        if let Some(s) = &since {
            u.push_str(&format!("&since={s}"));
            eprintln!("resuming from since={s}");
        }
        Some(u)
    };

    let mut total = 0usize;
    let mut max_updated = since.unwrap_or_default();

    while let Some(next) = url {
        let resp = get_with_backoff(&client, &next, &args.token).await?;

        let link_next = parse_next_link(resp.headers().get("link"));
        let body = resp.text().await?;
        let page: Vec<ReviewComment> =
            serde_json::from_str(&body).context("decoding comment page")?;

        if page.is_empty() {
            break;
        }

        for c in &page {
            if c.updated_at > max_updated {
                max_updated = c.updated_at.clone();
            }
            serde_json::to_writer(&mut out, c)?;
            out.write_all(b"\n")?;
            total += 1;
        }
        out.flush()?;
        // Persist progress after every page so a crash loses at most one page.
        std::fs::write(&checkpoint_path, &max_updated)?;
        eprintln!("+{} (total {total}, latest {max_updated})", page.len());

        if args.limit != 0 && total >= args.limit {
            eprintln!("hit --limit {}", args.limit);
            break;
        }
        url = link_next;
    }

    eprintln!("done: {total} comments -> {}", args.out.display());
    Ok(())
}

/// Crawl `[since, until)` with N time-sharded workers running concurrently.
///
/// A single `Link`-pagination chain can't be parallelized (each next URL comes
/// from the previous response), so instead we split the time range into N equal
/// windows and run one chain per window. Boundary overlaps are harmless:
/// `reviewer-prepare` dedups by comment id. Each worker writes its own shard,
/// which we append to `--out` at the end (preserving any existing data).
async fn run_concurrent(args: Args) -> Result<()> {
    use chrono::{DateTime, SecondsFormat, Utc};

    let start: DateTime<Utc> = parse_ts(
        args.since
            .as_deref()
            .context("--since is required when --workers > 1")?,
    )?;
    let end: DateTime<Utc> = match args.until.as_deref() {
        Some(u) => parse_ts(u)?,
        None => Utc::now(),
    };
    anyhow::ensure!(end > start, "--until must be after --since");

    let n = args.workers as i64;
    let span = (end - start).num_seconds();
    let step = (span / n).max(1);

    let client = reqwest::Client::builder()
        .user_agent("rustc-reviewer-extract/0.1")
        .build()?;

    eprintln!(
        "concurrent crawl: {} workers over {} -> {}",
        args.workers,
        start.to_rfc3339_opts(SecondsFormat::Secs, true),
        end.to_rfc3339_opts(SecondsFormat::Secs, true),
    );

    let mut handles = Vec::new();
    for i in 0..n {
        let w_start = start + chrono::Duration::seconds(step * i);
        let w_end = if i == n - 1 {
            end
        } else {
            start + chrono::Duration::seconds(step * (i + 1))
        };
        let shard = args.out.with_extension(format!("shard{i}"));
        let (client, repo, token) = (client.clone(), args.repo.clone(), args.token.clone());
        handles.push(tokio::spawn(async move {
            worker(&client, &repo, &token, w_start, w_end, &shard, i).await
        }));
    }

    // Join all workers and append their shards to the output file.
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.out)
        .with_context(|| format!("opening {}", args.out.display()))?;
    let mut grand_total = 0usize;
    for (i, h) in handles.into_iter().enumerate() {
        let (shard, count) = h.await.context("worker panicked")??;
        let bytes = std::fs::read(&shard)?;
        out.write_all(&bytes)?;
        std::fs::remove_file(&shard).ok();
        grand_total += count;
        eprintln!("merged shard{i}: {count} comments");
    }
    out.flush()?;
    eprintln!("done: {grand_total} comments appended -> {}", args.out.display());
    Ok(())
}

/// One time-window worker. Pages forward from `start`, writing comments until it
/// reaches `end` (RFC3339/UTC compares correctly as strings) or runs out.
async fn worker(
    client: &reqwest::Client,
    repo: &str,
    token: &str,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    shard: &std::path::Path,
    id: i64,
) -> Result<(PathBuf, usize)> {
    use chrono::SecondsFormat;
    let since = start.to_rfc3339_opts(SecondsFormat::Secs, true);
    let end_str = end.to_rfc3339_opts(SecondsFormat::Secs, true);

    let mut url = Some(format!(
        "{API}/repos/{repo}/pulls/comments?per_page={PER_PAGE}&sort=updated&direction=asc&since={since}"
    ));
    let mut out = std::fs::File::create(shard)
        .with_context(|| format!("creating {}", shard.display()))?;
    let mut count = 0usize;

    while let Some(u) = url {
        let resp = get_with_backoff(client, &u, token).await?;
        let link_next = parse_next_link(resp.headers().get("link"));
        let page: Vec<ReviewComment> =
            serde_json::from_str(&resp.text().await?).context("decoding comment page")?;
        if page.is_empty() {
            break;
        }

        let mut reached_end = false;
        for c in &page {
            if c.updated_at.as_str() >= end_str.as_str() {
                reached_end = true; // belongs to the next window
                break;
            }
            serde_json::to_writer(&mut out, c)?;
            out.write_all(b"\n")?;
            count += 1;
        }
        out.flush()?;
        if reached_end {
            break;
        }
        eprintln!("[w{id}] +{} (window total {count})", page.len());
        url = link_next;
    }
    Ok((shard.to_path_buf(), count))
}

fn parse_ts(s: &str) -> Result<chrono::DateTime<chrono::Utc>> {
    Ok(chrono::DateTime::parse_from_rfc3339(s)
        .with_context(|| format!("parsing timestamp {s:?} (want RFC3339 like 2022-09-03T00:00:00Z)"))?
        .with_timezone(&chrono::Utc))
}

/// GET with rate-limit awareness. Retries on secondary limits / 5xx.
async fn get_with_backoff(
    client: &reqwest::Client,
    url: &str,
    token: &str,
) -> Result<reqwest::Response> {
    let mut attempt = 0u32;
    let mut rate_limit_waits = 0u32;
    loop {
        let resp = client
            .get(url)
            .bearer_auth(token)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .send()
            .await?;

        let status = resp.status();
        if status.is_success() {
            // Proactively pause if we've nearly exhausted the primary budget.
            if remaining(&resp) == Some(0) {
                sleep_until_reset(&resp).await;
            }
            return Ok(resp);
        }

        // Genuine rate limiting: a 429, or a 403 that has *actually* exhausted
        // the budget (remaining == 0). GitHub attaches x-ratelimit-* headers to
        // essentially every response, including permission-denied 403s ("Resource
        // not accessible", SSO-blocked tokens) — so treating any 403-with-reset
        // as rate limiting sleeps a forbidden request until reset, retries, and
        // loops forever. Gate on remaining, and cap the number of waits.
        let rate_limited = status == 429 || (status == 403 && remaining(&resp) == Some(0));
        if rate_limited && rate_limit_waits < 5 {
            if let Some(wait) = retry_after(&resp).or_else(|| reset_in(&resp)) {
                rate_limit_waits += 1;
                eprintln!("rate limited; sleeping {}s", wait.as_secs());
                tokio::time::sleep(wait).await;
                continue;
            }
        }

        if status.is_server_error() && attempt < 5 {
            let wait = Duration::from_secs(2u64.pow(attempt));
            attempt += 1;
            eprintln!("{status}; retrying in {}s", wait.as_secs());
            tokio::time::sleep(wait).await;
            continue;
        }

        let snippet = resp.text().await.unwrap_or_default();
        bail!("GET {url} -> {status}: {}", snippet.chars().take(300).collect::<String>());
    }
}

fn remaining(resp: &reqwest::Response) -> Option<u64> {
    resp.headers()
        .get("x-ratelimit-remaining")?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn reset_in(resp: &reqwest::Response) -> Option<Duration> {
    let reset: u64 = resp
        .headers()
        .get("x-ratelimit-reset")?
        .to_str()
        .ok()?
        .parse()
        .ok()?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(Duration::from_secs(reset.saturating_sub(now) + 1))
}

fn retry_after(resp: &reqwest::Response) -> Option<Duration> {
    let secs: u64 = resp
        .headers()
        .get("retry-after")?
        .to_str()
        .ok()?
        .parse()
        .ok()?;
    Some(Duration::from_secs(secs + 1))
}

async fn sleep_until_reset(resp: &reqwest::Response) {
    if let Some(wait) = reset_in(resp) {
        eprintln!("primary budget exhausted; sleeping {}s", wait.as_secs());
        tokio::time::sleep(wait).await;
    }
}

/// Extract the `rel="next"` URL from a GitHub `Link` header.
fn parse_next_link(header: Option<&reqwest::header::HeaderValue>) -> Option<String> {
    let raw = header?.to_str().ok()?;
    for part in raw.split(',') {
        if part.contains("rel=\"next\"") {
            let start = part.find('<')? + 1;
            let end = part.find('>')?;
            return Some(part[start..end].to_string());
        }
    }
    None
}
