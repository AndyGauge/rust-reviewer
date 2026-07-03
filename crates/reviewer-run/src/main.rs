//! The review harness (stages 1–2 + 6 of `docs/harness-plan.md`): fetch a PR,
//! segment its diff into hunks, and render an HTML report — with **no model
//! yet**. This proves the deterministic spine end to end so that wiring the
//! adapter (stage 3) is a single endpoint call.
//!
//! ```text
//! GITHUB_TOKEN=ghp_xxx cargo run -p reviewer-run -- \
//!     --repo rust-lang/rust --pr 12345 --out review.html
//! ```
//!
//! `--dump-prompts prompts.jsonl` additionally writes the exact user-turn each
//! hunk *would* be sent to the model as (via `reviewer_core::user_prompt`), so
//! train/serve parity can be inspected before the adapter exists.

mod diff;
mod github;
mod render;

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(about = "Fetch + segment a PR into a review-ready HTML report (no model yet)")]
struct Args {
    /// `owner/name`, e.g. `rust-lang/rust` or `rust-lang-nursery/rust-cookbook`.
    #[arg(long)]
    repo: String,

    /// Pull-request number.
    #[arg(long)]
    pr: u64,

    /// GitHub token (public-repo read is enough).
    #[arg(long, env = "GITHUB_TOKEN")]
    token: String,

    /// Output HTML report path.
    #[arg(long, default_value = "review.html")]
    out: PathBuf,

    /// Also write the exact per-hunk model prompts here (JSONL), for verifying
    /// train/serve parity before the adapter is wired in.
    #[arg(long)]
    dump_prompts: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Stage 1: fetch.
    let client = github::Client::new(args.token.clone())?;
    eprintln!("fetching {} #{} …", args.repo, args.pr);
    let pr = client.fetch(&args.repo, args.pr).await?;
    eprintln!(
        "  diff: {} bytes · {} inline comments · {} discussion comments",
        pr.diff.len(),
        pr.review_comments.len(),
        pr.issue_comments.len(),
    );

    // Stage 2: segment.
    let files = diff::parse(&pr.diff);
    let hunks: usize = files.iter().map(|f| f.hunks.len()).sum();
    eprintln!("  segmented: {} files, {} hunks", files.len(), hunks);

    // Optional: dump the exact model inputs (stage-3 preview).
    if let Some(path) = &args.dump_prompts {
        let mut f = std::fs::File::create(path)
            .with_context(|| format!("creating {}", path.display()))?;
        let mut n = 0;
        for file in &files {
            for h in &file.hunks {
                let user = reviewer_core::user_prompt(&args.repo, Some(args.pr), &file.path, &h.raw);
                let rec = serde_json::json!({
                    "system": reviewer_core::SYSTEM,
                    "user": user,
                    "path": file.path,
                    "hunk_header": h.header,
                });
                serde_json::to_writer(&mut f, &rec)?;
                f.write_all(b"\n")?;
                n += 1;
            }
        }
        eprintln!("  wrote {n} model prompts -> {}", path.display());
    }

    // Stage 6: render.
    let html = render::report(&pr, &files);
    std::fs::write(&args.out, html)
        .with_context(|| format!("writing {}", args.out.display()))?;
    eprintln!("report -> {}", args.out.display());
    eprintln!("open:   file://{}", args.out.canonicalize()?.display());

    Ok(())
}
