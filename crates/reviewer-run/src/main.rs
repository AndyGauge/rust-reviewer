//! The review harness. Two subcommands:
//!
//! * `review` — stages 1–6: fetch a PR, segment its diff, run the critic over
//!   every hunk, ground-check and **persist** each finding to a durable JSONL,
//!   and render an HTML *view* of that record. The critic is a [`StubCritic`]
//!   until the adapter lands (`model_version = "stub"`).
//! * `label` — stage 7: the human-in-the-loop. Walk the unjudged findings,
//!   record accept/reject/unsure, and write the verdicts back. The labeled
//!   stream `(hunk + critic_comment) → verdict` is the judge model's training set.
//!
//! ```text
//! GITHUB_TOKEN=ghp_xxx cargo run -p reviewer-run -- review \
//!     --repo rust-lang/rust --pr 12345
//! cargo run -p reviewer-run -- label --findings findings.jsonl
//! ```

mod concurrency;
mod critic;
mod diff;
mod findings;
mod github;
mod judge;
mod render;

use concurrency::AdaptiveLimiter;

use std::io::{BufRead, Write as _};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use critic::StubCritic;
use reviewer_core::{HumanLabel, Verdict};

#[derive(Parser)]
#[command(about = "Fetch, review, and human-judge a PR — capturing the critic's findings")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fetch + segment + critic + ground + persist + render.
    Review(ReviewArgs),
    /// Human-in-the-loop: annotate findings with verdicts (the judge's labels).
    Label(LabelArgs),
    /// Automated judge: a judge model renders a verdict on every finding, in
    /// parallel, recorded in the `machine` field (kept apart from human labels).
    Judge(JudgeArgs),
}

#[derive(Parser)]
struct ReviewArgs {
    /// `owner/name`, e.g. `rust-lang/rust` or `rust-lang-nursery/rust-cookbook`.
    #[arg(long)]
    repo: String,
    /// Pull-request number.
    #[arg(long)]
    pr: u64,
    /// GitHub token (public-repo read is enough).
    #[arg(long, env = "GITHUB_TOKEN")]
    token: String,
    /// HTML report path (a rendered view of the findings).
    #[arg(long, default_value = "review.html")]
    out: PathBuf,
    /// Durable findings JSONL — the record the report is a view of, and the
    /// judge's eventual training data. Appended to across runs.
    #[arg(long, default_value = "findings.jsonl")]
    findings: PathBuf,
    /// Also write the exact per-hunk model prompts here (train/serve parity check).
    #[arg(long)]
    dump_prompts: Option<PathBuf>,

    /// OpenAI-compatible base URL, e.g. `http://spark:8000/v1`. When set, the
    /// real critic (HttpCritic) is used instead of the stub.
    #[arg(long)]
    endpoint: Option<String>,
    /// Served model name (the `model` field in the request).
    #[arg(long, default_value = "reviewer")]
    model: String,
    /// Bearer token for the endpoint, if it requires one.
    #[arg(long, env = "REVIEWER_API_KEY")]
    api_key: Option<String>,
    /// Checkpoint tag stamped on every finding, e.g. `reviewer-lora@epoch3`.
    #[arg(long, default_value = "http")]
    model_version: String,

    /// Adaptive concurrency floor / ceiling / starting point. The harness learns
    /// the working number between these by probing latency; the ceiling is the
    /// KV-cache/OOM guard. (1/1/1 pins it to sequential.)
    #[arg(long, default_value_t = 1)]
    concurrency_min: usize,
    #[arg(long, default_value_t = 12)]
    concurrency_max: usize,
    #[arg(long, default_value_t = 2)]
    concurrency_start: usize,
}

#[derive(Parser)]
struct LabelArgs {
    /// The findings JSONL to annotate in place.
    #[arg(long, default_value = "findings.jsonl")]
    findings: PathBuf,
    /// Recorded as `judged_by` on each verdict.
    #[arg(long, env = "USER", default_value = "unknown")]
    judged_by: String,
}

#[derive(Parser)]
struct JudgeArgs {
    /// The findings JSONL to judge in place (verdicts written to `machine`).
    #[arg(long, default_value = "findings.jsonl")]
    findings: PathBuf,
    /// OpenAI-compatible base URL of the judge model, e.g. `http://spark:8001/v1`.
    #[arg(long)]
    endpoint: String,
    /// Judge model name — the base model, distinct from the critic LoRA.
    #[arg(long, default_value = "Qwen/Qwen3.6-27B")]
    model: String,
    /// Bearer token for the endpoint, if it requires one.
    #[arg(long, env = "REVIEWER_API_KEY")]
    api_key: Option<String>,
    /// Recorded as `judged_by` on each machine verdict.
    #[arg(long, default_value = "Qwen3.6-27B")]
    judged_by: String,
    /// How many findings to judge concurrently (the endpoint batches them).
    #[arg(long, default_value_t = 8)]
    concurrency: usize,
    /// Re-judge findings that already carry a machine verdict.
    #[arg(long)]
    rejudge: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Review(a) => review(a).await,
        Cmd::Label(a) => label(a),
        Cmd::Judge(a) => judge_cmd(a).await,
    }
}

/// Stage 7 (automated): run a judge model over every unjudged finding in parallel.
async fn judge_cmd(args: JudgeArgs) -> Result<()> {
    let mut items = findings::load(&args.findings)?;
    let pending = items
        .iter()
        .filter(|f| args.rejudge || f.machine.is_none())
        .count();
    if pending == 0 {
        eprintln!("all {} findings already judged (use --rejudge to redo).", items.len());
        return Ok(());
    }
    let judge = judge::HttpJudge::new(&args.endpoint, &args.model, args.api_key.clone(), &args.judged_by)?;
    eprintln!(
        "judging {pending} of {} findings with {} (concurrency {}) …",
        items.len(),
        args.model,
        args.concurrency,
    );
    let t = std::time::Instant::now();
    let n = judge::judge_all(&judge, &mut items, args.concurrency, args.rejudge).await?;
    findings::save(&args.findings, &items)?;

    let (mut acc, mut rej, mut uns) = (0usize, 0usize, 0usize);
    for f in &items {
        if let Some(m) = &f.machine {
            match m.verdict {
                Verdict::Accept => acc += 1,
                Verdict::Reject => rej += 1,
                Verdict::Unsure => uns += 1,
            }
        }
    }
    eprintln!(
        "judged {n} in {:.1}s · accept {acc} · reject {rej} · unsure {uns} -> {}",
        t.elapsed().as_secs_f64(),
        args.findings.display(),
    );
    Ok(())
}

async fn review(args: ReviewArgs) -> Result<()> {
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

    if let Some(path) = &args.dump_prompts {
        dump_prompts(path, &args, &files)?;
    }

    // Stages 3+4: critic + ground. HttpCritic if an endpoint was given, else stub.
    // Concurrency is learned by the limiter as reviews stream (see concurrency.rs).
    let session_id = format!("{}-{}-{}", args.repo.replace('/', "_"), args.pr, now_compact());
    let mut limiter = AdaptiveLimiter::new(
        args.concurrency_min,
        args.concurrency_max,
        args.concurrency_start,
    );
    let (found, failures) = if let Some(endpoint) = &args.endpoint {
        eprintln!("  critic: {} [{}]", endpoint, args.model_version);
        let critic = critic::HttpCritic::new(
            endpoint,
            &args.model,
            args.api_key.clone(),
            &args.model_version,
        )?;
        findings::collect(&critic, &args.repo, args.pr, &session_id, &files, &mut limiter).await?
    } else {
        eprintln!("  critic: stub (no --endpoint given)");
        findings::collect(&StubCritic, &args.repo, args.pr, &session_id, &files, &mut limiter).await?
    };
    let grounded = found.iter().filter(|f| f.grounded).count();
    eprintln!(
        "  critic [{}]: {} findings ({} grounded)",
        critic_tag(&found),
        found.len(),
        grounded,
    );
    if failures > 0 {
        eprintln!(
            "  ! {failures} hunk(s) failed to review after retry — this run is INCOMPLETE"
        );
    }
    eprintln!(
        "  concurrency: learned ~{} in [{}, {}] (min service {:.0}ms, {} adjustments)",
        limiter.settled(),
        args.concurrency_min,
        args.concurrency_max,
        limiter.min_rtt_ms(),
        limiter.trajectory().len(),
    );

    // Merge into the durable record (dedupes, preserves prior human labels),
    // THEN render a view of *this PR's* findings — including any verdicts
    // recorded on a previous run.
    let store = findings::merge(&args.findings, &found)?;
    let judged = store
        .iter()
        .filter(|f| f.pr == Some(args.pr) && f.repo == args.repo && f.human.is_some())
        .count();
    eprintln!(
        "  findings -> {} ({} total records, {} judged for this PR)",
        args.findings.display(),
        store.len(),
        judged,
    );
    let this_pr: Vec<reviewer_core::CriticFinding> = store
        .into_iter()
        .filter(|f| f.pr == Some(args.pr) && f.repo == args.repo)
        .collect();

    let html = render::report(&pr, &files, &this_pr, failures);
    std::fs::write(&args.out, &html)
        .with_context(|| format!("writing {}", args.out.display()))?;
    eprintln!("report   -> {}", args.out.display());
    eprintln!("open:    file://{}", args.out.canonicalize()?.display());
    eprintln!("next:    reviewer-run label --findings {}", args.findings.display());
    Ok(())
}

/// Stage 7: walk unjudged findings on stdin and record verdicts.
fn label(args: LabelArgs) -> Result<()> {
    let mut items = findings::load(&args.findings)?;
    let pending: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, f)| f.human.is_none())
        .map(|(i, _)| i)
        .collect();

    if pending.is_empty() {
        eprintln!("all {} findings already judged.", items.len());
        return Ok(());
    }
    eprintln!(
        "{} unjudged of {} findings. [a]ccept [r]eject [u]nsure [s]kip [q]uit; \
         then optional note + Enter.\n",
        pending.len(),
        items.len(),
    );

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut judged = 0usize;
    for (n, &idx) in pending.iter().enumerate() {
        let f = &items[idx];
        println!("── {}/{}  {} · {} ──", n + 1, pending.len(), f.repo, f.path);
        println!("   hunk: {}", f.hunk_header);
        if !f.grounded {
            println!("   ⚠ ungrounded (cites a line not in the hunk)");
        }
        println!("   critic: {}", f.critic_comment.trim());
        print!("   verdict [a/r/u/s/q]> ");
        std::io::stdout().flush().ok();

        let Some(line) = lines.next() else { break };
        let choice = line?.trim().to_lowercase();
        let (verdict, is_design) = match choice.chars().next() {
            Some('a') => (Verdict::Accept, Some(true)),
            Some('r') => (Verdict::Reject, Some(false)),
            Some('u') => (Verdict::Unsure, None), // declined to judge — no label
            Some('q') => break,
            _ => {
                println!("   (skipped)\n");
                continue;
            }
        };

        print!("   note (Enter to skip)> ");
        std::io::stdout().flush().ok();
        let note = lines
            .next()
            .transpose()?
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        items[idx].human = Some(HumanLabel {
            verdict,
            is_design_problem: is_design,
            severity: None,
            note,
            judged_at: chrono::Utc::now().to_rfc3339(),
            judged_by: args.judged_by.clone(),
        });
        judged += 1;
        println!();
    }

    findings::save(&args.findings, &items)?;
    let total_judged = items.iter().filter(|f| f.human.is_some()).count();
    eprintln!(
        "recorded {judged} this session · {total_judged}/{} judged overall -> {}",
        items.len(),
        args.findings.display(),
    );
    Ok(())
}

fn dump_prompts(path: &PathBuf, args: &ReviewArgs, files: &[diff::FileDiff]) -> Result<()> {
    let mut f = std::fs::File::create(path)
        .with_context(|| format!("creating {}", path.display()))?;
    let mut n = 0;
    for file in files {
        for h in &file.hunks {
            let user = reviewer_core::user_prompt(&args.repo, Some(args.pr), &file.path, &h.raw);
            let rec = serde_json::json!({
                "system": reviewer_core::SYSTEM, "user": user,
                "path": file.path, "hunk_header": h.header,
            });
            serde_json::to_writer(&mut f, &rec)?;
            f.write_all(b"\n")?;
            n += 1;
        }
    }
    eprintln!("  wrote {n} model prompts -> {}", path.display());
    Ok(())
}

fn critic_tag(found: &[reviewer_core::CriticFinding]) -> &str {
    found.first().map(|f| f.model_version.as_str()).unwrap_or("none")
}

fn now_compact() -> String {
    chrono::Utc::now().format("%Y%m%dT%H%M%S").to_string()
}
