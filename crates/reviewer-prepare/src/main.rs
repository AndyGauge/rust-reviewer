//! Turn raw review comments into a training-ready chat dataset, biased toward
//! *design* feedback rather than nits.
//!
//! ```text
//! cargo run -p reviewer-prepare -- \
//!     --in data/raw/rust.jsonl --in data/raw/cookbook.jsonl \
//!     --out data/prepared/train.jsonl \
//!     --min-design-score 0.5
//! ```
//!
//! Two passes over the input: first count replies per thread (thread depth is
//! our strongest cheap signal for design discussion), then build, score, and
//! filter one record per *root* comment.

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use reviewer_core::{ChatRecord, Message, Meta, ReviewComment};

#[derive(Parser)]
#[command(about = "Clean + format review comments into chat JSONL")]
struct Args {
    /// One or more raw JSONL files from `reviewer-extract`.
    #[arg(long = "in", required = true)]
    inputs: Vec<PathBuf>,

    #[arg(long)]
    out: PathBuf,

    /// Drop records whose heuristic design score is below this (0.0–1.0).
    #[arg(long, default_value_t = 0.0)]
    min_design_score: f32,

    /// Also drop records shorter than this many characters after cleaning.
    #[arg(long, default_value_t = 20)]
    min_len: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let lex = Lexicon::new();

    // Load everything once; rustc's comment corpus fits comfortably in memory.
    let mut all: Vec<ReviewComment> = Vec::new();
    for path in &args.inputs {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            all.push(serde_json::from_str(line).context("decoding raw comment")?);
        }
    }
    eprintln!("loaded {} raw comments", all.len());

    // Pass 1: reply counts keyed by the root comment id.
    let mut replies: HashMap<u64, u32> = HashMap::new();
    for c in &all {
        if let Some(root) = c.in_reply_to_id {
            *replies.entry(root).or_default() += 1;
        }
    }

    // Pass 2: build records from root comments only.
    let mut out = std::fs::File::create(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;
    let mut stats = Stats::default();
    // Resuming a crawl re-fetches the inclusive `since` boundary, so the raw
    // file can hold duplicate ids. Keep the first occurrence of each.
    let mut seen_ids: HashSet<u64> = HashSet::new();

    for c in &all {
        stats.seen += 1;
        if !seen_ids.insert(c.id) {
            stats.skip_dup += 1;
            continue;
        }
        if c.in_reply_to_id.is_some() {
            stats.skip_reply += 1;
            continue;
        }
        if is_bot(c.login()) {
            stats.skip_bot += 1;
            continue;
        }
        let (Some(path), Some(hunk)) = (c.path.clone(), c.diff_hunk.clone()) else {
            stats.skip_no_diff += 1;
            continue;
        };
        let body = clean_body(&c.body);
        if body.len() < args.min_len || is_trivial(&body, &lex) {
            stats.skip_trivial += 1;
            continue;
        }

        let reply_count = replies.get(&c.id).copied().unwrap_or(0);
        let (category, design_score) = score(&body, reply_count, &lex);
        if design_score < args.min_design_score {
            stats.skip_low_score += 1;
            continue;
        }

        let repo = repo_of(&c.pull_request_url).unwrap_or_default();
        // Shared with the inference harness via reviewer-core: identical wire
        // format at train and serve time (no skew).
        let user_content = reviewer_core::user_prompt(&repo, c.pr_number(), &path, &hunk);

        let record = ChatRecord {
            messages: vec![
                Message { role: "system", content: reviewer_core::SYSTEM.to_string() },
                Message { role: "user", content: user_content },
                Message { role: "assistant", content: body },
            ],
            meta: Meta {
                source_id: c.id,
                repo,
                pr: c.pr_number(),
                path: Some(path),
                category,
                design_score,
                reply_count,
            },
        };
        serde_json::to_writer(&mut out, &record)?;
        out.write_all(b"\n")?;
        stats.kept += 1;
    }
    out.flush()?;
    stats.report(&args.out);
    Ok(())
}

/// Compiled keyword/phrase signals. Built once.
struct Lexicon {
    design: Regex,
    nit: Regex,
    question: Regex,
    trivial: Regex,
}

impl Lexicon {
    fn new() -> Self {
        // (?i) case-insensitive. Phrases chosen to fire on design discussion.
        let design = Regex::new(r"(?i)\b(design|architectur|abstraction|\bapi\b|interface|invariant|coupl|trade-?off|alternative|instead of|this approach|refactor|generaliz|breaking change|backwards?[- ]compat|semver|footgun|edge case|maintainab|encapsulat|leaky|orthogonal|have you considered|why not|should (this|we|it)|do we (need|want)|is it worth|in the long run)\b").unwrap();
        let nit = Regex::new(r"(?i)\b(typo|nit:|whitespace|rustfmt|formatting|indentation|spelling|trailing|missing newline|rename this|tiny nit)\b").unwrap();
        let question = Regex::new(r"\?\s*$").unwrap();
        // Whole-comment trivialities with no training signal.
        let trivial = Regex::new(r"(?i)^\s*(lgtm|ditto|same|same here|done|thanks|thank you|nice|👍|:\+1:|\+1|r\?|@bors|@rustbot|@rust-timer)\b").unwrap();
        Self { design, nit, question, trivial }
    }
}

/// Transparent, tunable design score in [0,1]. Deliberately simple so it can be
/// explained in a blog post and ablated. A future pass can replace this with an
/// LLM-judge label distilled into the same `design_score` field.
fn score(body: &str, reply_count: u32, lex: &Lexicon) -> (String, f32) {
    let design_hits = lex.design.find_iter(body).count() as f32;
    let nit_hits = lex.nit.find_iter(body).count() as f32;
    let is_question = lex.question.is_match(body);

    let mut s = 0.0f32;
    s += (design_hits * 0.25).min(0.6); // keyword evidence, capped
    s += (reply_count as f32 * 0.12).min(0.36); // spawned a discussion
    s += (body.len() as f32 / 600.0).min(0.2); // substance ~ length
    if is_question {
        s += 0.1; // probing questions are often design feedback
    }
    s -= nit_hits * 0.3; // formatting talk pulls it down
    let s = s.clamp(0.0, 1.0);

    let category = if nit_hits > design_hits {
        "nit"
    } else if design_hits >= 1.0 || reply_count >= 2 {
        "design"
    } else if is_question {
        "question"
    } else {
        "other"
    };
    (category.to_string(), s)
}

fn clean_body(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
        let t = line.trim_end();
        // Drop quoted text (replies often quote the parent) and HTML comments.
        if t.trim_start().starts_with('>') {
            continue;
        }
        out.push_str(t);
        out.push('\n');
    }
    // Collapse runs of blank lines and trim.
    let collapsed: Vec<&str> = out.lines().collect();
    let mut result = String::new();
    let mut blank = false;
    for l in collapsed {
        if l.trim().is_empty() {
            if !blank {
                result.push('\n');
            }
            blank = true;
        } else {
            result.push_str(l);
            result.push('\n');
            blank = false;
        }
    }
    result.trim().to_string()
}

fn is_trivial(body: &str, lex: &Lexicon) -> bool {
    lex.trivial.is_match(body)
}

fn is_bot(login: &str) -> bool {
    const BOTS: &[&str] = &[
        "bors", "rust-highfive", "rustbot", "rust-timer", "rust-log-analyzer",
        "rust-cloud-vms", "craterbot", "rfcbot", "dependabot",
    ];
    login.ends_with("[bot]") || BOTS.contains(&login)
}

/// `https://api.github.com/repos/rust-lang/rust/pulls/123` -> `rust-lang/rust`
fn repo_of(pull_request_url: &str) -> Option<String> {
    let after = pull_request_url.split("/repos/").nth(1)?;
    let mut it = after.split('/');
    Some(format!("{}/{}", it.next()?, it.next()?))
}

#[derive(Default)]
struct Stats {
    seen: usize,
    kept: usize,
    skip_dup: usize,
    skip_reply: usize,
    skip_bot: usize,
    skip_no_diff: usize,
    skip_trivial: usize,
    skip_low_score: usize,
}

impl Stats {
    fn report(&self, out: &PathBuf) {
        eprintln!("--- prepare summary ---");
        eprintln!("seen           {}", self.seen);
        eprintln!("kept           {}  -> {}", self.kept, out.display());
        eprintln!("skip dup       {}", self.skip_dup);
        eprintln!("skip reply     {}", self.skip_reply);
        eprintln!("skip bot       {}", self.skip_bot);
        eprintln!("skip no-diff   {}", self.skip_no_diff);
        eprintln!("skip trivial   {}", self.skip_trivial);
        eprintln!("skip low-score {}", self.skip_low_score);
    }
}
