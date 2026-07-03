//! Shared data types for the rustc-reviewer data pipeline.
//!
//! [`ReviewComment`] is the subset of GitHub's pull-request review-comment
//! payload that we persist. [`ChatRecord`] is the training-ready chat format
//! emitted by `reviewer-prepare`.

use serde::{Deserialize, Serialize};

/// A single pull-request review comment, as returned by
/// `GET /repos/{owner}/{repo}/pulls/comments`.
///
/// We deserialize only the fields we need and re-serialize the same shape, so
/// `data/raw/*.jsonl` is a compact, stable record we can reprocess offline
/// without re-hitting the API.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReviewComment {
    pub id: u64,
    /// e.g. `https://api.github.com/repos/rust-lang/rust/pulls/12345`
    pub pull_request_url: String,
    pub path: Option<String>,
    /// The unified-diff hunk the comment is anchored to. This is the gold:
    /// a ready-made (code change -> reviewer comment) pair.
    pub diff_hunk: Option<String>,
    pub body: String,
    pub user: Option<User>,
    pub created_at: String,
    pub updated_at: String,
    /// Present when this comment is a reply within a thread. We use thread
    /// structure as a signal for design discussion (see `reviewer-prepare`).
    #[serde(default)]
    pub in_reply_to_id: Option<u64>,
    pub line: Option<u64>,
    pub original_line: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct User {
    pub login: String,
}

impl ReviewComment {
    /// The PR number parsed out of [`Self::pull_request_url`].
    pub fn pr_number(&self) -> Option<u64> {
        self.pull_request_url.rsplit('/').next()?.parse().ok()
    }

    pub fn login(&self) -> &str {
        self.user.as_ref().map(|u| u.login.as_str()).unwrap_or("")
    }
}

/// Training-ready chat record: `{"messages": [...]}` — the de-facto SFT format
/// consumed by every trainer (and by a future all-Rust `candle`/`burn` loop).
#[derive(Debug, Clone, Serialize)]
pub struct ChatRecord {
    pub messages: Vec<Message>,
    /// Provenance + curation metadata. Ignored by trainers; invaluable for
    /// auditing the dataset and for ablations.
    pub meta: Meta,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

/// System prompt for the design-review model.
///
/// Shared by `reviewer-prepare` (training-data construction) and the inference
/// harness (`reviewer-run`) so the model sees the *identical* instruction at
/// train and serve time. Editing this is a train/serve skew hazard: the adapter
/// was conditioned on this exact string, so changing it means retraining.
pub const SYSTEM: &str = "You are a senior reviewer for the Rust project. You look \
for design problems — API shape, abstractions, invariants, edge cases, \
backwards-compatibility, and maintainability — not formatting nits. Given a \
diff hunk from a pull request, write the review comment a maintainer would \
leave, or say it looks good if there is nothing to raise.";

/// Build the user-turn content for one diff hunk, in the exact shape the model
/// was trained on.
///
/// Both the trainer (`reviewer-prepare`) and the inference harness
/// (`reviewer-run`) call this, so the wire format can never drift between train
/// and serve. If you change the layout here, you change it for both — which is
/// the whole point.
pub fn user_prompt(repo: &str, pr: Option<u64>, path: &str, hunk: &str) -> String {
    format!(
        "Repository: {repo}\nPull request: #{}\nFile: {path}\n\n```diff\n{}\n```",
        pr.map(|n| n.to_string()).unwrap_or_default(),
        hunk.trim_end(),
    )
}

/// One thing the critic (the LoRA reviewer) said about one hunk — persisted as
/// the harness's durable record.
///
/// This is the substrate for the specialist-stack thesis: the HTML report is
/// just a *view* of a stream of these, and a `CriticFinding` carries everything
/// three downstream consumers need — enough to (a) re-render the report, (b)
/// collect a human verdict, and (c) become a **training row for a second
/// "judge" model** that learns to predict which critic findings a human keeps.
/// If it isn't captured here, it's lost: you cannot train a judge on comments
/// you didn't save.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticFinding {
    /// Stable content hash — dedups re-runs, keys human labels to findings.
    pub finding_id: String,
    /// One review run (one `reviewer-run review` invocation).
    pub session_id: String,
    /// Which critic produced this — adapter/checkpoint tag, e.g.
    /// `reviewer-lora@epoch2` or `stub`. **Load-bearing:** findings from
    /// different checkpoints are different distributions, and the judge's
    /// training data has to know which critic spoke.
    pub model_version: String,
    pub created_at: String,

    pub repo: String,
    pub pr: Option<u64>,
    pub path: String,
    pub hunk_header: String,
    /// Verbatim hunk fed to the critic — replayable for retraining / re-render.
    pub hunk_raw: String,
    /// The exact `user_prompt` string the critic saw (train/serve record).
    pub prompt: String,

    /// The critic's raw output.
    pub critic_comment: String,
    /// Line the comment cites, if one could be parsed from it.
    pub cited_line: Option<u64>,
    /// Deterministic safety-net check: if the comment cites a line, does that
    /// line actually fall inside the hunk? `true` when it does — or when the
    /// comment cites no line at all (nothing to disprove). Only a citation of a
    /// line *not* in the hunk makes this `false`. Not a quality judgment.
    pub grounded: bool,

    /// Human verdict — `None` until reviewed. This is the label the judge trains on.
    #[serde(default)]
    pub human: Option<HumanLabel>,
}

/// A human's judgment of one [`CriticFinding`] — the training target for the
/// judge model, and the accept/reject that drives the human-in-the-loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanLabel {
    pub verdict: Verdict,
    /// Whether this is genuinely a *design* problem (vs a nit or a non-issue).
    /// `None` for `Unsure` — don't bake a confident negative into the judge's
    /// training data for a case the human explicitly declined to judge.
    pub is_design_problem: Option<bool>,
    #[serde(default)]
    pub severity: Option<String>,
    #[serde(default)]
    pub note: Option<String>,
    pub judged_at: String,
    pub judged_by: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Accept,
    Reject,
    Unsure,
}

#[derive(Debug, Clone, Serialize)]
pub struct Meta {
    pub source_id: u64,
    pub repo: String,
    pub pr: Option<u64>,
    pub path: Option<String>,
    /// Heuristic category guess: see `reviewer-prepare`.
    pub category: String,
    /// Heuristic "is this design feedback?" score in [0.0, 1.0].
    pub design_score: f32,
    /// Number of replies this comment received (thread depth signal).
    pub reply_count: u32,
}
