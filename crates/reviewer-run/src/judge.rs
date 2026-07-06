//! The automated judge — stage 7's model-driven sibling. Where `label` (in
//! `main.rs`) collects *human* verdicts one keypress at a time, this asks a
//! *judge model* to render a verdict on each finding, in parallel, and records
//! it in the `machine` field (never `human`, so the gold set stays clean).
//!
//! The judge is deliberately a *different* model from the critic — typically the
//! base model the LoRA was fine-tuned from. It has the same strong Rust prior but
//! isn't pattern-locked to the reviewer's voice, so it's a genuine second opinion:
//! it can, and does, reject comments the specialist critic was overconfident about.

use std::time::Duration;

use anyhow::{Context, Result};
use futures::stream::{self, StreamExt};
use reviewer_core::{CriticFinding, MachineLabel, Verdict};

/// The judge's instructions. It sees a diff hunk and the critic's comment on it,
/// and decides whether that comment is worth keeping. Verdict-first so the reply
/// parses deterministically even when the model adds prose.
const JUDGE_SYSTEM: &str = "You are a senior Rust maintainer acting as a second \
reviewer. Another reviewer has left a comment on a diff hunk. Decide whether that \
comment is correct and useful — a real design, correctness, or maintainability \
concern a maintainer would keep — or whether it is wrong, off-base, or a \
non-issue. Reply with exactly one word on the first line: ACCEPT if the comment is \
worth keeping, REJECT if it is wrong or not useful, or UNSURE if you genuinely \
cannot tell. Then, on the next line, give one sentence of reasoning.";

/// Talks to an OpenAI-compatible chat endpoint as the judge. Mirrors
/// [`crate::critic::HttpCritic`] but with a fixed judging prompt and a verdict
/// parser instead of free-text passthrough.
pub struct HttpJudge {
    http: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    /// Stamped as `judged_by` on every verdict — which model spoke.
    judged_by: String,
}

impl HttpJudge {
    pub fn new(
        endpoint: &str,
        model: &str,
        api_key: Option<String>,
        judged_by: &str,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(180))
            .build()?;
        Ok(Self {
            http,
            endpoint: endpoint.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            judged_by: judged_by.to_string(),
        })
    }

    /// Judge one finding: show the judge the hunk and the critic's comment, parse
    /// the verdict out of the reply.
    async fn verdict(&self, f: &CriticFinding) -> Result<MachineLabel> {
        let user = format!(
            "File: {}\n\n```diff\n{}\n```\n\nThe reviewer's comment on this hunk:\n{}",
            f.path,
            f.hunk_raw.trim_end(),
            f.critic_comment.trim(),
        );
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": JUDGE_SYSTEM },
                { "role": "user", "content": user },
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "stream": false,
            // The base model is a reasoning model; judging is a snap decision, so
            // skip the <think> pass for speed and a parseable verdict-first reply.
            "chat_template_kwargs": { "enable_thinking": false },
        });
        let url = format!("{}/chat/completions", self.endpoint);
        let mut req = self.http.post(&url).json(&body);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = req.send().await.context("judge request")?;
        let status = resp.status();
        let text = resp.text().await.context("reading judge response")?;
        if !status.is_success() {
            let snippet = text.chars().take(200).collect::<String>();
            anyhow::bail!("judge endpoint {url} -> {status}: {snippet}");
        }
        let out = extract_content(&text)?;
        let (verdict, reason) = parse_verdict(&out);
        Ok(MachineLabel {
            verdict,
            reason,
            judged_at: chrono::Utc::now().to_rfc3339(),
            judged_by: self.judged_by.clone(),
        })
    }
}

/// Judge every finding that needs it, `concurrency`-at-a-time so the server's
/// continuous batching stays fed. Writes each verdict into `machine` in place and
/// returns how many were judged. Findings that already carry a machine verdict are
/// skipped unless `rejudge`. A per-finding failure is logged and skipped, not fatal
/// — one bad hunk shouldn't sink a whole set.
pub async fn judge_all(
    judge: &HttpJudge,
    findings: &mut [CriticFinding],
    concurrency: usize,
    rejudge: bool,
) -> Result<usize> {
    // Snapshot the jobs as owned clones so the in-flight futures don't borrow
    // `findings` (which we need to mutate on write-back).
    let jobs: Vec<(usize, CriticFinding)> = findings
        .iter()
        .enumerate()
        .filter(|(_, f)| rejudge || f.machine.is_none())
        .map(|(i, f)| (i, f.clone()))
        .collect();

    let results: Vec<(usize, Result<MachineLabel>)> = stream::iter(
        jobs.into_iter()
            .map(|(i, f)| async move { (i, judge.verdict(&f).await) }),
    )
    .buffer_unordered(concurrency.max(1))
    .collect()
    .await;

    let mut judged = 0usize;
    for (i, r) in results {
        match r {
            Ok(label) => {
                findings[i].machine = Some(label);
                judged += 1;
            }
            Err(e) => eprintln!("  judge failed on {}: {e:#}", findings[i].finding_id),
        }
    }
    Ok(judged)
}

/// Pull the assistant text out of a chat-completions response. Falls back to
/// `reasoning_content` for servers that put the answer there with empty `content`.
fn extract_content(json: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(json).context("decoding judge completion")?;
    let msg = &v["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    let out = if content.trim().is_empty() {
        msg["reasoning_content"].as_str().unwrap_or("")
    } else {
        content
    };
    Ok(out.to_string())
}

/// Parse a verdict out of the judge's reply: the earliest of ACCEPT/REJECT/UNSURE
/// to appear wins (case-insensitive), defaulting to `Unsure` if none is present.
/// The whole trimmed reply is kept as the rationale.
fn parse_verdict(s: &str) -> (Verdict, Option<String>) {
    let up = s.to_uppercase();
    let verdict = [
        ("ACCEPT", Verdict::Accept),
        ("REJECT", Verdict::Reject),
        ("UNSURE", Verdict::Unsure),
    ]
    .into_iter()
    .filter_map(|(w, v)| up.find(w).map(|i| (i, v)))
    .min_by_key(|(i, _)| *i)
    .map(|(_, v)| v)
    .unwrap_or(Verdict::Unsure);
    let reason = s.trim();
    let reason = (!reason.is_empty()).then(|| reason.to_string());
    (verdict, reason)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_verdict_first() {
        let (v, r) = parse_verdict("REJECT\nTokio's recv is async, it does not block a thread.");
        assert_eq!(v, Verdict::Reject);
        assert!(r.unwrap().contains("async"));
    }

    #[test]
    fn earliest_keyword_wins_and_defaults_unsure() {
        assert_eq!(parse_verdict("ACCEPT, though one could reject it").0, Verdict::Accept);
        assert_eq!(parse_verdict("no clear verdict here").0, Verdict::Unsure);
    }
}
