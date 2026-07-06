//! Stages 3+4 output and its persistence. Runs the critic over every hunk,
//! applies the deterministic grounding check, and turns the result into
//! [`CriticFinding`] records — the durable stream that the HTML report is a mere
//! view of, and that the judge model will eventually train on.

use std::collections::VecDeque;
use std::io::Write as _;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures::stream::{FuturesUnordered, StreamExt};
use reviewer_core::CriticFinding;

use crate::concurrency::AdaptiveLimiter;
use crate::critic::{Critic, RawComment};
use crate::diff::{FileDiff, Hunk};

/// One unit of work: a hunk plus the exact prompt it'll be reviewed with.
struct Job<'a> {
    path: &'a str,
    hunk: &'a Hunk,
    prompt: String,
}

/// Run `critic` over every hunk in `files` under an adaptive concurrency limit,
/// grounding and recording each comment. Generic (not `dyn`) so the critic's
/// `review` can be a native `async fn`.
///
/// Concurrency is I/O-side: many reviews are kept in flight on one task to feed
/// the server's continuous batching — no client CPU threads. The `limiter`
/// decides how many; it's mutated in place, so afterward it holds what it
/// learned (settled limit, trajectory). Output order is stable regardless of
/// completion order.
pub async fn collect<C: Critic>(
    critic: &C,
    repo: &str,
    pr: u64,
    session_id: &str,
    files: &[FileDiff],
    limiter: &mut AdaptiveLimiter,
) -> Result<(Vec<CriticFinding>, usize)> {
    let created_at = chrono::Utc::now().to_rfc3339();

    let jobs: Vec<Job> = files
        .iter()
        .flat_map(|f| {
            f.hunks.iter().map(move |h| Job {
                path: f.path.as_str(),
                hunk: h,
                prompt: reviewer_core::user_prompt(repo, Some(pr), &f.path, &h.raw),
            })
        })
        .collect();

    // Reviews complete out of order; slot each result back by index. A ready
    // queue feeds a single launch site (so all in-flight futures share one type)
    // and lets a failed hunk simply re-enqueue itself for one retry.
    let mut results: Vec<Option<Vec<RawComment>>> = (0..jobs.len()).map(|_| None).collect();
    let mut attempts = vec![0u8; jobs.len()];
    let mut failures = 0usize;
    let mut ready: VecDeque<(usize, Option<Duration>)> =
        (0..jobs.len()).map(|i| (i, None)).collect();
    let mut inflight = FuturesUnordered::new();

    while !ready.is_empty() || !inflight.is_empty() {
        // Top up to the *current* adaptive limit (it moves as results arrive).
        while inflight.len() < limiter.limit() {
            let Some((idx, backoff)) = ready.pop_front() else {
                break;
            };
            let job = &jobs[idx];
            let start = Instant::now();
            inflight.push(async move {
                if let Some(d) = backoff {
                    tokio::time::sleep(d).await;
                }
                let r = critic.review(&job.prompt, job.hunk).await;
                (idx, start.elapsed(), r)
            });
            attempts[idx] += 1;
        }
        let Some((idx, rtt, r)) = inflight.next().await else {
            continue;
        };
        match r {
            Ok(comments) => {
                limiter.on_success(rtt); // latency feeds the gradient
                results[idx] = Some(comments);
            }
            Err(e) => {
                // Only *congestion* throttles the limiter — a 401/decode error
                // would otherwise make it needlessly slow a doomed run.
                if e.is_overload() {
                    limiter.on_error();
                }
                if attempts[idx] < 2 {
                    // Retry once (a failed hunk might have held the real finding).
                    eprintln!("  ~ hunk {idx} ({}) failed ({e}); retrying", jobs[idx].path);
                    let backoff = e.is_overload().then(|| Duration::from_millis(500));
                    ready.push_back((idx, backoff));
                } else {
                    failures += 1;
                    eprintln!("  ! hunk {idx} ({}) failed after retry: {e}", jobs[idx].path);
                    results[idx] = Some(Vec::new());
                }
            }
        }
    }

    // Assemble findings in stable (file, hunk) order.
    let mut out = Vec::new();
    for (idx, job) in jobs.iter().enumerate() {
        for c in results[idx].take().unwrap_or_default() {
            let grounded = is_grounded(c.cited_line, job.hunk);
            let finding_id = finding_id(
                critic.model_version(),
                repo,
                pr,
                job.path,
                &job.hunk.header,
                &c.body,
            );
            out.push(CriticFinding {
                finding_id,
                session_id: session_id.to_string(),
                model_version: critic.model_version().to_string(),
                created_at: created_at.clone(),
                repo: repo.to_string(),
                pr: Some(pr),
                path: job.path.to_string(),
                hunk_header: job.hunk.header.clone(),
                hunk_raw: job.hunk.raw.clone(),
                prompt: job.prompt.clone(),
                critic_comment: c.body,
                cited_line: c.cited_line,
                grounded,
                human: None,
                machine: None,
            });
        }
    }
    Ok((out, failures))
}

/// Grounding is a safety net, not a quality judgment: a comment is grounded
/// unless it cites a line the hunk contradicts. No citation ⇒ nothing to
/// disprove ⇒ grounded.
fn is_grounded(cited_line: Option<u64>, hunk: &Hunk) -> bool {
    match cited_line {
        None => true,
        Some(l) => {
            let (lo, hi) = hunk.new_line_range();
            l >= lo && l <= hi
        }
    }
}

/// Stable content hash keyed on *what was said about what* — the critic, the
/// hunk, and the comment — but **not** the session. So re-running `review` on
/// the same PR yields the same ids (idempotent capture), letting [`merge`]
/// preserve human labels across runs. Provenance (which session first saw it)
/// lives in the record's fields, not its identity.
///
/// FNV-1a, deliberately: `DefaultHasher`'s algorithm is *not* guaranteed stable
/// across Rust releases, and label preservation depends on this id being
/// reproducible forever — a toolchain bump must not silently re-hash every
/// finding and orphan the human verdicts.
fn finding_id(
    model_version: &str,
    repo: &str,
    pr: u64,
    path: &str,
    hunk_header: &str,
    body: &str,
) -> String {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let pr = pr.to_string();
    let mut h = OFFSET;
    for field in [model_version, repo, pr.as_str(), path, hunk_header, body] {
        for &b in field.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        // Field separator so ("ab","c") and ("a","bc") don't collide.
        h ^= 0xff;
        h = h.wrapping_mul(PRIME);
    }
    format!("{h:016x}")
}

/// Merge freshly-collected findings into the durable store and persist it,
/// preserving any existing human labels: a re-run of `review` refreshes the
/// record without clobbering verdicts or creating duplicates. Returns the full
/// merged store.
pub fn merge(path: &Path, fresh: &[CriticFinding]) -> Result<Vec<CriticFinding>> {
    let mut store = if path.exists() {
        load(path)?
    } else {
        Vec::new()
    };
    for f in fresh {
        match store.iter_mut().find(|s| s.finding_id == f.finding_id) {
            Some(existing) => {
                // Keep the human verdict; refresh everything else.
                let human = existing.human.take();
                *existing = f.clone();
                existing.human = human;
            }
            None => store.push(f.clone()),
        }
    }
    save(path, &store)?;
    Ok(store)
}

/// Load all findings from a session JSONL.
pub fn load(path: &Path) -> Result<Vec<CriticFinding>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).context("decoding finding"))
        .collect()
}

/// Rewrite a session JSONL wholesale — used after labeling mutates `human`.
///
/// Atomic: write a sibling temp file, fsync it, then `rename` it over the
/// original. A crash or Ctrl-C mid-write leaves the old file intact rather than
/// truncating the irreplaceable record (there is no re-deriving a human verdict).
pub fn save(path: &Path, findings: &[CriticFinding]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut f = std::fs::File::create(&tmp)
            .with_context(|| format!("writing {}", tmp.display()))?;
        for finding in findings {
            serde_json::to_writer(&mut f, finding)?;
            f.write_all(b"\n")?;
        }
        f.flush()?;
        f.sync_all()?; // durable on disk before the rename swings the pointer
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::critic::StubCritic;
    use crate::diff;
    use reviewer_core::{HumanLabel, Verdict};

    fn limiter() -> AdaptiveLimiter {
        AdaptiveLimiter::new(1, 8, 2)
    }

    const SAMPLE: &str = "\
diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,2 +10,3 @@ impl Widget {
     pub fn area(&self) -> u32 {
+        self.w.saturating_mul(self.h)
     }
";

    #[tokio::test]
    async fn stub_produces_grounded_findings() {
        let files = diff::parse(SAMPLE);
        let (f, failures) = collect(&StubCritic, "rust-lang/rust", 1, "sess1", &files, &mut limiter())
            .await
            .unwrap();
        assert_eq!(failures, 0);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].model_version, "stub");
        assert!(f[0].grounded, "stub cites a real added line, so grounded");
        assert!(f[0].human.is_none());
        assert!(f[0].prompt.starts_with("Repository: rust-lang/rust"));
    }

    #[test]
    fn grounding_rejects_out_of_range_citation() {
        let files = diff::parse(SAMPLE);
        let hunk = &files[0].hunks[0];
        assert!(is_grounded(Some(11), hunk)); // inside the hunk
        assert!(is_grounded(None, hunk)); // no citation ⇒ grounded
        assert!(!is_grounded(Some(9999), hunk)); // fabricated line ⇒ not grounded
    }

    #[tokio::test]
    async fn finding_id_is_stable_across_sessions() {
        let files = diff::parse(SAMPLE);
        let (a, _) = collect(&StubCritic, "r", 1, "sess1", &files, &mut limiter())
            .await
            .unwrap();
        let (b, _) = collect(&StubCritic, "r", 1, "sess2", &files, &mut limiter())
            .await
            .unwrap();
        // Same critic + hunk + comment ⇒ same id, regardless of session — so a
        // re-run can dedupe and preserve labels.
        assert_eq!(a[0].finding_id, b[0].finding_id);
        // Different PR ⇒ different id.
        let (c, _) = collect(&StubCritic, "r", 2, "sess1", &files, &mut limiter())
            .await
            .unwrap();
        assert_ne!(a[0].finding_id, c[0].finding_id);
    }

    #[tokio::test]
    async fn merge_preserves_labels_and_dedupes() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rr-findings-{}.jsonl", std::process::id()));
        std::fs::remove_file(&path).ok();
        let files = diff::parse(SAMPLE);

        // First run: capture, then a human labels the finding.
        let (mut f, _) = collect(&StubCritic, "r", 1, "sess1", &files, &mut limiter())
            .await
            .unwrap();
        merge(&path, &f).unwrap();
        f[0].human = Some(HumanLabel {
            verdict: Verdict::Accept,
            is_design_problem: Some(true),
            severity: Some("medium".into()),
            note: None,
            judged_at: "now".into(),
            judged_by: "andy".into(),
        });
        save(&path, &f).unwrap();

        // Second run of the *same* review: must not duplicate, must keep the label.
        let (again, _) = collect(&StubCritic, "r", 1, "sess2", &files, &mut limiter())
            .await
            .unwrap();
        let merged = merge(&path, &again).unwrap();
        assert_eq!(merged.len(), 1, "dedup by stable finding_id");
        assert_eq!(merged[0].human.as_ref().unwrap().verdict, Verdict::Accept);

        let back = load(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].human.as_ref().unwrap().verdict, Verdict::Accept);
        std::fs::remove_file(&path).ok();
    }
}
