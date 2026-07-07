//! Stage 6: render the fetched + segmented PR into a self-contained HTML report
//! opened via `file://`. No network, no model. Until the review stages are
//! wired, the model column shows a "pending" banner — the deterministic spine
//! (diff, hunks, existing human comments, anchors) is fully rendered so it can
//! be validated on a real PR today.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use reviewer_core::{CriticFinding, Verdict};

use crate::diff::{FileDiff, LineKind};
use crate::github::FetchedPr;

pub fn report(
    pr: &FetchedPr,
    files: &[FileDiff],
    findings: &[CriticFinding],
    failures: usize,
) -> String {
    let m = &pr.meta;
    let hunks: usize = files.iter().map(|f| f.hunks.len()).sum();
    let author = m.user.as_ref().map(|u| u.login.as_str()).unwrap_or("?");

    // Group inline comments by file path for anchoring.
    let mut by_file: BTreeMap<&str, Vec<&reviewer_core::ReviewComment>> = BTreeMap::new();
    for c in &pr.review_comments {
        if let Some(p) = c.path.as_deref() {
            by_file.entry(p).or_default().push(c);
        }
    }
    // Group critic findings by file path too.
    let mut critic_by_file: BTreeMap<&str, Vec<&CriticFinding>> = BTreeMap::new();
    for f in findings {
        critic_by_file.entry(f.path.as_str()).or_default().push(f);
    }

    let mut s = String::new();
    s.push_str("<style>\n");
    s.push_str(CSS);
    s.push_str("</style>\n");

    // Header.
    let _ = write!(
        s,
        "<h1><a href=\"{}\">{} #{}</a></h1>\n<p class=\"title\">{}</p>\n",
        esc(&m.html_url),
        "PR",
        m.number,
        esc(&m.title),
    );
    let _ = write!(
        s,
        "<p class=\"meta\">by <b>{}</b> into <code>{}</code> · \
         {} files · <span class=\"add\">+{}</span> <span class=\"del\">−{}</span></p>\n",
        esc(author),
        esc(&m.base.name),
        m.changed_files,
        m.additions,
        m.deletions,
    );

    if let Some(body) = m.body.as_deref().map(str::trim).filter(|b| !b.is_empty()) {
        let _ = write!(s, "<div class=\"cmt\">{}</div>\n", esc(&truncate(body, 1200)));
    }

    // Critic summary. Distinguishes the stub from a real adapter, and reports
    // grounding + how much has been human-judged (the flywheel's fill level).
    let model = findings
        .first()
        .map(|f| f.model_version.as_str())
        .unwrap_or("none");
    let grounded = findings.iter().filter(|f| f.grounded).count();
    let judged = findings.iter().filter(|f| f.human.is_some()).count();
    let machine_judged = findings.iter().filter(|f| f.machine.is_some()).count();
    let stub_note = if model == "stub" {
        " <b>(stub — no adapter wired yet)</b>"
    } else {
        ""
    };
    let _ = write!(
        s,
        "<div class=\"banner\">Critic: <code>{}</code>{} · {} findings over {} files / \
         {} hunks · {} grounded · {} human-judged · {} machine-judged</div>\n",
        esc(model),
        stub_note,
        findings.len(),
        files.len(),
        hunks,
        grounded,
        judged,
        machine_judged,
    );

    // Incomplete-run warning: failed hunks might have held the real finding, so a
    // clean-looking report over a partial review would be quietly misleading.
    if failures > 0 {
        let _ = write!(
            s,
            "<div class=\"banner warn\">⚠ {failures} hunk(s) failed to review after retry — \
             this report is <b>incomplete</b>. A missing finding may have been the real one.</div>\n",
        );
    }

    // General discussion (issue comments).
    if !pr.issue_comments.is_empty() {
        s.push_str("<h2>Discussion</h2>\n");
        for c in &pr.issue_comments {
            let who = c.user.as_ref().map(|u| u.login.as_str()).unwrap_or("?");
            let _ = write!(
                s,
                "<div class=\"cmt\"><span class=\"who\">{}</span> {}</div>\n",
                esc(who),
                esc(&truncate(&c.body, 800)),
            );
        }
    }

    // Per-file diff with anchored existing comments.
    for f in files {
        let _ = write!(
            s,
            "<h2 class=\"file\"><span class=\"badge {}\">{}</span> <code>{}</code></h2>\n",
            f.status.label(),
            f.status.label(),
            esc(&f.path),
        );
        if f.binary {
            s.push_str("<p class=\"muted\">binary file, no textual diff</p>\n");
        }

        let existing = by_file.get(f.path.as_str());
        if let Some(cs) = existing {
            for c in cs {
                let who = c.user.as_ref().map(|u| u.login.as_str()).unwrap_or("?");
                let ln = c.line.or(c.original_line);
                let _ = write!(
                    s,
                    "<div class=\"cmt human\"><span class=\"who\">{}</span>{} {}</div>\n",
                    esc(who),
                    ln.map(|n| format!(" <span class=\"lno\">L{n}</span>")).unwrap_or_default(),
                    esc(&truncate(&c.body, 600)),
                );
            }
        }

        // Critic findings for this file — the persisted record, rendered.
        if let Some(cf) = critic_by_file.get(f.path.as_str()) {
            for finding in cf {
                let anchor = finding
                    .cited_line
                    .map(|n| format!(" <span class=\"lno\">L{n}</span>"))
                    .unwrap_or_default();
                let ground = if finding.grounded {
                    ""
                } else {
                    " <span class=\"tag ungrounded\">ungrounded</span>"
                };
                let human_tag = match finding.human.as_ref().map(|h| h.verdict) {
                    Some(Verdict::Accept) => " <span class=\"tag accept\">accepted</span>",
                    Some(Verdict::Reject) => " <span class=\"tag reject\">rejected</span>",
                    Some(Verdict::Unsure) => " <span class=\"tag unsure\">unsure</span>",
                    None => "",
                };
                // The judge model's verdict, shown as an outlined tag so it reads
                // as a distinct (machine) second opinion, not a human label.
                let machine_tag = match finding.machine.as_ref().map(|m| m.verdict) {
                    Some(Verdict::Accept) => " <span class=\"tag jaccept\">judge: accept</span>",
                    Some(Verdict::Reject) => " <span class=\"tag jreject\">judge: reject</span>",
                    Some(Verdict::Unsure) => " <span class=\"tag junsure\">judge: unsure</span>",
                    None => "",
                };
                let pending = if finding.human.is_none() && finding.machine.is_none() {
                    " <span class=\"tag pending\">unjudged</span>"
                } else {
                    ""
                };
                let _ = write!(
                    s,
                    "<div class=\"cmt critic\"><span class=\"who\">critic</span>{}{}{}{}{} {}</div>\n",
                    anchor,
                    ground,
                    human_tag,
                    machine_tag,
                    pending,
                    esc(&truncate(&finding.critic_comment, 800)),
                );
                // The judge's rationale, as its own voice under the critic's comment.
                if let Some(m) = &finding.machine {
                    if let Some(reason) = m.reason.as_deref().map(strip_verdict).filter(|r| !r.is_empty()) {
                        let _ = write!(
                            s,
                            "<div class=\"cmt judge-reason\"><span class=\"who\">{}</span>{}</div>\n",
                            esc(&m.judged_by),
                            esc(&truncate(reason, 600)),
                        );
                    }
                }
            }
        }

        for h in &f.hunks {
            s.push_str("<table class=\"hunk\">\n");
            let _ = write!(
                s,
                "<tr class=\"hdr\"><td colspan=\"3\">{}</td></tr>\n",
                esc(&h.header),
            );
            for line in &h.lines {
                let (cls, marker) = match line.kind {
                    LineKind::Add => ("add", "+"),
                    LineKind::Del => ("del", "−"),
                    LineKind::Context => ("ctx", " "),
                };
                let old = line.old_lineno.map(|n| n.to_string()).unwrap_or_default();
                let new = line.new_lineno.map(|n| n.to_string()).unwrap_or_default();
                let _ = write!(
                    s,
                    "<tr class=\"{cls}\"><td class=\"lno\">{old}</td><td class=\"lno\">{new}</td>\
                     <td class=\"code\">{}<span>{}</span></td></tr>\n",
                    marker,
                    esc(&line.text),
                );
            }
            s.push_str("</table>\n");
        }
    }

    s
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Drop a leading verdict word (ACCEPT/REJECT/UNSURE) the judge often prefixes
/// its rationale with, so the report shows only the reasoning.
fn strip_verdict(s: &str) -> &str {
    let t = s.trim_start();
    for w in ["ACCEPT", "REJECT", "UNSURE", "Accept", "Reject", "Unsure"] {
        if let Some(rest) = t.strip_prefix(w) {
            return rest.trim_start_matches([':', '.', ',', ' ', '\n', '\r', '-']);
        }
    }
    t
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(max).collect();
        t.push('…');
        t
    }
}

const CSS: &str = "\
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1000px;\
margin:2rem auto;padding:0 1rem;color:#1c1c1e}
h1{margin:.2rem 0}h1 a{color:inherit;text-decoration:none}
.title{font-size:1.1rem;color:#333;margin:.2rem 0}
.meta{color:#666}.add{color:#1a7f37}.del{color:#cf222e}.muted{color:#999}
code{background:#f0f0f2;padding:.1em .35em;border-radius:4px;font-size:.9em}
.banner{background:#fff8e1;border:1px solid #f0d000;border-radius:8px;padding:.6rem .9rem;margin:1rem 0}
.banner.warn{background:#ffebe9;border-color:#cf222e}
h2.file{margin-top:2rem;border-top:1px solid #e0e0e2;padding-top:1rem}
.badge{font-size:.7rem;text-transform:uppercase;padding:.15em .5em;border-radius:4px;color:#fff}
.badge.added{background:#1a7f37}.badge.deleted{background:#cf222e}
.badge.modified{background:#0969da}.badge.renamed{background:#8250df}
.cmt{background:#f6f8fa;border-left:3px solid #8250df;padding:.4rem .7rem;margin:.4rem 0;border-radius:0 6px 6px 0}
.cmt.human{border-left-color:#0969da}
.cmt.critic{border-left-color:#1a7f37;background:#f2fbf4}
.who{font-weight:600;margin-right:.5em}.lno{color:#999;font-size:.85em}
.tag{font-size:.7rem;padding:.1em .4em;border-radius:4px;margin-right:.3em}
.tag.ungrounded{background:#ffe0e0;color:#82071e}
.tag.accept{background:#d3f8d3;color:#0a5223}.tag.reject{background:#ffe0e0;color:#82071e}
.tag.unsure{background:#fff3cd;color:#7a5c00}.tag.pending{background:#eee;color:#666}
.tag.jaccept{background:transparent;border:1px solid #1a7f37;color:#0a5223}
.tag.jreject{background:transparent;border:1px solid #cf222e;color:#82071e}
.tag.junsure{background:transparent;border:1px solid #bf8700;color:#7a5c00}
.cmt.judge-reason{border-left-color:#8250df;background:#faf7ff;margin-left:1.4rem;color:#444;font-size:.95em}
.cmt.judge-reason .who{color:#8250df;font-weight:600}
table.hunk{border-collapse:collapse;width:100%;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;\
margin:.5rem 0;overflow-x:auto;display:block}
table.hunk td{padding:0 .5em;white-space:pre;vertical-align:top}
td.lno{color:#aaa;text-align:right;user-select:none;width:1%}
tr.hdr td{background:#f0f0f5;color:#57606a}
tr.add{background:#e6ffec}tr.del{background:#ffebe9}
tr.add .code span{color:#0a5223}tr.del .code span{color:#82071e}
";
