//! Stage 2: parse a unified diff (GitHub's `.diff` for a PR) into per-file,
//! per-hunk units. Pure and deterministic — no network, no model — so it's
//! unit-tested against recorded fixtures.
//!
//! A hunk is the unit the model trained on. The verbatim `raw` text of each
//! hunk is what gets handed to [`reviewer_core::user_prompt`] at review time,
//! so it must look exactly like the `diff_hunk` field GitHub attaches to a
//! review comment (a `@@ … @@` header line followed by ` `/`+`/`-` lines).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileStatus {
    Added,
    Deleted,
    Modified,
    Renamed,
}

impl FileStatus {
    pub fn label(&self) -> &'static str {
        match self {
            FileStatus::Added => "added",
            FileStatus::Deleted => "deleted",
            FileStatus::Modified => "modified",
            FileStatus::Renamed => "renamed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineKind {
    Context,
    Add,
    Del,
}

#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: LineKind,
    /// 1-based line number in the new file (None for deletions).
    pub new_lineno: Option<u64>,
    /// 1-based line number in the old file (None for additions).
    pub old_lineno: Option<u64>,
    /// Line text without the leading `+`/`-`/space marker.
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Hunk {
    /// The full `@@ -a,b +c,d @@ section heading` line.
    pub header: String,
    /// Starting line of this hunk in the new file.
    pub new_start: u64,
    pub lines: Vec<DiffLine>,
    /// Header + body, verbatim — exactly what we feed the model.
    pub raw: String,
}

impl Hunk {
    /// The line range this hunk covers in the new file, for anchoring comments.
    /// Used by stage 5 (distill) to match model comments to the hunk they cite.
    #[allow(dead_code)] // wired in when stage 3/5 land
    pub fn new_line_range(&self) -> (u64, u64) {
        let last = self
            .lines
            .iter()
            .filter_map(|l| l.new_lineno)
            .max()
            .unwrap_or(self.new_start);
        (self.new_start, last)
    }
}

#[derive(Debug, Clone)]
pub struct FileDiff {
    /// The reviewed path (new path for adds/mods, old path for deletions).
    pub path: String,
    pub old_path: Option<String>,
    pub status: FileStatus,
    pub binary: bool,
    pub hunks: Vec<Hunk>,
}

/// Parse a full unified diff into files. Tolerant of GitHub's extended-header
/// lines (`index`, `new file mode`, `rename from`, `Binary files …`).
pub fn parse(diff: &str) -> Vec<FileDiff> {
    let mut files: Vec<FileDiff> = Vec::new();
    let mut cur: Option<FileDiff> = None;
    let mut old_path_hdr: Option<String> = None;
    let mut new_path_hdr: Option<String> = None;
    // In-progress hunk state.
    let mut hunk: Option<Hunk> = None;
    let mut old_ln = 0u64;
    let mut new_ln = 0u64;

    // Close the open hunk (if any) into the current file.
    fn flush_hunk(cur: &mut Option<FileDiff>, hunk: &mut Option<Hunk>) {
        if let (Some(f), Some(h)) = (cur.as_mut(), hunk.take()) {
            f.hunks.push(h);
        }
    }
    // Finalize path/status of the current file from the ---/+++ headers.
    fn finalize_paths(
        cur: &mut Option<FileDiff>,
        old: &mut Option<String>,
        new: &mut Option<String>,
    ) {
        if let Some(f) = cur.as_mut() {
            let old_p = old.take();
            let new_p = new.take();
            match (old_p.as_deref(), new_p.as_deref()) {
                (Some("/dev/null"), Some(n)) => {
                    f.status = FileStatus::Added;
                    f.path = n.to_string();
                }
                (Some(o), Some("/dev/null")) => {
                    f.status = FileStatus::Deleted;
                    f.path = o.to_string();
                    f.old_path = Some(o.to_string());
                }
                (Some(o), Some(n)) => {
                    if o != n {
                        f.status = FileStatus::Renamed;
                        f.old_path = Some(o.to_string());
                    }
                    f.path = n.to_string();
                }
                _ => {}
            }
        }
    }

    for line in diff.lines() {
        if let Some(rest) = line.strip_prefix("diff --git ") {
            // New file section. Close out whatever's open.
            flush_hunk(&mut cur, &mut hunk);
            finalize_paths(&mut cur, &mut old_path_hdr, &mut new_path_hdr);
            if let Some(f) = cur.take() {
                files.push(f);
            }
            // Provisional path from `a/… b/…`; ---/+++ will correct it.
            let path = git_header_path(rest);
            cur = Some(FileDiff {
                path,
                old_path: None,
                status: FileStatus::Modified,
                binary: false,
                hunks: Vec::new(),
            });
            continue;
        }

        if line.starts_with("Binary files ") {
            if let Some(f) = cur.as_mut() {
                f.binary = true;
            }
            continue;
        }
        if let Some(p) = line.strip_prefix("--- ") {
            old_path_hdr = Some(strip_ab_prefix(p));
            continue;
        }
        if let Some(p) = line.strip_prefix("+++ ") {
            new_path_hdr = Some(strip_ab_prefix(p));
            finalize_paths(&mut cur, &mut old_path_hdr, &mut new_path_hdr);
            continue;
        }
        if line.starts_with("@@") {
            flush_hunk(&mut cur, &mut hunk);
            let (o, n) = parse_hunk_header(line);
            old_ln = o;
            new_ln = n;
            hunk = Some(Hunk {
                header: line.to_string(),
                new_start: n,
                lines: Vec::new(),
                raw: line.to_string(),
            });
            continue;
        }

        // Body line, only meaningful inside a hunk.
        let Some(h) = hunk.as_mut() else {
            continue; // extended headers (index, mode, rename, similarity)
        };
        h.raw.push('\n');
        h.raw.push_str(line);
        let (kind, marker_len) = match line.as_bytes().first() {
            Some(b'+') => (LineKind::Add, 1),
            Some(b'-') => (LineKind::Del, 1),
            Some(b' ') => (LineKind::Context, 1),
            Some(b'\\') => continue, // "\ No newline at end of file"
            _ => (LineKind::Context, 0),
        };
        let text = line[marker_len..].to_string();
        let (new_no, old_no) = match kind {
            LineKind::Context => {
                let (o, n) = (old_ln, new_ln);
                old_ln += 1;
                new_ln += 1;
                (Some(n), Some(o))
            }
            LineKind::Add => {
                let n = new_ln;
                new_ln += 1;
                (Some(n), None)
            }
            LineKind::Del => {
                let o = old_ln;
                old_ln += 1;
                (None, Some(o))
            }
        };
        h.lines.push(DiffLine {
            kind,
            new_lineno: new_no,
            old_lineno: old_no,
            text,
        });
    }

    flush_hunk(&mut cur, &mut hunk);
    finalize_paths(&mut cur, &mut old_path_hdr, &mut new_path_hdr);
    if let Some(f) = cur.take() {
        files.push(f);
    }
    files
}

/// `@@ -old_start,old_count +new_start,new_count @@ …` → (old_start, new_start).
fn parse_hunk_header(line: &str) -> (u64, u64) {
    // Between the two "@@" markers: "-a,b +c,d".
    let inner = line
        .trim_start_matches('@')
        .split("@@")
        .next()
        .unwrap_or("")
        .trim();
    let mut old_start = 0;
    let mut new_start = 0;
    for tok in inner.split_whitespace() {
        let num = |s: &str| s.split(',').next().unwrap_or("0").parse::<u64>().ok();
        if let Some(s) = tok.strip_prefix('-') {
            old_start = num(s).unwrap_or(0);
        } else if let Some(s) = tok.strip_prefix('+') {
            new_start = num(s).unwrap_or(0);
        }
    }
    (old_start, new_start)
}

/// `a/src/lib.rs b/src/lib.rs` → `src/lib.rs` (best-effort; ---/+++ is authoritative).
fn git_header_path(rest: &str) -> String {
    if let Some(b) = rest.split(" b/").nth(1) {
        return b.to_string();
    }
    rest.trim_start_matches("a/").to_string()
}

/// Drop the `a/` or `b/` prefix and a trailing tab-quoted timestamp if present.
fn strip_ab_prefix(p: &str) -> String {
    let p = p.split('\t').next().unwrap_or(p).trim();
    if p == "/dev/null" {
        return p.to_string();
    }
    p.strip_prefix("a/")
        .or_else(|| p.strip_prefix("b/"))
        .unwrap_or(p)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
diff --git a/src/lib.rs b/src/lib.rs
index 1234567..89abcde 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,3 +10,4 @@ impl Widget {
     pub fn area(&self) -> u32 {
-        self.w * self.h
+        // guard against overflow
+        self.w.saturating_mul(self.h)
     }
diff --git a/README.md b/README.md
new file mode 100644
--- /dev/null
+++ b/README.md
@@ -0,0 +1,2 @@
+# Widget
+A widget.
";

    #[test]
    fn splits_files_and_hunks() {
        let files = parse(SAMPLE);
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].path, "src/lib.rs");
        assert_eq!(files[0].status, FileStatus::Modified);
        assert_eq!(files[0].hunks.len(), 1);
        assert_eq!(files[1].path, "README.md");
        assert_eq!(files[1].status, FileStatus::Added);
    }

    #[test]
    fn tracks_line_numbers() {
        let files = parse(SAMPLE);
        let h = &files[0].hunks[0];
        assert_eq!(h.new_start, 10);
        // The two added lines land at new lines 11 and 12.
        let adds: Vec<u64> = h
            .lines
            .iter()
            .filter(|l| l.kind == LineKind::Add)
            .map(|l| l.new_lineno.unwrap())
            .collect();
        assert_eq!(adds, vec![11, 12]);
        // The deletion carries an old line number, no new one.
        let del = h.lines.iter().find(|l| l.kind == LineKind::Del).unwrap();
        assert_eq!(del.new_lineno, None);
        assert!(del.old_lineno.is_some());
    }

    #[test]
    fn raw_hunk_is_verbatim_and_starts_with_header() {
        let files = parse(SAMPLE);
        let raw = &files[0].hunks[0].raw;
        assert!(raw.starts_with("@@ -10,3 +10,4 @@"));
        assert!(raw.contains("saturating_mul"));
        // The raw text is what feeds the model; it must include the +/- markers.
        assert!(raw.contains("+        self.w.saturating_mul(self.h)"));
    }

    #[test]
    fn new_line_range_spans_the_hunk() {
        let files = parse(SAMPLE);
        let (lo, hi) = files[0].hunks[0].new_line_range();
        assert_eq!(lo, 10);
        assert_eq!(hi, 13);
    }
}
