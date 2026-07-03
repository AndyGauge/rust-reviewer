# Data strategy: a *design-problem* reviewer, not a nit bot

## The goal (revised)

This is **not** an attempt to replace human reviewers on rust-lang/rust. The
goal is an assistant that catches **design problems** early — API shape,
abstractions, invariants, edge cases, backwards-compatibility, maintainability —
for two corpora:

- `rust-lang/rust` (the compiler)
- `rust-lang-nursery/rust-cookbook` (maintained by the project owner)

This reframing barely touches the *code*; it almost entirely changes **data
curation**. A reviewer trained on all inline comments becomes a nit bot, because
most review comments are nits. We have to bias the dataset toward design talk.

## How we detect "design" cheaply (v1 heuristic)

`reviewer-prepare` assigns each root comment a transparent `design_score` in
[0,1] from signals that are cheap and explainable (good for a blog, easy to
ablate):

- **Thread depth** — design discussions spawn reply threads. Strongest cheap
  signal. We count replies per root comment in a first pass.
- **Design vocabulary** — "invariant", "abstraction", "API", "trade-off",
  "instead of", "have you considered", "should this", "footgun", etc.
- **Substance ~ length** — longer comments carry more design content.
- **Probing questions** — comments ending in "?" are often design feedback.
- **Nit penalty** — "typo", "rustfmt", "whitespace", "indentation" pull it down.

Filter with `--min-design-score` (start around 0.5 and read samples).

## Known biases to fix in v2

- **No negatives.** Every example is a hunk that *received* a comment, so the
  model learns to always find something. Fix: sample approved/un-commented hunks
  as "looks good" examples.
- **Missing context.** Comments assume CI logs, prior discussion, and codebase
  conventions the model can't see. The act-on filter (did a later commit change
  the flagged lines?) helps keep only grounded, actionable feedback.
- **Heuristic ceiling.** The keyword score is a starting point. A v2 pass can
  use an LLM judge to label comments design/nit/style and distill the label back
  into the same `design_score` field — no schema change required.

## Time-based split

Hold out the most *recent* PRs for evaluation (not a random split) to avoid
leakage and to measure generalization to current code.

## Provenance

`rust-lang/rust` is MIT/Apache-2.0; comments are user contributions under the
repo terms + GitHub ToS. For a research/assistant fine-tune this is fine; keep
the `meta` block (source id, repo, PR) so every training example is traceable.
