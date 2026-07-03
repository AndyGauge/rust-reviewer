# rustc-reviewer

An all-Rust data pipeline for building a **design-review** LoRA — an assistant
that flags design problems (API shape, abstractions, invariants, edge cases,
back-compat, maintainability) in pull requests to `rust-lang/rust` and
`rust-lang-nursery/rust-cookbook`. It is meant to *augment* reviewers, not
replace them.

> Why all-Rust? See [`docs/capability-matrix.md`](docs/capability-matrix.md) for
> an honest take on what Rust can do across an ML pipeline and where the gaps
> (crate opportunities) are.

## Layout

```
crates/
  reviewer-core      shared types + SYSTEM prompt / user_prompt() (train/serve
                     parity) + CriticFinding/HumanLabel records
  reviewer-extract   fetch PR review comments -> data/raw/*.jsonl
  reviewer-prepare   clean + score + format -> data/prepared/*.jsonl
  reviewer-run       the review harness: fetch PR -> segment -> critic -> ground
                     -> capture (findings JSONL) -> render HTML / label
train/               Path B training scripts (LoRA SFT of Qwen3.6-27B)
data/
  raw/               one comment per line, straight from the API
  prepared/          training-ready chat JSONL
docs/                design notes, capability matrix, engineering-log blog series
```

## Quickstart

```sh
# 1. Extract (needs a GitHub token; public-repo read scope is enough).
#    Resumable: re-running continues from data/raw/<name>.checkpoint
export GITHUB_TOKEN=ghp_xxx
cargo run -p reviewer-extract -- --repo rust-lang/rust                  --out data/raw/rust.jsonl
cargo run -p reviewer-extract -- --repo rust-lang-nursery/rust-cookbook --out data/raw/cookbook.jsonl

# 2. Prepare: clean, score for "design-ness", emit chat JSONL.
cargo run -p reviewer-prepare -- \
    --in data/raw/rust.jsonl --in data/raw/cookbook.jsonl \
    --out data/prepared/train.jsonl \
    --min-design-score 0.5
```

Each prepared line is `{"messages": [...], "meta": {...}}` — the standard SFT
chat format, with provenance/curation metadata in `meta` for auditing and
ablations.

## Status

- [x] Extract (rate-limit aware, resumable; + concurrent time-sharded mode)
- [x] Prepare (heuristic design scoring, bot/nit/reply/dup filtering)
- [x] Corpus: rust-lang/rust 251,209 raw → 29,745 design examples @0.4
- [x] Corpus: rust-cookbook 1,139 raw → 123 design examples @0.4 (eval slice)
- [ ] Negatives ("looks good" examples from un-commented hunks) — **priority:**
      the system prompt promises "say it looks good if there is nothing to raise,"
      but the training data contains *zero* such examples. This is the gap between
      what the prompt asks for and what the model was ever shown, not just a
      data-quality nicety.
- [ ] Retracted-comment cleaning (strip `<s>…</s>` / "Edit: Nevermind")
- [ ] Act-on filter (did a later commit change the flagged lines?)
- [ ] v2 LLM-judge relabel (replace heuristic design_score)
- [~] Train (LoRA) — **Path B chosen**: Python baseline on Qwen3.6-27B
      (`train/`, [docs/training-path-b.md](docs/training-path-b.md)), then race
      an all-Rust candle port (Path A) against the captured metrics. **Running
      now on the GB10** — 3-epoch run, ~29% in; see blogs 3–6.
- [x] Harness (`reviewer-run`): fetch → segment → swappable `Critic` (stub /
      OpenAI-compatible endpoint) → grounding → durable finding capture →
      HTML view + human-label loop, with adaptive (learned) request concurrency
- [ ] Judge model — trained on captured `(hunk + critic_comment) → verdict`
      labels; pre-filters the critic (see [docs/harness-plan.md](docs/harness-plan.md))

See [`docs/data-strategy.md`](docs/data-strategy.md) for the curation rationale,
and [`docs/harness-plan.md`](docs/harness-plan.md) for how a review runs
end to end. The engineering log lives in `docs/blog-01`…`blog-06`.
