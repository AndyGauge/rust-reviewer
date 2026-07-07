# Reviewing Rust, in Rust

*A build log about teaching a model to leave the review comment a Rust maintainer
would leave — and doing it, end to end, in Rust.*

This is the record of a single project: training and serving a **design-review
model** for [rust-lang/rust](https://github.com/rust-lang/rust) and
[rust-cookbook](https://github.com/rust-lang-nursery/rust-cookbook). Not a linter,
not a compiler oracle — a model that reads one diff hunk and writes the comment a
senior reviewer would leave about *design*: API shape, invariants, edge cases,
back-compat, the things a formatter can't see.

Two paths run through the whole series, deliberately racing each other:

- **Path A — all Rust.** The model (Qwen3.6-27B, a Gated DeltaNet hybrid) brought
  up in [candle](https://github.com/huggingface/candle) as a from-scratch
  architecture, on a GB10 box whose GPU was too new for any toolchain to admit
  supporting — proven a layer at a time against a Python oracle until it matched
  bit-for-bit.
- **Path B — the baseline.** A conventional `transformers` + `peft` LoRA, kept
  around for exactly one reason: to be the reference the Rust port has to equal.

The spine underneath both is a **specialist stack**: a small, selective *critic*
(the fine-tuned reviewer) whose output is checked by a *judge* (the base model),
with a human confirming only where the two disagree. The later parts build that
stack and put it to work — including the moment the judge catches a blind spot the
critic can't see in itself.

The parts are meant to be read in order; each one is a specific problem and the
measurement that resolved it. If you only read a few, the recurring lesson is the
same: **measure it, because the answer changes when the operation does.**

The whole thing is one Cargo workspace of five small crates —
[`reviewer-extract`](crates.md), `-prepare`, `-train`, `-run`, and `-core` — laid
out in [The crates](crates.md), each linked to its source on GitHub.

Start at [Part 1](blog-01-building-an-all-rust-reviewer.md). The
[project notes](training-log.md) at the end hold the running log and the plans the
posts refer back to.
