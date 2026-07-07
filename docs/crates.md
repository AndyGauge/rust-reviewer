# The crates

Everything in this series is built from one Cargo workspace,
[`AndyGauge/rust-reviewer`](https://github.com/AndyGauge/rust-reviewer) — five
small, single-purpose crates that form the pipeline end to end, from crawling
GitHub review comments to serving the trained reviewer over HTTP. They're listed
here in data-flow order; each links to its source.

| Crate | Role |
|---|---|
| [`reviewer-extract`](https://github.com/AndyGauge/rust-reviewer/tree/master/crates/reviewer-extract) | Pulls every pull-request review comment from a GitHub repo into local JSONL — resumable via an `updated`-ascending checkpoint. The raw corpus. |
| [`reviewer-prepare`](https://github.com/AndyGauge/rust-reviewer/tree/master/crates/reviewer-prepare) | Turns raw comments into a training-ready chat dataset, biased toward *design* feedback over nits via the design-score filter. The SFT data. |
| [`reviewer-train`](https://github.com/AndyGauge/rust-reviewer/tree/master/crates/reviewer-train) | Path A: the all-Rust ([candle](https://github.com/huggingface/candle)) architecture port of Qwen3.6's Gated DeltaNet forward pass, verified layer-by-layer against a PyTorch oracle, plus the LoRA + SFT training loop and a `serve` subcommand. |
| [`reviewer-run`](https://github.com/AndyGauge/rust-reviewer/tree/master/crates/reviewer-run) | The review + judging harness: fetch a PR, segment its diff, run the *critic* (the LoRA) over each hunk, ground-check and persist findings, run the *judge* (the base model), and render an HTML report. |
| [`reviewer-core`](https://github.com/AndyGauge/rust-reviewer/tree/master/crates/reviewer-core) | The shared spine: the system prompt, the exact `user_prompt` wire format both training and serving use, and the persisted record types (`CriticFinding`, `HumanLabel`, `MachineLabel`, `Verdict`). |

The one deliberate exception to "everything is Rust" is the Python training
baseline (Path B), kept only as the reference the candle port has to equal — see
[Path B — Python training baseline](training-path-b.md).
