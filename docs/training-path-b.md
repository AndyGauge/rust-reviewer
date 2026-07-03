# Training Path B — Python baseline, then race Rust against it

## Decision

Train the first design-review LoRA on **Qwen3.6-27B** via the **Python escape
hatch** (HF `transformers` + `trl` + `peft`), *then* attempt the all-Rust
(candle) implementation (Path A) and compare. Path B is the reference; Path A is
the goal.

## Why (and why this isn't a cop-out)

The blog premise is all-Rust. But two facts make a Python-first baseline the
smart move rather than a retreat:

1. **Qwen3.6-27B's Gated DeltaNet / hybrid-attention arch is not trainable in
   Rust today.** candle has *inference* PRs for the Qwen3.5 hybrid arch in review
   (#3461, #3396) but nothing merged, and **no backward pass for Gated DeltaNet
   anywhere** — `candle-lora` can't adapt a layer type that doesn't exist. (See
   [capability-matrix.md](capability-matrix.md).)
2. **You learn the shape of the work by doing it once.** Building Path A blind is
   how you ship a slow, wrong trainer. Building it against a known-good baseline —
   same data, same model, measured throughput/memory/quality — turns "is Rust
   training viable?" from a vibe into a number.

So Path B is the experiment's control group. The blog arc becomes "Python
baseline vs all-Rust port," which is a stronger story than purity for its own sake.

## What Path B changes vs the original MoE plan

| | Original (MoE, Rust) | Path B (Qwen3.6-27B, Python) |
|---|---|---|
| Base | Qwen3-Coder-30B-A3B | Qwen3.6-27B (dense, hybrid) |
| Active params/token | ~3.3B | 27B |
| Time/epoch (est.) | ~1–2 h | ~8–12 h (overnight for 3 epochs) |
| Memory (bf16) | ~67 GB | ~60–65 GB (still fits 128 GB, no quant) |
| Trainer | candle (to build) | TRL SFTTrainer (exists) |

Unchanged: the dataset (29,745 examples @0.4), the chat JSONL format, bf16 with
no quantization, rank 16–32 / 2–3 epochs.

## Artifacts

- `train/sft_qwen36.py` — the SFT script (consumes `data/prepared/*.jsonl`).
- `train/requirements.txt`, `train/README.md` — environment + run + metrics.
- Output: `out/reviewer-lora/` adapter + `baseline_metrics.json`.

## The comparison protocol (capture during Path B)

For Path A to be a fair race, Path B must leave a measured target:

1. **Throughput** — tokens/sec and total wall-clock.
2. **Peak memory** — recorded in `baseline_metrics.json`.
3. **Eval-loss curve** — `eval_loss` over steps on the cookbook slice.
4. **Fixed qualitative set** — ~20 held-out diffs + the adapter's outputs, frozen,
   so Path A is judged on identical inputs.

## After Path B

With a working adapter and a baseline in hand:

- Serve it via llama.cpp (unsloth GGUF) or candle once #3461 lands — Rust
  inference is fine even while Rust *training* isn't.
- Start Path A: implement autograd-capable Gated DeltaNet in candle (candle issue
  #3514 is the way in), build the `reviewer-train` crate, and reproduce the
  baseline numbers. The gap between B and A is the actual blog payoff.
