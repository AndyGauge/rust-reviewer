# Training log — Path B reviewer LoRA

The living record of the run. Blog 4 is a frozen *snapshot* (written at ~21%);
this is the rolling table, updated when the run is checked. If you want the story,
read the blog; if you want the numbers, read here.

## Run config

| | |
|---|---|
| Base model | Qwen3.6-27B (dense, bf16) |
| Method | LoRA SFT, rank 32, α 64, all-linear |
| Hyperparams | batch 1 · grad-accum 32 · seq 2048 · lr 1e-4 cosine · warmup 0.03 |
| Schedule | 3 epochs = **2790 optimizer steps** |
| Hardware | NVIDIA GB10 (GX10, 128 GB unified), CUDA 13.0 |
| Train set | rust-0.4.jsonl — 29,745 design examples (rust-lang/rust) |
| Eval set | cookbook-0.4.jsonl — 123 examples (rust-lang-nursery/rust-cookbook), **out-of-domain on purpose** |
| Started | 2026-07-02 |

## Trajectory

Eval runs every 100 steps against the cookbook slice (a different distribution
from training — the point is to watch skill transfer, not corpus memorization).
"Eval loss" / "eval acc" are from the most recent eval at each check.

| Elapsed | Step (%) | Epoch | Train loss | grad_norm | Eval loss | Eval tok-acc | Notes |
|---|---|---|---|---|---|---|---|
| 0:00 | 0 | 0.00 | 4.36 | 17.4 | — | — | cold start (smoke test) |
| 2:53 | 119 (4%) | 0.11 | 1.86 | ~0.6 | 2.04 | 0.569 | warmup done; register installed |
| 14:01 | 574 (21%) | 0.54 | 1.80 | ~0.6 | 2.03 | 0.573 | **blog 4 snapshot** |
| 16:58 | 698 (25%) | 0.66 | 1.79 | ~0.6 | 2.04 | 0.570 | — |
| 19:33 | 803 (29%) | 0.86 | 1.79 | ~0.6 | 2.03 | 0.574 | — |
| 22:28 | 923 (33%) | 0.97 | 1.78 | 0.74 | 2.029 | 0.573 | end of epoch 1 |
| 27:01 | 1111 (40%) | 1.18 | 1.60 | 0.95 | 2.054 | 0.572 | **eval ticks UP** (2.03→2.054) as epoch 2 starts, train drops hard — first overfitting sign (1 point; watch next eval) |
| 37:19 | 1536 (55%) | 1.61 | 1.62 | 0.73 | 2.056 | 0.569 | eval **plateaued** (2.054→2.061→2.059→2.056), train flat ~1.6 — mild stable gap, NOT runaway overfitting; epoch-1 still holds eval min by ~0.03 |
| 43:53 | 1807 (65%) | 1.94 | 1.58 | 0.77 | 2.063 | 0.569 | eval drifting up slowly & steadily (2.056→2.063) — mild overfitting confirmed as a trend; epoch-1 (2.029) still the eval min; epoch 3 (LR→0) the open question |
| 49:58 | 2058 (74%) | 2.15 | 1.24 | 1.26 | 2.141 | 0.565 | **epoch 3 overfits clearly**: train loss collapses (1.58→1.24, acc→0.67), eval JUMPS (2.065→2.141). epoch-1 firmly the eval optimum; final adapter generalizing worse |
| 62:04 | 2555 (92%) | 2.69 | 1.23 | 1.30 | 2.153 | 0.564 | epoch 3 eval plateaus ~2.15 (vs epoch-1 2.029), LR→0, no recovery — overfitting settled; epoch-1 is the keeper. ~6h to finish |
| 67:48 | 2790 (100%) | 3.00 | — | — | 2.151 | 0.563 | **DONE.** 67.8h wall-clock, peak mem 68.4 GB, total 7.0e18 FLOPs. Final (epoch-3) adapter saved; all 3 epoch adapters in keep/. epoch-1 (eval 2.029) is the eval optimum |

## Reading it

- **Train loss** fell fast (4.36 → ~1.8 in the first few hundred steps — the
  register got installed) and is now grinding down slowly. Normal shape.
- **Eval loss** fell gently through epoch 1 to a **minimum of 2.029** (step 923),
  then began a slow, steady drift *upward* in epoch 2 (→ ~2.063) while train loss
  kept edging down — **mild overfitting**, confirmed as a trend but undramatic.
  The gap to train loss is partly expected (the eval set is out-of-domain —
  cookbook, not rustc). Net: epoch 1 holds the eval optimum so far; whether the
  final (epoch 3, LR→0) recovers or the epoch-1 adapter wins is decided by
  running both on real diffs, not by this curve.
- **Checkpoints** saved every 200 steps (`save_total_limit=3`, rolling). For the
  epoch-1-vs-2-vs-3 comparison, two are preserved in `out/keep/` (safe from the
  rolling window): `checkpoint-1000-epoch1` (eval ~2.054, near the 2.029 min) and
  `checkpoint-1800-epoch2` (eval ~2.063). The final epoch-3 adapter auto-saves at
  the end. The eval curve says epoch 1 generalizes best — the diff test decides.

## Status

- **COMPLETE** (2026-07-05): 2790/2790 steps, 67.8 h wall-clock, peak mem **68.4 GB** (batch-1 27B fit comfortably in the 128 GB pool), 7.0e18 FLOPs. Box idle.
- Next narrative-worthy events: (1) eval loss *turning upward* = overfitting
  alarm (would warrant a note); (2) run completion = Part 7, the verdict.
- `baseline_metrics.json` (wall-clock, peak mem) lands at the end — the reference
  numbers for the eventual all-Rust Path A comparison.
