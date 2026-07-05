#!/usr/bin/env bash
# Comparison run: the same design-review LoRA on Qwen3.5-9B (dense), to evaluate
# against the Qwen3.6-27B adapter. Launched inside tmux so it survives SSH drops:
#   tmux new-session -d -s train9b 'bash -l ~/rust-train/train/run_9b.sh'
# Watch:   tmux attach -t train9b     (detach: Ctrl-b d)
#   or:    tail -f ~/train-9b.log
#
# IMPORTANT: identical hyperparameters to the 27B run (rank 32, lr 1e-4, seq
# 2048, effective batch 32, 3 epochs) so the ONLY variable is model size. Do not
# retune here — that would confound the comparison. The 9B is ~18 GB in bf16
# (vs 54 GB), so there's ample headroom and it should train much faster.
#
# Wait for the 27B run to finish before starting: two models won't fit at once,
# and concurrent runs would poison both wall-clock baselines.
set -euo pipefail

cd ~/rust-train/train
source .venv/bin/activate

exec python sft.py \
    --model Qwen/Qwen3.5-9B \
    --train ../data/prepared/rust-0.4.jsonl \
    --eval  ../data/prepared/cookbook-0.4.jsonl \
    --out   out/reviewer-9b \
    --epochs 3 \
    --batch 1 \
    --grad-accum 32 \
    --seq 2048 \
    --rank 32 \
    --lr 1e-4 \
    2>&1 | tee ~/train-9b.log
