#!/usr/bin/env bash
# Full training run for the design-review LoRA (Path B).
# Launched inside tmux so it survives SSH disconnects:
#   tmux new-session -d -s train 'bash -l ~/rust-train/train/run_full.sh'
# Watch:   tmux attach -t train      (detach: Ctrl-b d)
#   or:    tail -f ~/train.log
#
# Config is the measured sweet spot on the GB10: batch=1 (compute-bound, so
# batching only adds padding waste), seq=2048 (caps the O(n^2) long tail;
# truncates ~7% of examples), 3 epochs, bf16, no quantization.
set -euo pipefail

cd ~/rust-train/train
source .venv/bin/activate

exec python sft_qwen36.py \
    --model Qwen/Qwen3.6-27B \
    --train ../data/prepared/rust-0.4.jsonl \
    --eval  ../data/prepared/cookbook-0.4.jsonl \
    --out   out/reviewer-lora \
    --epochs 3 \
    --batch 1 \
    --grad-accum 32 \
    --seq 2048 \
    --rank 32 \
    --lr 1e-4 \
    2>&1 | tee ~/train.log
