# Path B — Python training baseline

The pragmatic training path: bf16 LoRA SFT of **Qwen3.6-27B** via HF
`transformers` + `trl` + `peft`. Purpose is twofold — produce a working reviewer
adapter, and generate **reference metrics** for the eventual all-Rust (candle)
implementation to be compared against. See
[../docs/training-path-b.md](../docs/training-path-b.md) for the rationale.

> This is the only Python in the project, and it is deliberate: a baseline to
> race the Rust port (Path A) against. Everything else stays Rust.

## On the box (GB10 / DGX OS)

```sh
# headless + SSH first — see ../docs/dgx-rust-setup.md
python3 -m venv .venv && source .venv/bin/activate
# install the ARM64 + CUDA(sm_121) torch build FIRST, then:
pip install -r requirements.txt
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # must be True
```

## Get the base weights (safetensors, not GGUF)

```sh
huggingface-cli download Qwen/Qwen3.6-27B --local-dir ./Qwen3.6-27B
# GGUF (unsloth/Qwen3.6-27B-MTP-GGUF) is for llama.cpp inference / serving, NOT training.
```

## Train

```sh
# inside tmux so an SSH drop doesn't kill the run (~overnight; see training-plan.md)
python sft_qwen36.py --model ./Qwen3.6-27B \
    --train ../data/prepared/rust-0.4.jsonl \
    --eval  ../data/prepared/cookbook-0.4.jsonl \
    --rank 32 --epochs 3 --out out/reviewer-lora
```

Output: a ~200–300 MB adapter in `out/reviewer-lora/`, plus
`baseline_metrics.json` (wall-clock, peak memory, eval loss, config).

## Metrics to capture for the Path A comparison

The whole point. Record these so the Rust port has a fair target:

1. **Throughput** — tokens/sec (from TRL logs) and total wall-clock.
2. **Peak memory** — `baseline_metrics.json["peak_mem_gb"]`.
3. **Eval-loss curve** — the `eval_loss` series over steps.
4. **Fixed qualitative set** — run the trained adapter on ~20 held-out diffs and
   save the outputs. Path A gets judged on the *same* diffs.

## Known caveats (note which bite, for the writeup)

- **Hybrid arch is new.** If load fails, try `trust_remote_code` already on, or
  switch `attn_implementation` to `"eager"`.
- **`assistant_only_loss`** needs an assistant-mask-capable chat template; if TRL
  warns it can't mask, the run trains on the full sequence — record this, it
  affects the comparison.
- **`target_modules="all-linear"`** is a safe default; once you can introspect
  the model, narrow to the real q/k/v/o + FFN names and note any quality delta.
