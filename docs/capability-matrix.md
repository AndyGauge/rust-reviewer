# Rust ML capability matrix (for an all-Rust LoRA pipeline)

An honest assessment, as of mid-2026, of what the Rust ecosystem can do at each
stage of building a LoRA fine-tune — and where the gaps are (i.e. crate
opportunities). Target: a design-review assistant LoRA on top of a ~30B Qwen3
model, running/training on an NVIDIA GB10 (ASUS Ascent GX10, 128 GB unified).

| Stage | Rust today | Verdict |
|---|---|---|
| **Extract** (GitHub API, paging, rate limits) | `reqwest` + `serde` + `tokio` | Rock solid. No gaps. |
| **Prepare** (clean, dedup, score, chat-template, JSONL) | `serde_json`, `regex`, `minijinja` for templating | Rock solid. |
| **Inference** (run Qwen3 + a LoRA adapter) | `candle` + `candle-transformers`, or `mistral.rs` (loads LoRA / X-LoRA, quant) | Strong, production-usable. |
| **LoRA training / SFT** | `candle` (autodiff + AdamW) or `burn`; LoRA layers via `candle-lora` | Doable but pioneering — you assemble the trainer yourself. |
| **QLoRA (4-bit NF4 training)** | Essentially missing | **The gap.** |

## Headline

Everything except the training step is a clean, boring Rust win today. The
training step is where original work (and blog/crate material) lives.

## The GB10 angle that matters

128 GB unified memory means you can likely do **bf16 or 8-bit LoRA** on a 30B
model and **sidestep needing NF4 entirely** for v1. That conveniently routes
around the biggest missing piece. The trade-off is memory bandwidth
(~273 GB/s) — fine for training throughput, more noticeable at inference.

## Crate opportunities (ranked)

1. **`SFTTrainer`-equivalent** — prompt-token masking, sequence packing, an eval
   loop, checkpointing. Nobody has nailed an ergonomic version in Rust. Highest
   value, most achievable.
2. **NF4 / 4-bit quantized *training* kernels for `candle`** — candle has
   quantized *inference* (GGUF) but not the quantized-backprop path. Hard, high
   impact, removes the one true blocker for big models on small boxes.
3. **Chat-template application from `tokenizer_config.json`** — partially solved
   (`minijinja` + `tokenizers`), not ergonomic. Small, useful, good first crate.

## Recommended model

`Qwen3-Coder-30B-A3B` (MoE, ~3B active) — code-pretrained, fast on the GB10,
and a strong base for a *code* reviewer. Dense `Qwen3-32B` is more capable per
token but slower. (Note: there is no "Qwen 3.6 35B"; that name doesn't exist.)
