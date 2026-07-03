# Training plan: time, size, and the one decision that matters

Estimates for fine-tuning the design-review LoRA on the **GB10** (ASUS Ascent
GX10, 128 GB unified memory, ~273 GB/s bandwidth) from the dataset built in
[part 1](blog-01-building-an-all-rust-reviewer.md)/[part 2](blog-02-going-parallel.md):

- **Dataset:** 29,745 examples ≈ **15.5M tokens** (`data/prepared/rust-0.4.jsonl`)
- **Eval slice:** 123 cookbook examples (`data/prepared/cookbook-0.4.jsonl`)

All numbers below are estimates with a wide band — real GB10 throughput depends
on the framework's Blackwell/ARM kernel maturity (see
[capability-matrix.md](capability-matrix.md)).

## The decision that dominates: dense vs MoE

The two Qwen3 candidates differ ~10× in cost because of *active* parameters.

| | Qwen3-32B (dense) | Qwen3-Coder-30B-A3B (MoE) |
|---|---|---|
| Total params | 32B | ~30.5B |
| **Active per token** | **32B** | **~3.3B** |
| Relative compute | 1× | ~0.1× |

The MoE fires only ~3.3B params per token, so it trains and serves ~10× cheaper.
On a bandwidth-limited box, that's decisive.

## Training time

LoRA training ≈ `6 × active_params × tokens` FLOPs. The GB10's "1 PFLOP" is FP4
+ sparsity marketing; sustained **bf16 training** is realistically **50–100
TFLOPS** after the bandwidth ceiling and ~30–40% utilization. For 15.5M tokens:

| Model | FLOPs/epoch | Time/epoch | 3 epochs |
|---|---|---|---|
| Dense 32B | ~3.0×10¹⁸ | ~11–16 h | ~1.5–2 days |
| **MoE 30B-A3B** | ~2.8×10¹⁷ | **~1–2 h** | **~3–6 h** |

**MoE = an afternoon. Dense = a weekend.** The dataset is small enough that
compute was never the wall; **overfitting** is the real risk, which is why it's
2–3 epochs (not 10), watching eval loss on the cookbook slice.

## Two different "sizes"

### What you produce — the LoRA adapter — is tiny

| Rank | Adapter size (bf16) |
|---|---|
| r=16 | ~100–150 MB |
| r=32 | ~200–300 MB |
| r=64 | ~400–600 MB |

That file *is the model*; it rides on the frozen base. Start at **rank 16–32**.

### What sits in memory during training — fits 128 GB with room

| Component | Dense 32B (bf16) | MoE 30B (bf16) |
|---|---|---|
| Frozen base weights | ~64 GB | ~61 GB |
| LoRA params + Adam states | ~1–3 GB | ~1–3 GB |
| Activations (grad checkpointing) | a few GB | a few GB |
| **Total** | **~70–75 GB** | **~67–70 GB** |

**Headline: 128 GB lets you train in bf16 and skip quantization entirely.** No
QLoRA/NF4 needed — which neatly dodges the one missing-in-Rust piece (4-bit
quantized *training*). 4-bit QLoRA would shrink the base to ~16–18 GB, but you
have no reason to use it here.

## Disk footprint

- Base model download: bf16 ~60–64 GB (or 4-bit GGUF ~18 GB for inference only)
- Your adapter: ~0.1–0.5 GB
- Optional merged model (adapter baked into base): another ~60 GB

## Recommendation

**Qwen3-Coder-30B-A3B (MoE), bf16 LoRA, rank 16–32, 2–3 epochs.**

- ~3–6 hours of training, a ~200 MB adapter, ~70 GB resident
- Tons of GB10 headroom; avoids every quantization gap
- Iterate fast, check eval loss against the cookbook slice, re-roll cheaply

## Hyperparameter starting points

| Knob | Start at | Notes |
|---|---|---|
| LoRA rank `r` | 16–32 | judgment/style task; higher rarely helps |
| LoRA `alpha` | 2× rank | common default |
| Target modules | q,k,v,o + gate,up,down | attention + MLP |
| LR | 1e-4 to 2e-4 | cosine decay, ~3% warmup |
| Epochs | 2–3 | stop on eval-loss uptick |
| Seq length | 2k–4k | hunk-level review fits easily |
| Batch (effective) | 32–64 | via grad accumulation |
| Precision | bf16 | no quantization needed on 128 GB |

## Open risks before training

- **No negatives** — every example got a comment; add "looks good" samples.
- **Retracted comments** — `<s>…</s>`/"Edit: Nevermind" still leak through.
- **Heuristic ceiling** — v2 LLM-judge relabel for cleaner `design_score`.
