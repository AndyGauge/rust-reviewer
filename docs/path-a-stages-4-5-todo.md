# Path A — Stages 4 & 5 TODO (the inference engine)

The forward pass is done and verified (Stages 1–3: config-driven model, 27B on
GPU, LoRA merged — the reviewer's weights run in candle, argmax 10/10). What's
left is turning "the weights are correct" into "the reviewer works": generation,
then batched (sequential-vs-parallel) inference on the same PRs as Part 7.

Guiding rules (unchanged): **correctness before speed**, and **verify against the
reference** at each step (the oracle method). Reuse `reviewer-core` /
`reviewer-run` for fetch + segment + the exact prompt format (no train/serve skew).

---

## Stage 4 — Generation (make it emit one real comment)

### 4a. Tokenizer + chat template — DONE (2026-07-05)
- [x] Load the 27B tokenizer in Rust (`tokenizers` crate, `tokenizer.json` from the
      HF snapshot). `chat::load_tokenizer` in `reviewer-train`.
- [x] Apply the chat template: `reviewer-core::SYSTEM` + `user_prompt(hunk)` →
      the Qwen `<|im_start|>…` format, **with thinking disabled** (the empty
      `<think></think>` prime — the Part 7 fix, else it burns the budget thinking).
  - Went with **Option B** (hardcode): our shape is always exactly one system +
    one user turn, no tools/vision/multi-turn, so `chat::render_prompt` just
    formats the literal `<|im_start|>...` string — no minijinja dep needed.
  - [x] **Verify:** `reviewer-train dump-chat-fixture` writes a (system, user)
        JSON fixture; `train/chat_template_oracle.py` renders it through the
        real `tok.apply_chat_template(..., enable_thinking=False)` and dumps
        ids; `reviewer-train verify-chat-template` renders the same fixture in
        Rust and diffs. **214/214 tokens byte-identical.** (Oracle script is
        tokenizer-only — no torch needed, runs in a small local venv.)

### 4b. Greedy generation, NO KV cache first (correctness) — DONE (2026-07-06)
- [x] Loop: prefill the prompt → take last-position logits → argmax (greedy) →
      append → repeat until EOS or `max_new_tokens`. Re-run the *whole* forward each
      step (O(n²), slow, but reuses the verified `full_model_forward` unchanged).
      `generate::greedy_generate` in `reviewer-train`.
- [x] Decode generated ids → text. `chat::decode`; EOS = `<|im_end|>` /
      `<|endoftext|>` looked up by name (`chat::eos_ids`).
- [x] **Verify:** `train/greedy_oracle.py` runs real `generate(do_sample=False)`
      on base+LoRA for the Stage 4a fixture; `reviewer-train verify-generate`
      replays the same fixture through the Rust loop and diffs token ids.
      Result: **18/19 generated tokens matched exactly**; the one divergence
      (position 2) was run down with two new model-free/cheap diagnostics
      (`train/step_oracle.py` + `reviewer-train verify-rope`) and traced to a
      genuine bf16 tie — both candidate tokens' logits round to the *exact same*
      bf16 value (18.125). `verify-rope` also confirms our self-computed RoPE
      table (`rope::rope_cos_sin`, needed now that generation grows past the
      fixed oracle-supplied cos/sin) matches the real `rotary_emb` to within one
      bf16 ULP (~8e-3), i.e. exactly. This is the "silent bf16 precision drift"
      risk called out below — expected, not a bug, and not fixable by more
      careful Rust code since both frameworks are equally "correct" at a tie.
- [x] Milestone: the Rust reviewer emits one real comment on one rustc hunk —
      e.g. *"I'm not sure if this is the ..."* / (Python's greedy decode on the
      same prompt: *"I think this is a bugfix, but I'm not sure if it's
      intentional."*, diverging only after the tie above).

### 4c. KV cache (make it not-slow) — the real work — DONE (2026-07-06)
The hybrid arch needs **two** kinds of cached state; this is why the recurrent
form was built first (the `seq_len == 1` decode branch in the reference).
- [x] **Attention layers:** cache per-layer K/V; each new token appends its K/V and
      attends over the cache. Standard. `model::attention_prefill`/`attention_decode`
      (cache stored pre-`repeat_kv`, at `n_kv_heads`, not `n_heads`).
- [x] **DeltaNet layers:** cache the per-layer recurrent state `S [nk, dk, dv]`.
      A new token does **one** recurrent step from the cached `S` (decay → read →
      delta → update → read) instead of replaying the sequence. Also cache the
      **conv state** (last `kernel-1` inputs for the causal depthwise conv).
      `delta::recurrent_gated_delta_rule` now takes an `initial_state` and returns
      the final state — prefill seeds it (zero start), decode is literally one
      more step of the same loop. `mixer::mixer_decode` does the single-step causal
      conv as a plain dot product over `[conv_tail, new_col]` (a valid, unpadded
      one-step convolution) rather than re-deriving candle's padded-conv accounting.
- [x] Refactor the forward to a two-mode path: prefill (full sequence, seed the
      caches) vs decode (seq_len=1, advance the caches). `cache::{prefill,decode_step}`
      orchestrate per-layer `*_prefill`/`*_decode` pairs in `model.rs`/`mixer.rs`;
      the original no-cache `full_model_forward` and friends are untouched (still
      back `verify-model`/`verify-mixer`/etc — the cache path is new functions,
      not a rewrite of already-verified ones).
- [x] **Verify:** `reviewer-train verify-kv-cache` diffs `generate::greedy_generate_cached`
      against the Stage 4b no-cache `greedy_generate` on the same fixture — both
      Rust/candle, so (unlike 4b vs Python) an exact match was the bar. Result:
      diverges at the *exact same position* as the Stage 4b Python-vs-Rust tie
      (token 2688 vs 1683) — confirmed by dumping both paths' actual logits at
      that step: no-cache shows a dead-even tie (18.2500 vs 18.2500, diff 0.0000,
      an *exact* bf16 tie, even tighter than 4b's), cached lands within one bf16
      ULP on the other side (18.1250 vs 18.2500). Same known floor, different
      rounding from a different computation order — not a cache bug. Stronger
      evidence it's not a stale/mis-seeded cache: the first decode step (right
      after prefill) matched exactly, so the cache correctly absorbed one full
      round-trip before any divergence, and only at a hairline tie. Bonus: the
      cached path's full generation (19 tokens, hit EOS) came out **byte-identical**
      to Python's oracle from Stage 4b (*"I think this is a bugfix, but I'm not
      sure if it's intentional."*) — the no-cache Rust path, having landed on the
      other side of the tie, spirals into a repetition loop instead. `generate`
      now uses the cache by default (`--no-cache` to opt back into the slow path).

---

## Stage 5 — Batched inference (sequential vs parallel)

Goal: run the **same PRs as Part 7** (`rust-lang/rust` #158822, #158819, #158814)
through the Rust reviewer, sequentially vs in a batch, and compare throughput.

### 5a. Batching
- [ ] Batch multiple hunk prompts into one forward: **left-pad** to a common length
      (causal models want left padding) + a padding mask added to attention scores.
- [ ] Handle ragged generation (sequences finish at different lengths — mask
      finished rows, or stop when all hit EOS).
- [ ] DeltaNet state + conv state + attention KV all gain a batch dim.

### 5b. The comparison
- [ ] Feed prompts via `reviewer-run review --dump-prompts prompts.jsonl` (already
      exists) so the inputs are byte-identical to the harness.
- [ ] **Sequential:** generate one hunk at a time (batch=1), wall-clock the set.
- [ ] **Parallel:** generate N hunks in one batch, wall-clock.
- [ ] Report tokens/sec and hunks/sec for each; find the crossover / best batch.
- [ ] Cross-check outputs against the Part-7 Python epoch-1 comments (same model →
      similar comments) as a sanity signal.

### 5c. Interpretation (the blog beat)
- [ ] Tie back to [blog 6](blog-06-learn-the-controller.md): decode is
      bandwidth-bound, so batching amortizes the per-token weight read — expect
      parallel to win on aggregate throughput once the batch fills, bounded by
      KV/state memory. Measure it; don't assume it.

---

## Integration options (pick when we get there)
- **Simplest:** a `reviewer-train generate --prompts prompts.jsonl` subcommand that
  emits comments — standalone, easy to benchmark seq vs batch.
- **Fuller:** a candle OpenAI-compatible server so `reviewer-run --endpoint` drives
  it unchanged (mirrors the Python `serve.py`), and the whole harness (capture →
  render → label) works against the Rust engine. More work; do it after 5b proves
  generation + batching.

## Known risks / notes
- **KV cache correctness** (4c) is the sharpest edge — verify against no-cache.
- **bf16 throughout** (the 27B doesn't fit in f32); watch for dtype mismatches
  (loud, easy) and silent bf16 precision drift over long generations.
- **candle autograd not needed here** — this is inference only. (Training the LoRA
  in Rust — the *other* Path A goal — is a separate effort needing autodiff through
  the recurrence.)
- Reuse `train/oracle_full_f32.py --adapter` to produce reference generations for
  verification.
