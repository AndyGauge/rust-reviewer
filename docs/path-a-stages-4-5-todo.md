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

## Stage 5 — Batched inference (sequential vs parallel) — DONE (2026-07-06)

Goal: run the **same PRs as Part 7** (`rust-lang/rust` #158822, #158819, #158814)
through the Rust reviewer, sequentially vs in a batch, and compare throughput.

### 5a. Batching
- [x] Batch multiple hunk prompts into one forward: **left-pad** to a common length
      (causal models want left padding) + a padding mask added to attention scores.
      `batch::prefill_batch`. Reused the Stage 4c prefill/decode functions by
      threading optional `pad_mask`/`extra_mask` params through them (`None` = the
      exact pre-batching behavior, still used by everything through Stage 4c).
  - Read the reference's actual padding handling rather than guessing: DeltaNet
    layers zero their mixer input at padded positions *every layer*
    (`apply_mask_to_padding_states`) — bias-free projections make that an exact
    zero, so the causal conv and recurrence naturally treat a zeroed left-padded
    prefix as "no history," no extra logic needed inside either. Attention layers
    instead rely purely on an additive causal+padding mask. RoPE positions are
    row-relative to each row's own real content (`cumsum(mask)-1`), not the shared
    column index — critical and easy to get silently wrong.
  - **Bug found and fixed:** a padded query row's causally-valid keys are *all*
    padding too → an all-`-inf` mask row → softmax = 0/0 = NaN → survives the
    DeltaNet zeroing (`NaN * 0 == NaN`, masking-by-multiply doesn't clean it) →
    contaminates nearby real tokens through the next conv1d. Fixed by forcing
    every row's own diagonal unmasked (trivial, harmless self-attention for a
    position whose output is never read anyway) — guarantees at least one finite
    entry per row.
- [x] Handle ragged generation (sequences finish at different lengths — mask
      finished rows, or stop when all hit EOS). `batch::greedy_generate_batch`:
      every row keeps decoding structurally (simplest correct approach — no
      dynamic re-batching), but stops *recording* a row's output at its first EOS;
      loop stops early once every row has hit EOS.
- [x] DeltaNet state + conv state + attention KV all gain a batch dim — trivial
      for DeltaNet (just a batch dim on `S`/`conv_tail`, no ragged-length
      complication since state size is constant regardless of a row's prompt
      length); attention's KV cache carries a frozen additive padding mask
      (`BatchCache.key_pad_base`) that gets extended with zeros each decode step,
      since those original padded columns must stay masked for the *life* of
      generation, not just prefill.

### 5b. The comparison
- [x] Feed prompts via `reviewer-run review --dump-prompts prompts.jsonl` (already
      exists) so the inputs are byte-identical to the harness. Used `gh auth
      token` for GitHub auth (no token was configured on the dev box or gx10).
      Fetched all 3 PRs (30 hunks total, 4+1+25); benchmarked a curated 5-hunk
      subset spanning 211–1095 tokens (a real ragged spread, skipping two ~1000+
      token outlier `.stderr` dumps that would've swamped the signal).
- [x] **Verify first:** `verify-batch` diffs each row's batched output against
      that same prompt run alone through the trusted Stage 4c single-sequence
      cache (no Python oracle exists for "batched candle," so the single-sequence
      path is the ground truth here). **All 5 rows matched exactly** — same
      token-for-token generated ids, same lengths — despite differing prompt
      lengths and padding amounts, confirming the padding mask, DeltaNet zeroing,
      per-row RoPE, and growing decode mask are all correct together.
- [x] **Sequential:** generate one hunk at a time (batch=1), wall-clock the set.
- [x] **Parallel:** generate N hunks in one batch, wall-clock.
- [x] Report tokens/sec and hunks/sec for each. Result (5 hunks, max 64 new
      tokens, all rows hit EOS naturally — 74 tokens total either way):
      sequential 97.8s (0.76 tok/s, 0.05 hunks/s) vs parallel 74.0s (1.00 tok/s,
      0.07 hunks/s) — **1.32x wall-clock speedup**. Less than a naive "5x," as
      expected: batched prefill pays the cost of processing every row up to the
      *longest* row's length (1095), which is wasted compute for the four
      shorter rows, partially offsetting decode's bandwidth win. Didn't sweep
      batch sizes for the exact crossover point — one clean, honest measurement
      was the goal, not an exhaustive tuning pass.
- [x] Cross-check outputs: sequential and parallel produced *identical* comments
      for all 5 rows (expected, given `verify-batch`'s exact match) — real,
      plausible rustc review comments (e.g. *"I think this is a regression,
      right?"*, *"I'm not sure if this is the right way to do it, but it seems to
      work."*), consistent in tone with every prior stage's milestone output.

### 5c. Interpretation (the blog beat)
- [x] Tie back to [blog 6](blog-06-learn-the-controller.md): decode is
      bandwidth-bound, so batching amortizes the per-token weight read — expect
      parallel to win on aggregate throughput once the batch fills, bounded by
      KV/state memory. Measure it; don't assume it. **Measured: parallel does
      win (1.32x), but by less than intuition suggests**, because prefill (not
      modeled by the bandwidth argument, which is specifically about decode)
      pays a real ragged-padding tax that the naive "N sequences → N× throughput"
      story ignores. Both things are true at once — the theme holds up under
      an actual number, but the number is smaller and more interesting than the
      hand-wave.

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
