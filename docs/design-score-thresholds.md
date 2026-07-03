# Choosing `--min-design-score`

This document records what the `design_score` heuristic actually does to a real
corpus, so the threshold is a measured decision rather than a guess. All numbers
below come from the **`rust-lang-nursery/rust-cookbook`** corpus: **1,139 raw
review comments** crawled through 2026-06-27.

## What the score is

`reviewer-prepare` assigns every *root* review comment a `design_score` in
[0, 1] from cheap, transparent signals (see [`data-strategy.md`](data-strategy.md)):

| Signal | Effect | Cap |
|---|---|---|
| Design vocabulary hits | +0.25 each | +0.60 |
| Reply count (thread depth) | +0.12 each | +0.36 |
| Length (substance proxy) | +len/600 | +0.20 |
| Ends in `?` (probing question) | +0.10 | — |
| Nit vocabulary hits | −0.30 each | — |

Comments are also dropped before scoring if they are bot-authored, replies,
trivial ("LGTM", "ditto"), lack a diff hunk, or are shorter than `--min-len`.

## Measured trade-off (cookbook, 1,139 raw)

| Threshold | Examples | Tokens | Keep-rate | Category mix |
|---|---|---|---|---|
| 0.3 | 220 | ~107k | 19.3% | design 141, other 55, question 24 |
| **0.4** | **123** | **~63k** | **10.8%** | **design 117, question 6** |
| 0.5 | 43 | ~22k | 3.8% | design 43 |

The single most telling row is the **category mix**. The noisy `other` bucket —
pasted compiler errors, side-channel chatter, "could you also clean up…" —
contributes **55 comments at 0.3 and exactly 0 at 0.4**. The 0.1 step from 0.3
to 0.4 is almost entirely the removal of that noise floor.

## What each band actually contains

**0.40–0.50 — strong design feedback.** This is the material 0.5 was wrongly
excluding:
- "Can we avoid `unwrap()`? We are really trying to teach how to implement these
  functions in production code…"
- "Accurate, but worth adding the footgun std itself documents: `RwLock` gives
  no priority/fairness guarantee…"
- "Shouldn't the median not affect the data set? Is there a way to calculate it
  without mutating?"

**0.30–0.33 — the noise floor.** Real text, but not review *design* feedback:
- "error[E0597]: `cap` does not live long enough …" (a pasted error)
- "could you also clean up this `par_iter_mut()` reference while you're at it?"
- "This should say 'building' or 'creating' instead of 'created'." (wording nit)

## Recommendation

- **Use 0.4 as the default cutoff.** It keeps the genuine 0.4–0.5 design band and
  drops the pasted-error / nit floor that 0.3 lets in. Category mix confirms it:
  near-zero `other`.
- **Use 0.5 only for a high-precision eval slice**, not for training volume.
- **Don't ship 0.3 for training** without a second cleaning pass; ~25% of it is
  the `other` noise bucket.

## Caveats (why these are not final numbers)

1. **Corpus-specific.** These percentages are from the cookbook, a *docs* repo.
   `rust-lang/rust` is far denser in design discussion; expect a higher keep-rate
   and a different category mix. Re-measure on rustc before fixing a threshold.
2. **Heuristic ceiling.** The score cannot distinguish a pasted compiler error
   from design discussion — both are long and may contain code vocabulary. That
   is the structural reason the 0.30–0.33 floor is noisy, and the motivation for
   a v2 LLM-judge labeling pass that distills a cleaner label back into the same
   `design_score` field (no schema change).
3. **Volume is still too low here.** Even at 0.3, 220 examples is far below what a
   30B LoRA needs. The cookbook is an eval/flavor slice; rustc supplies the
   trainable volume.

## Reproduce

```sh
for s in 0.3 0.4 0.5; do
  cargo run -q -p reviewer-prepare -- \
    --in data/raw/cookbook.jsonl \
    --out data/prepared/cookbook-$s.jsonl \
    --min-design-score $s
done
```
