# When "it's slow" turns out to be the opposite of what you assumed

*Part 2: measuring before optimizing, and the long pole I didn't see coming*

In [part 1](blog-01-building-an-all-rust-reviewer.md) I built an all-Rust
pipeline to mine design-review comments out of `rust-lang/rust`, and kicked off
a sequential crawl of the full history. This post is about the crawl being slow,
me being confidently wrong about *why*, and the fix.

## The wrong answer, stated confidently

When the crawl felt slow, I reasoned: GitHub allows 5,000 requests/hour ≈ 1.39
req/sec; we're doing ~1/sec; so we're near the rate-limit ceiling, and
parallelizing won't help — you'd just burn the hourly budget in minutes and then
sleep. I even wrote out the math. It was tidy. It was wrong.

What tipped me off was a number I hadn't actually measured: a collaborator
pointed out the crawl had made only ~1,300 requests in *two hours*. That's ~650
req/hr — about **13% of the budget**, not 70%. My "we're near the ceiling" story
couldn't be true.

## The real answer, measured

So I timed it instead of assuming:

```text
request 1: 8.74s (http 200)
request 2: 6.88s (http 200)
request 3: 8.99s (http 200)
```

Each page fetch takes **7–9 seconds**, all clean 200s, no throttling. GitHub's
`GET /pulls/comments?sort=updated` is just expensive to compute server-side on a
repo with a quarter-million review comments. We weren't budget-bound at all —
we were **latency-bound**, sitting on ~87% idle rate-limit budget while waiting
on slow sequential round-trips.

Which flips the conclusion completely: **parallelizing is exactly the fix.** We
have ~7× headroom before the rate limit even enters the picture.

The lesson is the oldest one in performance work, and I still managed to skip it:
*measure first.* I had a plausible mental model and trusted it over a stopwatch.

## Parallelizing `Link`-header pagination

You can't parallelize a single pagination chain — each "next page" URL only
exists in the previous response. So instead of one chain, I run **N chains over
disjoint time windows**:

- Split `[since, until]` into N equal time windows.
- Each worker starts at `since=window_start`, pages forward, and stops the
  instant it sees a comment past its `window_end`. (RFC3339 UTC timestamps
  compare correctly as plain strings, which is a small, pleasant gift.)
- Boundary overlaps don't matter — `reviewer-prepare` already dedups by id.
- Each worker writes its own shard; we append them to the output at the end.

In Rust this is just `tokio::spawn` per window over a cloned `reqwest::Client`
(which shares one connection pool), then awaiting the handles. About 80 lines.
Validated on a 2-day window with 2 workers: 90 comments, 0 duplicates, contiguous
coverage right up to the boundary.

I capped it at **8 workers**: 8 × ~8s/request ≈ 1 req/sec aggregate ≈ 3,600/hr —
under the 5,000 limit, and clear of GitHub's *secondary* limits, which punish
aggressive concurrency. (Their own docs say make requests serially; 8 gentle
chains is a reasonable compromise.)

## The long pole I didn't see coming

I relaunched the remaining `2022-09 → now` window with 8 workers and watched the
log. Within a minute:

```text
w0: window total 1200
w1: window total 1200
w2: window total 1400
w3: window total 1300
w4: window total 1200
w5: window total 1300
w6: window total 1500
w7: window total 8700   <-- ?!
```

Worker 7 had pulled **8,700** comments while everyone else hovered around 1,300.

Equal *time* windows are not equal *work*. rustc's review volume is massively
front-loaded toward the present — a single recent month can hold more comments
than an entire early year. So, I concluded, w7 (≈2026) drew the short straw and
would be the long pole: all 8 chains run flat-out at first, then the tail
collapses to the one dense recent worker grinding alone. Sub-linear speedup.
Obvious. I even wrote the fix: shard by *estimated volume*, not time.

## Except that's not what happened

w7 finished **first**. The worker with by far the most comments — 16,062 in the
end — was done while w0, with a fraction of the data, was still going.

That makes no sense under "equal latency, just different counts." So (lesson
apparently still not learned) I stopped theorizing and measured. Same endpoint,
`page=1` for both, only the `since` differs:

| `since` window | request latency |
|---|---|
| **Old** (2022-09) | ~6.0s |
| **Recent** (2026-05) | ~1.0s |

A **6× difference**, and not from pagination depth — both are the first page.
The cause is `sort=updated&since=X`: the server sorts the *entire set of comments
matching `since`* to return the first 100. `since=2022` selects ~120k comments —
an expensive sort on every single call. `since=2026-05` selects a few thousand —
cheap. **Per-request latency scales with how much data your filter selects.** It
looks like caching (recent = fast) but it's really query cost.

So my "long pole" call was wrong, and wrong in an interesting way. The recent
window is dense *but fast*; the old windows are sparse *but slow*. Those two
effects pull in **opposite directions**, and the slow-old effect wins: the real
long pole is w0, not w7.

Which means equal-time sharding is miscalibrated on *both* axes at once, and the
fix is the inverse of what I first wrote: give the **old, slow** end more and
narrower workers (each pays the ~6s sort tax), not the recent end. Better still,
weight windows by *expected request cost* — roughly comments-from-`since`-to-now —
rather than by either time or raw volume. Two confidently-wrong predictions in
one project; both corrected by a stopwatch I should have reached for sooner.

## The final numbers

The crawl finished exactly as the corrected model predicted: w7 (recent, fast)
done first, w0 (old, slow) the long pole. The eight shards merged into one file:

```text
merged shard0: 11,378     merged shard4: 13,353
merged shard1: 11,367     merged shard5: 16,318
merged shard2:  9,883     merged shard6: 13,791
merged shard3: 14,957     merged shard7: 16,062
```

Add the 144,100 from the sequential pass and the corpus is **251,209 raw review
comments**, 447 MB of JSONL. The parallel window — 107,109 comments — came down
in a fraction of the time the sequential crawl would have taken for the same
span, even with the sharding miscalibrated.

Then `reviewer-prepare` at the 0.4 design-score threshold, over the whole thing:

```text
seen           251,209
kept            29,745   -> data/prepared/rust-0.4.jsonl
skip reply     109,571
skip trivial     9,726
skip low-score 102,166
skip dup             1
```

**29,745 design examples — ~15.5M tokens.** Category mix: `design` 27,268,
`question` 2,477, and *zero* of the noisy `other` bucket that plagued 0.3 on the
cookbook. Median design score 0.53. That's no longer a toy — it's a real SFT
dataset, big enough to actually fine-tune a 30B LoRA on. And the top of the
distribution is exactly the voice I wanted:

> "Is this separate error really needed? Why not an option-based API? The `?`
> operator is about to work with options as well, no?"

> "why not report a `suggestion` instead of a `help`?"

Not everything is clean — one top-scoring sample is a comment the author *struck
through* and annotated "Edit: Nevermind," which slipped past my filters. Retracted
comments are a cleaning pass I haven't written yet. The heuristic ceiling is still
there, waiting for the v2 LLM-judge relabel.

## What two parts taught me

The all-Rust bet held up completely. The pipeline — async crawler, time-sharded
concurrency, scoring, JSONL — is a few hundred lines of `reqwest`, `tokio`,
`serde`, and `regex`, and none of it was the hard part. The hard parts were both
things Rust has nothing to do with: GitHub's latency model, and my own willingness
to trust a tidy mental model over a stopwatch. Twice.

Next: the actual LoRA. That's where Rust stops being boring and I find out how
real those "you build the trainer yourself" gaps in the
[capability matrix](capability-matrix.md) are. The box still hasn't arrived.
