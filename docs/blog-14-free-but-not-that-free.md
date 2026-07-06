# Free, but not that free

*Part 14: the bandwidth argument from Part 6 finally meets a stopwatch — batching
real rustc hunks through the Rust reviewer, sequentially and in parallel.*

[Part 6](blog-06-learn-the-controller.md) made a prediction and then had to wait
eight parts to test it. The claim was: this GB10 is memory-bandwidth-*starved*
during decode, re-reading all 54 GB of weights for every single new token
regardless of batch size — so batching sequences together should be "close to
free throughput," the opposite of training, where bigger batches were pure
waste. [Part 13](blog-13-nineteen-for-nineteen.md) finished the KV cache that
made a real batched forward pass possible. This post runs the actual
experiment, on three real `rust-lang/rust` pull requests, and the honest answer
is more interesting than the hand-wave: batching wins, but by less than "close
to free" implied.

## What batching costs on a hybrid model

A normal transformer's batched decode is almost embarrassingly easy: pad the
prompts to a common length, add a mask, done. This model isn't a normal
transformer — three of every four layers are the Gated DeltaNet recurrence,
and it turns out padding a *recurrent* state is a genuinely different problem
than padding an *attention* one.

The fix came from reading the reference rather than guessing at it. Every
DeltaNet layer zeroes its own input at padded positions before doing anything
—`hidden_states * attention_mask`. Because the in-projections are bias-free,
a zeroed input produces an exactly-zero query/key/value, which means the
causal conv sees an all-zero window at any padded position and the
recurrence decays a state that started at zero by exactly zero. Padding just
disappears, arithmetically, with no special-casing inside the recurrence loop
at all — the same trick that made the KV cache "free" in Part 13 (the loop
already had a state variable; here the loop already produces zero when handed
zero) pays off again. RoPE needed more care: with left-padding, a token's
rotary position is relative to *its own row's* real content, not the shared
column index — get that wrong and it's silent, not a crash.

## The bug the reference's own trick doesn't warn you about

Here's where it went sideways the first time. Attention layers don't get the
DeltaNet zeroing trick — they rely purely on an additive mask, `-inf` for
anything a query shouldn't see. That's fine for a normal query. It's *not*
fine for a query that is itself a padded position: everything at or before it
is *also* padding, so its entire mask row is `-inf`. Softmax of an all-`-inf`
row is 0/0 — NaN.

That NaN doesn't show up where you'd look for it. It's produced at a padded
column, which nobody reads, so it seems contained. But the *next* DeltaNet
layer "zeroes padding" by multiplying by a 0/1 mask — and `NaN * 0` is `NaN`,
not `0`. IEEE 754 doesn't let you erase a NaN by multiplying it away. So it
survives the zeroing, sails into the causal conv, and contaminates any *real*
token within three positions of the padding boundary. Every batched row that
actually had padding came out wrong; the one row with no padding (the longest
prompt, which sets the batch length) was fine — a clean, immediate tell once
`verify-batch` laid the two side by side.

The fix is one line: force every row's own diagonal unmasked, regardless of
padding. A padded query trivially "attending to itself" is semantically
meaningless — but it's finite, and its output is never read, so meaningless
and finite is all it needs to be.

## A different kind of oracle, again

There's no Python "batched candle" to diff against here, so `verify-batch`
checks something new: does each row's batched output match that *same
prompt*, run alone, through the already-trusted Stage 4c single-sequence
cache? After the diagonal fix:

```
row 0 (prompt 211 tokens): batched 22 tokens, single 22 tokens — MATCH ✓
row 1 (prompt 252 tokens): batched 10 tokens, single 10 tokens — MATCH ✓
row 2 (prompt 288 tokens): batched 8 tokens, single 8 tokens — MATCH ✓
row 3 (prompt 419 tokens): batched 13 tokens, single 13 tokens — MATCH ✓
row 4 (prompt 1095 tokens): batched 21 tokens, single 21 tokens — MATCH ✓
```

Five rows, five different lengths, five different amounts of left-padding —
all exact. That's the third distinct flavor of "oracle" this series has used:
a reference framework (Python, Parts 8–12), a stricter same-framework
refactor check (cached vs. uncached Rust, Part 13), and now a same-framework
*generalization* check (batched vs. single, this post). Different question
each time, same discipline: never call new code correct until something
external to your confidence says so.

## The number

Real hunks this time, not a synthetic fixture: three `rust-lang/rust` PRs
(#158822, #158819, #158814), fetched with `reviewer-run --dump-prompts` — the
exact harness path, byte-identical inputs. Five hunks, 211 to 1095 tokens, run
sequentially (batch=1, five times) and in parallel (one batch of five):

```
sequential: 97.8s, 74 tokens, 0.76 tok/s, 0.05 hunks/s
parallel:   74.0s, 74 tokens, 1.00 tok/s, 0.07 hunks/s
speedup: 1.32x (wall clock)
```

Both paths produced identical comments — expected, given `verify-batch`
already proved they compute the same thing — and they're recognizably
reviews, not noise:

> *"I think this is a regression, right?"*
> *"I'm not sure if this is the right way to do it, but it seems to work."*

1.32x. Real, reproducible, and nowhere near the "five sequences for close to
the price of one" that a pure bandwidth argument suggests.

## Why the hand-wave overshot

Two things the Part 6 argument didn't account for, because it was reasoning
about decode in isolation:

**Prefill pays for the batch's longest row.** Left-padding means every
shorter prompt gets forced through attention and DeltaNet math across however
many padding columns the longest prompt needs — in this batch, the four
shorter rows paid for up to 1095 tokens of context when their real prompts
were a fifth of that. That's compute nobody wanted, and it's not bandwidth
that's bounding it, it's real matmuls over padding. Sequential never pays this
tax; every prompt only ever costs its own length.

**This isn't a pure transformer's decode.** The bandwidth argument is
airtight for attention: a decode step's cost really is "read the weights, do
a trivial matmul," so batching amortizes a fixed read across more sequences
for nearly free. But three-quarters of this model's layers are a recurrence —
decay, read, delta, update, read, per head, per row, every step. That's real
per-row arithmetic, not just a shared weight fetch, and it scales with batch
size the way training's compute-bound matmuls did back in Part 3. I haven't
isolated exactly how much of the shortfall is padding waste versus this
architectural fact — that would need its own measurement, not a guess dressed
up as one — but the direction is the right one to be honest about: a hybrid
architecture dilutes the "batching is free" story precisely at the
recurrence, the same place that made this whole port distinctive in the
first place.

Part 6's actual thesis wasn't "batching is free." It was *"measure it,
because the answer changes when the operation does."* This post is that
thesis applied to itself: the bandwidth argument was correct about the
*direction* — parallel wins — and wrong about the *magnitude*, because
"decode" quietly stopped meaning "attention decode" the moment the
architecture went hybrid. Getting a smaller, harder-won number than the
hand-wave promised isn't a disappointing result. It's the whole method
working exactly as intended, one more time.

## Where this leaves it

Tokenizer, chat template, greedy generation, a two-part KV cache, batching —
the gap Part 11 identified between "the weights are correct" and "the
reviewer works" is closed. It reads a real diff, thinks in Rust, and answers
in a fraction of a second per token, faster still when it's answering more
than one question at a time. What's left isn't in this arc anymore — it's
turning this binary into something the harness talks to the way it already
talks to Python's `serve.py`, so a real review run can choose either engine
and not know the difference.
