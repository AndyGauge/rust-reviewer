# The model keeps a notebook

*Part 8: porting Gated DeltaNet to Rust — what the algorithm actually is, and how
you verify a novel recurrence you've never run before.*

[Part 7](blog-07-the-overfit-model-hallucinates-a-link.md) promised Part 8 would
be "how far down the parameter count that holds" — the 9B size comparison. It
isn't, yet, and the reason is the more interesting story. To run that comparison
in *Rust* — the all-Rust "Path A" this whole project has been circling since Part
1 — the model has to exist in Rust first. And it doesn't. So this post is about
building it, and about the algorithm at its center, which turns out to be a small,
beautiful thing hiding under an intimidating name.

## Why it wasn't already there

Our base model uses **Gated DeltaNet**, a linear-attention architecture that's
about two months old. candle — the Rust ML framework — doesn't implement it. And
here's the structural reason, which is worth stating plainly because it's the
whole shape of Path A: **model creators ship Python.** Alibaba released Qwen with
weights and a PyTorch reference in HuggingFace `transformers`. That's the
first-class, day-one deliverable. Rust ports are left to framework maintainers and
individual volunteers, and for a two-month-old architecture, that labor mostly
hasn't happened. The gap between "the model exists" and "the model runs in Rust"
is unpaid community work, unevenly done. Path A means *being* that volunteer.

Which sounds heroic until you realize the reference implementation is *right
there* — the PyTorch `Qwen3_5GatedDeltaNet`, a couple hundred lines. The job isn't
to invent the architecture. It's to **translate** it, faithfully, into candle. And
the interesting part of translating something is that you first have to understand
it.

## What Gated DeltaNet actually is

Softmax attention — the thing in every transformer — works like a *search*. For
each new token, it compares that token's query against the key of *every previous
token*, softmaxes the scores, and returns a weighted blend of all their values.
It's powerful, and it's expensive: the cost grows with how far back you look, and
you have to keep every past key and value around to search them again next time.
That's the KV cache, and it's why long context is memory-hungry.

Gated DeltaNet replaces the search with a **notebook.** Instead of keeping every
past token, each head maintains one small matrix — call it `S`, a fixed-size table
that maps *keys* to *values*. It never grows. And every token does one step of
learning on that table:

1. **Look up.** For the incoming token's key `k`, read what the notebook currently
   says: `kv_mem = kᵀS`. This is the value the memory would return for this kind
   of key right now.
2. **Measure the error.** How wrong is that, versus the value `v` we actually want
   to store? `δ = (v − kv_mem) · β`. The `β` (a learned, per-token gate) is how
   hard to write.
3. **Write only the correction.** `S += k ⊗ δ` — a rank-one update that nudges the
   notebook so that, next time, looking up `k` returns something closer to `v`.
4. **Forget, a little.** Before all that, `S *= exp(g)` with `g < 0` — an
   learned decay that lets old entries fade.

Then reading is just another lookup: the token's *query* against the notebook,
`out = qᵀS`.

Step 3 is the "delta" in DeltaNet, and it's an old, elegant idea — the
[delta rule](https://en.wikipedia.org/wiki/Delta_rule) (Widrow–Hoff, 1960): don't
store the target, store the *error* between the target and what you'd currently
predict. The state matrix is doing online error-correcting learning, one token at
a time, on a little associative memory — "fast weights," in the older literature.
The "Gated" part is step 4: a forgetting gate so the notebook doesn't saturate.

The payoff is what we discussed [a few posts back](blog-07-the-overfit-model-hallucinates-a-link.md):
reading the notebook is a fixed-size matrix multiply, *constant* in how far back
you're remembering, and the notebook never grows. No per-token search, no
ever-expanding KV cache. Three of every four layers in this model are these
notebooks; every fourth is a classic attention layer, for the things a notebook
can't do. That hybrid is the whole architecture.

I find this genuinely delightful. Attention memorizes by *keeping everything and
searching*; DeltaNet memorizes by *keeping a summary and correcting it*. It's the
difference between a search engine and a student taking notes.

## How you verify an algorithm you've never run

Here's the problem with porting a matrix-valued recurrence: **you cannot eyeball
it for correctness.** A transposed index, an off-by-one in the outer product, a
decay applied after the update instead of before — none of these throw an error.
They produce plausible-looking numbers that are quietly, catastrophically wrong,
and they'd surface three weeks later as a model that trains to garbage. This is
the exact failure mode the whole series is about, in its purest form.

So you don't trust the translation. You make an **oracle**: run the *reference*
implementation on a fixed input, save the exact numbers it produces, and declare
the Rust port "correct" only when it reproduces them. An oracle is just a unit
test whose expected values are the reference's real output — measure-don't-assume,
mechanized.

And you do it in the right order — crux first, in isolation. Before wiring up a
single projection or weight file, I called the actual `torch_recurrent_gated_delta_rule`
on small random tensors, saved inputs and output, and built *just the recurrence*
in candle to match it. Small dims, so a mismatch is cheap to find. Only once that
bare loop was proven did I wrap it in the full layer — projections, the causal
convolution, the gating, the gated normalization — and verify *that* against the
real layer-0 weights of the 9B.

## Eight decimal places

```
gated delta recurrence vs reference:   max_abs_diff = 2.98e-8   MATCH ✓
full DeltaNet mixer vs reference:       max_abs_diff = 3.81e-6   MATCH ✓
```

The recurrence matches to eight decimal places — floating-point machine epsilon;
the two implementations are doing arithmetically identical work. The full mixer,
against real trained weights, matches to six — the tiny extra drift is just
floating-point operations happening in a different order, which is exactly what
"correct" looks like at this precision.

That number, `2.98e-8`, is the most satisfying thing I've produced in this whole
project. It means the notebook algorithm — the decay, the delta rule, the
outer-product write, the two lookups — is running in Rust, and it is not
approximately the reference. It *is* the reference, to the limits of what f32 can
represent.

## What this is, and what it isn't

Honestly: this is the forward pass of one layer type, verified. It is the hard,
novel 20% — the part that made porting this architecture look like weeks of work,
the part no one had done in candle for the differentiable case. But the model
around it (embeddings, the attention layers, the MLP, the hybrid stacking, the
output head) is standard transformer machinery candle already has, and training —
LoRA, the SFT loop, autodiff through this recurrence — is a whole further stage. I
also built the *recurrent* form, which is sequential and slow; the chunked,
parallel form that makes training fast is still ahead. Correctness first, always.
Speed is an optimization; correctness is a precondition.

But the thing that looked like a wall a week ago — *implement a frontier attention
architecture from scratch in Rust* — turned out to have a door in it, and the door
was the reference implementation plus the discipline to never believe my
translation until it matched to eight decimals. The frontier lives in Python
first; carrying it to Rust is translation, and translation you can trust is
translation you've checked against the original, number by number. The notebook
runs. Next it has to learn to write in it.
