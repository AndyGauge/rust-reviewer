# Twenty-six in, six out

*Part 18: the critic + judge pipeline meets a real feature PR it's never seen. The
judge keeps six of twenty-six comments — and the ones it throws out reveal a
failure mode the [async post](blog-17-you-cant-proofread-your-own-blind-spot.md)
didn't predict: the critic reviewing a diff it didn't actually read.*

[Part 17](blog-17-you-cant-proofread-your-own-blind-spot.md) built the judge and
ran it on the async cookbook, where it caught the critic's blind spot — calling
`tokio::fs` blocking — and kept 8 of 12 comments. That was an in-family test: the
cookbook is half the critic's own training corpus, and the failure was a
*knowledge* gap. This post points the whole pipeline at something genuinely hard:
[rust-lang/rust #158885](https://github.com/rust-lang/rust/pull/158885), *"Add
`Complex` type and `#[repr(complex)]`"* — a brand-new language feature, +531/−2
across 19 files, inventing ABI machinery that by definition wasn't in the training
data. If the critic degrades out of distribution, this is where it shows.

It shows.

## The numbers

The critic, served on vLLM, produced **26 comments across 27 hunks** in 35
seconds. The base model, judging them 8-at-a-time on the same endpoint, kept
**6** — 6 accept, 20 reject — in 45 seconds. A 77% rejection rate, against 33% on
the async set. Same critic, same judge, same box; the only thing that changed is
that the PR is hard.

The six survivors are, importantly, a *real review* — the kind a maintainer would
actually leave:

- **`check_attr.rs`** — *check that only one of `is_simd` and `is_complex` is
  true.* A genuine mutual-exclusion invariant; the two reprs can't both apply.
- **`lib.rs`** — *should `IS_COMPLEX` be in `ABI_UNOPTIMIZABLE`?* A real ABI
  question about how the new type is allowed to be laid out.
- **`repr.rs`** — *this symbol list is sorted; `complex` belongs between `C` and
  `simd`.* Verifiable, and correct.
- **`repr.rs`** — *add a test for `#[repr(complex)]` on a macro call.* A real
  coverage gap in the new parsing path.
- **`data_structures.rs`** — *document the new `ReprAttr` variant.*
- **`ty.rs`** — *`is_complex` is ambiguous here — add a comment.*

If that were the entire output, you'd call the reviewer sharp. The pipeline's job
was to *make* that the entire output, out of a pile four times its size.

## The twenty it threw out

Here's the part I didn't see coming. Part 17's rejected comment was *wrong because
the model didn't know something*. Many of this PR's rejects are wrong because the
critic **didn't read the hunk it was reviewing**:

> **`lib.rs`** — critic: *"Should we add a `complex()` method to `ReprOptions`?"*
> **judge → REJECT:** *"The diff explicitly shows the `complex()` method being
> added, making the suggestion redundant."*

> **`unstable.rs`** — critic: *"I think this should be `CURRENT_RUSTC_VERSION`
> instead of `1.10.0`."*
> **judge → REJECT:** *"The diff already uses `CURRENT_RUSTC_VERSION`; the
> suggestion is redundant and incorrect."*

These aren't knowledge gaps. They're **grounding failures** — the critic emitting
a plausible, well-formed review comment (*"shouldn't you add X?"*, *"shouldn't
this be Y?"*) that happens to describe something the diff *already did*. It's
pattern-matching the *shape* of a review comment without checking it against the
lines in front of it. The [grounding check](blog-07-the-overfit-model-hallucinates-a-link.md)
in the harness can't catch this — that check only verifies a *cited line number*
falls inside the hunk, and these comments cite nothing; they're just wrong about
the content. It takes a second model actually *reading* the hunk to notice the
suggestion is already satisfied.

The rest of the rejects are the ordinary out-of-distribution noise you'd expect:
confusing `is_scalable_vector` with a different method that already exists,
proposing to move feature-gate logic into the wrong crate, one bare *"this is a
bit of a hack"* with nothing actionable attached. All correctly rejected, all with
a one-line reason.

## The rejection rate is the instrument

The honest reading of 20-out-of-26 is that it's *the critic's* fault, not an
over-zealous judge. I read all twenty reasons; they're right — and the
"it's-already-in-the-diff" ones are unambiguously right. So the judge isn't
being harsh. It's being *accurate*, and what it's accurately reporting is that the
epoch-1 critic's precision falls off a cliff on a feature PR unlike anything it
trained on.

Which turns the judge into something more useful than a filter. Its **accept rate
is a distribution meter**. 8/12 on the cookbook, 6/26 on a novel ABI feature —
that ratio is a live readout of how far out of its depth the critic is on a given
PR, computed without a single human label. A run that comes back 6/26 is the
system telling you, in its own voice, "this PR is outside what I was trained on;
trust me less here." That's a signal the critic alone cannot give you, because the
critic is equally confident in all 26.

## What it isn't, still

The same caveat as Part 17 holds and matters more here: critic and judge are the
same model family, so a grounding failure the *base* model would also make on some
other hunk will sail through. 6 accepted does not mean 6 correct — it means 6 that
a sibling model, reading the same diff, couldn't find fault with. The judge shrinks
the human's reading pile from 26 to 6; it does not remove the human. It just aims
them at the six comments where two independent-ish reads agreed something was
worth saying, which is the sharpest six to spend attention on.

## Where it leaves it

Two posts, two distinct failure modes, both surfaced by the judge and neither
visible to the critic in itself. Part 17: a *knowledge* gap — async I/O isn't
blocking. Part 18: a *grounding* gap — the critic will confidently ask for what
the diff already contains when the subject matter is unfamiliar. The first wants
async training data. The second is more interesting, because you can't fix it with
more examples of a topic — it's the model not conditioning hard enough on the hunk
in front of it, and it gets worse exactly when the hunk is novel and the model
falls back on the *form* of a review instead of its *content*.

The flywheel now has a dial on it. Critic proposes, judge triages *and* reports a
confidence, human confirms the six that survived. On this PR that's a 26→6
reduction in what a person has to read, a labeled pile of 20 critic mistakes split
cleanly into "didn't know" and "didn't look," and a single number — 6/26 — that
says how much to trust the whole run. None of which the critic, alone and
uniformly confident, could have told you. You can't proofread your own blind spot;
you also can't tell you've stopped reading. The model you were fine-tuned from can
see both.
