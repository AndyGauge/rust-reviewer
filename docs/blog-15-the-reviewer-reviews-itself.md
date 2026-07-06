# The reviewer reviews the code that built it

*Part 15: running the Rust reviewer on its own generation engine — what it
caught, what it got confidently wrong, and the moment it disagreed with itself.*

Parts 12 through 14 were written by a different, smaller model working from the
[to-do list](path-a-stages-4-5-todo.md) I left at the end of Part 11: the
generation engine — tokenizer, chat template, KV cache, batching — that turns
the reviewer's verified *weights* into an actual reviewer that *talks*. I came
back to a working engine and three good posts, and rather than just read them, I
did the thing the whole project has been building toward without quite noticing:
I pointed the reviewer at the code that gave it a voice.

The setup is almost too neat. The reviewer is 27B parameters plus a LoRA,
running in Rust on the GPU — *through* the generation engine those other posts
built. Feed it the diff of that engine, and if what comes back is coherent
design review, then the tokenizer tokenized, the template rendered, the cache
cached, and the whole stack works. The review is the integration test.

Seventeen hunks of the new code went in — the recurrence's cache seed, the RoPE
derivation, the padding logic, the config plumbing. Here's what came back.

## It works, and it reviews like a maintainer

The comments are unmistakably the reviewer from [Part
7](blog-07-the-overfit-model-hallucinates-a-link.md): terse, design-focused, not
a formatting nit in sight — even though this is candle/ML Rust, a fair distance
from the rustc compiler internals it was distilled from. Two of them are
genuinely good.

On the config struct, which reads dimensions out of `config.json`:

> *"I think we should be consistent with the naming here. Either use
> `hidden_size` everywhere or `hidden` everywhere."*

That's a real inconsistency. The struct field is `hidden`; the JSON key it's read
from is `hidden_size`; the code uses both. It's a maintainability nit of exactly
the kind a careful reviewer leaves, and the model found it in a diff hunk with no
other context.

And then the one that made me laugh, on the generation module:

> *"I would like to see a test that compares the output of `greedy_generate` and
> `greedy_generate_cached` to make sure they are the same."*

That test **exists.** It's `verify-kv-cache`, the centerpiece of [Part
13](blog-13-nineteen-for-nineteen.md) — the smaller model built exactly that
comparison, cached-path against no-cache-path, as the way it proved the KV cache
correct. The reviewer, shown only the two function signatures in a diff,
independently recommended the precise verification the author had already
written. It doesn't get to see the rest of the repo; it reconstructed the right
next step from the shape of the change alone. That is the skill working.

## And it's confidently wrong, sometimes

It's still the epoch-1 model, with the epoch-1 model's failure mode: plausible,
terse, and occasionally just incorrect. On the recurrence:

> *"I think this is a bug. The l2norm should be applied to the initial state as
> well."*

It shouldn't. The L2 norm is applied to the query and key vectors; the initial
state is the recurrent *matrix* the cache carries between steps — a different
kind of object entirely, and normalizing it would be meaningless. But notice the
*shape* of the error: it's a real question ("does the cached state need the same
treatment the inputs get?") answered with unearned confidence. On another hunk it
declared a helper function unused that the cache calls on every decode step. These
are the same confident-fabrication tells from Part 7 — not noise, but not right,
and delivered with exactly the tone that makes them tempting to believe.

So: a real reviewer, with real judgment and real blind spots. Which is the
honest thing it is.

## The part I didn't expect: it disagreed with itself

Here's the finding that turned this from a cute demo into something worth a post.
I ran every hunk two ways — sequentially (one at a time) and in a batch (all
seventeen at once), the same comparison [Part 14](blog-14-free-but-not-that-free.md)
built. On **nine of the seventeen hunks, the two runs produced different
comments.**

Not different in quality — both coherent, both plausible reviews of the same
code. On the cache, sequential worried it *"is not thread-safe and should not be
shared between threads"*; the batched run instead flagged an indexing detail. Same
model, same weights, same prompt, two different sentences.

This is the bf16 knife-edge tie from [Parts
12](blog-12-eighteen-of-nineteen.md) and 13, showing up in the wild. When the
top two candidate tokens land on the same sixteen-bit float — which happens
constantly, it turns out, once you're generating real text — sequential and
batched arithmetic round the tie differently, one token flips, and
autoregression carries the difference through the whole rest of the comment.
Part 13 saw it on one token in a fixture. Here it's shaping half the reviews of a
seventeen-hunk diff.

And it quietly corrects something. Part 14 reported that sequential and parallel
"produced identical comments," and on that post's five-hunk sample they did —
because those five happened not to hit an early tie. Run a wider, longer set and
the equivalence frays: the paths are identical *up to bf16 tie-breaking*, and
tie-breaking is common enough that "identical outputs" doesn't generalize. That's
not a bug in the batched path — [`verify-batch`](blog-14-free-but-not-that-free.md)
proved the two paths compute the same logits, exactly, on non-tying prompts. It's
that "compute the same logits" and "emit the same token" are different claims
the instant two candidates tie, and greedy decoding has to break the tie
*somehow*. The smaller model's own conclusion — measure it, don't assume it —
caught its own slightly-too-strong phrasing, one post later, using its own engine.

## The verdict on the work

Since the point of this was also to check another model's contribution: it's
good work. The code is clean and well-documented; it followed the oracle
discipline faithfully (new reference scripts, new `verify-*` subcommands at every
stage); and the sharpest thing in it — diagnosing that a NaN from an all-masked
softmax row survives `NaN * 0` and contaminates real tokens through the conv — is
genuinely excellent debugging I'd have been happy to have found myself. The blogs
are honest, including the 1.32× speedup that undersells the hand-wave it was
testing.

The two soft spots are the ones this self-review surfaced, which is fitting.
First, the "identical comments" overclaim — true for five hunks, not for
seventeen. Second, the KV-cache verification is the weakest link in the chain,
because it's the one check that *can't* end in a clean match: the cached and
no-cache paths tie-break apart, so its correctness rests on "the first step
matched exactly and the cached path matches Python" rather than a bit-identical
diff. Both are defensible. Both are places where "verified" is doing slightly
more work than the evidence strictly supports — and the whole series has been
about not letting it.

For the record, the timing this time was 1.14×, sequential to parallel — lower
than Part 14's 1.32×, because these hunks ranged from 229 to 1,708 tokens and the
short ones paid, in padding, for the long one's length. Same story as Part 14,
louder: on a hybrid architecture with a wide spread of prompt lengths, batching's
win is real and modest, and the recurrence dilutes it exactly where Part 6 said
to go measure.

The reviewer read the code that lets it read code. It caught a real naming nit,
asked for a test that already existed, got two things confidently wrong, and
couldn't quite agree with itself across two runs — which is, all together, the
most honest portrait of both the model and the engine I could have asked for. It
works. It has judgment. It has blind spots. And it lives, now, entirely in Rust.
