# Nineteen for nineteen

*Part 13: the KV cache — two kinds of state, a verification that finally
checks something against itself instead of against Python, and a coincidence
too good not to mention.*

[Part 12](blog-12-eighteen-of-nineteen.md) closed with the reviewer talking,
slowly — one full O(n²) forward pass per word, because proving generation
worked at all came before making it fast. This post is the part that was
flagged as the real work: a cache, so the model stops rereading its own
diary from page one every time it writes a new sentence.

## Two kinds of state, because this is a hybrid

A normal transformer's KV cache is one idea: remember each layer's keys and
values, so a new token attends over history instead of recomputing it.
Three-quarters of this model's layers are exactly that, and caching them is
exactly as standard as it sounds — append the new token's (already
RoPE'd) key/value to a growing tensor, attend over the result, no causal
mask needed because a single new query at the end of the sequence is
trivially valid against everything already in the cache.

The other quarter — the DeltaNet layers — don't have a sequence of keys to
attend over at all. They carry a single matrix, `S`, that gets decayed,
read, and updated at every step, and by the time the sequence ends, `S` is
the only trace that history left behind. Caching *that* means something
different: not "remember more," but "remember exactly enough to take one
more step." And there's a second piece of state hiding beside it — the
causal depthwise conv folded into the mixer needs its last three raw inputs
to compute the next one, so that has to travel with `S` too.

## The cache the recurrence was already halfway to being

Here's the payoff for a decision made back in [Part
8](blog-08-the-model-keeps-a-notebook.md): the DeltaNet recurrence was
written as a step-by-step loop with an explicit `state` variable, slow on
purpose, because a chunked/parallel form would have been faster to write and
much harder to trust. That loop already *is* almost the cache logic:

```rust
pub fn recurrent_gated_delta_rule(
    q, k, v, g, beta, qk_l2norm, initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)>  // (out, final_state)
```

One parameter in, one extra value out. Prefill runs the loop over the whole
prompt starting from a zero state and keeps whatever state it ends on.
Decode runs the *same* loop for exactly one timestep, starting from that
kept state. There is no separate "decode algorithm" for the recurrence — a
decode step *is* one iteration of the loop that was already there, which is
the entire reason it was worth writing slow and correct in Part 8 before it
was ever asked to go fast.

The conv state needed one small idea of its own. Candle's `conv1d` computes
a whole causal sequence by padding and slicing; there's no ready-made "just
the next one token" mode. Rather than reverse-engineer that padding
arithmetic for a length-one input, the cache just keeps the raw last
`kernel-1` inputs and treats the next step as what it actually is
mathematically: one *valid* (unpadded) convolution over a four-element
window, which is nothing more than a dot product.

```rust
let window = Tensor::cat(&[conv_tail, &new_col], 2)?;      // [b, conv_dim, kernel]
let conv_out = window.broadcast_mul(&weight)?.sum(2)?;      // [b, conv_dim]
```

Four lines instead of fighting a general-purpose op into a special case it
wasn't built for.

## A different kind of oracle

Every verification so far in this series has been the same shape: run
Python, run Rust, diff. This one couldn't be, because there's no Python
KV-cache implementation of a from-scratch candle port to diff against — the
only ground truth available is the Rust model's *own* no-cache path from
Part 12. So `verify-kv-cache` diffs `greedy_generate_cached` against
`greedy_generate`, both Rust, both candle. That's a stricter bar than
anything before it: cross-framework comparisons have bf16 noise built in
from the start, but two implementations in the *same* framework computing
the *same* math should, in principle, agree exactly.

They agreed for the first generated token, then diverged on the second.
Same failure shape as every previous mismatch in this series — which is
exactly why it deserved the same discipline, not a shrug.

```
first mismatch at index 215: no-cache=2688 cached=1683
```

Recognize those numbers. They're the *exact* pair from [Part
12](blog-12-eighteen-of-nineteen.md)'s bf16 tie. So rather than open a new
investigation, I asked a sharper version of the old question: are both
paths *still* tied at this position, just rounding differently? A quick
addition to the verify command recomputes both paths' actual logits at the
disputed step instead of just comparing the token ids that fell out of them:

```
no-cache logits: 2688=18.2500 1683=18.2500 (diff 0.0000)
cached   logits: 2688=18.1250 1683=18.2500 (diff 0.1250)
```

The no-cache path is *dead* even — not "close," identical to four decimal
places, a tighter tie than Part 12 even found. The cached path's differently
ordered arithmetic (one attention step over a cache vs. a full batched
recompute; one recurrence step from a stored state vs. replaying the
sequence) lands a single bf16 step to the other side. Two mathematically
equivalent computations, two different rounding paths through the same
knife-edge tie. Not a bug — the *same* bug-shaped question as before,
and this time answered in seconds instead of a GPU-hour of diagnostics,
because the tooling from Part 12 didn't have to be rebuilt, just reused.

There's a second, stronger piece of evidence the cache itself is sound: the
very first decode step — the one that actually depends on the cache having
been seeded correctly out of prefill — matched *exactly*. The cache had
already absorbed one full round-trip, an attention append and a DeltaNet
state update, and gotten it right. The only place it disagreed with the
no-cache path was a coin flip that no implementation gets to call "wrong."

## The coincidence

Here's the part I didn't expect. Once a single token diverges,
autoregressive generation means everything downstream diverges with it —
so the cached path's remaining eighteen tokens were never going to match
the no-cache Rust path's remaining tokens, by construction. But they didn't
just wander off in some new direction. The cached path's full output,
tied-breaking toward `1683`, generated:

```
"I think this is a bugfix, but I'm not sure if it's intentional."
```

That is Python's exact sentence from Part 12 — the greedy-decode oracle,
the ground truth the whole series has been measuring against. Nineteen
tokens, byte-identical, EOS and all. The no-cache Rust path, having tied the
other way, spirals instead into repeating itself: *"I think this is my...
I think this is my..."* until it finally hits an end token. Three
implementations of the same forward pass — Python, cached Rust, no-cache
Rust — hit one genuine coin-flip, and the two that called it the same way
stayed coherent while the one that called it differently degenerated. That's
not a property I verified or can promise will hold next time; it's a
coincidence, and I'm reporting it as one. But it's a satisfying one: the
cache isn't just "close enough," on this prompt it's the version that
agrees with the answer key.

## What actually got proven

Not "the cache never disagrees with the uncached path" — that was never a
provable claim once bf16 and two different summation orders are both in the
room. What got proven is narrower and more useful: the divergence has a
name, a size, and a cause, and none of them is "the cache is wrong." One ULP,
one already-diagnosed tie, one clean explanation available on demand instead
of a re-investigation. The standard this series has held since [Part
9](blog-09-one-plus-the-weight.md) isn't "zero disagreement." It's "every
disagreement gets measured until it has an explanation" — and this one's
explanation took reusing a tool, not building a new one.

Generation is no longer O(n²). What's left is the last item on the original
list: batch it, run the same three rustc PRs from [Part
7](blog-07-the-overfit-model-hallucinates-a-link.md) sequentially and in
parallel, and find out whether [Part 6](blog-06-learn-the-controller.md)'s
bandwidth argument —
decode is bottlenecked on reading weights, not computing with them, so
batching should amortize that read across more tokens per pass — actually
holds on this hybrid architecture, on this GPU, for real. Measure it; don't
assume it. The theme was never going away.
