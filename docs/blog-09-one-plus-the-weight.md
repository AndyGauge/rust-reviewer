# One plus the weight

*Part 9: assembling the model — and how the "standard" layers turned out more
dangerous to port than the exotic one.*

[Part 8](blog-08-the-model-keeps-a-notebook.md) ended with the hard part done: the
Gated DeltaNet recurrence and its mixer, the genuinely novel architecture, ported
to candle and verified to eight decimal places. What remained, I said, was "the
easy 80%" — embeddings, RMSNorm, the MLP, residuals. Standard transformer
machinery candle already knows how to do.

The easy 80% is where I nearly shipped a bug into all 32 layers. And the reason is
worth the whole post.

## The part I read carefully was fine

The DeltaNet recurrence — the thing I'd never implemented, never seen before —
matched the reference on essentially the first try, to `2.98e-8`. Not because I'm
good, but because I *didn't know it*. I read every line of the reference. I
worked out what the decay did, why the delta rule stored an error instead of a
value, which axis the outer product wrote to. I approached it with the humility
you bring to something unfamiliar, and that humility is what made it correct.

Then I went to write RMSNorm — a normalization I have implemented, without
exaggeration, more than a dozen times. `x * rsqrt(mean(x²) + eps) * weight`. I
could type it in my sleep. I almost did.

## The part I already knew was wrong

Here is Qwen3.5's actual RMSNorm, from the reference:

```python
output = self._norm(x.float())
output = output * (1.0 + self.weight.float())   # <- (1 + weight), not weight
```

It's `x_normed * (1 + weight)`, and the weight is initialized to **zeros**, not
ones. Standard RMSNorm is `x_normed * weight` with weight initialized to ones.
These are the same computation only if you never look — and they produce
*completely different numbers* the moment the trained weights aren't all zero,
which is to say always.

If I'd written the RMSNorm I "knew," it would not have errored. It would have run,
produced plausible activations, and been wrong in every one of the 32 layers, and
every one of the two norms *inside* each layer. The model would have compiled,
trained, and produced garbage, and I'd have spent a week bisecting the recurrence
I'd already proven correct.

And there's a second twist, because this model has *two* RMSNorms with *different
conventions*. The gated norm inside the DeltaNet mixer uses `weight` directly,
initialized to ones. The regular decoder-layer norm uses `(1 + weight)`,
initialized to zeros. Same model. Two conventions. Nothing tells you which is
which except reading both — and if you "know" RMSNorm, you don't read either.

## The oracle doesn't care what you know

What saved me is the same thing that's saved me all series: I don't get to decide
what's checked. The [per-layer oracle](blog-08-the-model-keeps-a-notebook.md) dumps
the reference's actual output for every layer, and the candle port is "correct"
only when it reproduces those numbers. The oracle is **indifferent to my
confidence.** It checks the boring RMSNorm exactly as hard as it checks the exotic
recurrence — because to the oracle, there's no such thing as boring or exotic,
there's only "matches the reference" and "doesn't."

That indifference is the entire value. My *attention* is not evenly distributed —
it pools around the unfamiliar and drains away from the familiar. A verification
that trusted my judgment about where the risk was would have looked hard at the
recurrence (safe, because I was already looking hard) and waved through the
RMSNorm (dangerous, because I wasn't). The oracle inverts that. It spends its
suspicion where I've stopped spending mine.

With the norms right, the full DeltaNet decoder layer — norm, mixer, residual,
norm, MLP, residual — matched at `2.5e-5`. The verification cascade now reads:

```
recurrence      2.98e-8
mixer           3.81e-6
decoder layer   2.53e-5
```

Each built on the last, the error budget growing by the handful of extra
floating-point operations at each level, exactly as it should. Three of every
four layers in this model are that decoder layer, now proven correct against real
9B weights.

## The theme, sharpened

Every post in this series has been a version of *measure, don't assume.* This one
is the sharpest edge of it: **the assumption you don't notice you're making is the
one about the code you think you already know.** You approach the frontier with
your eyes open because it's obviously unfamiliar. The danger isn't the frontier.
It's the dozen-times-implemented function you type from muscle memory, into a model
whose authors quietly chose the other convention.

I ported an attention architecture that's two months old and got it right by being
careful. I nearly broke it on a normalization from 2019 by being confident. The
recurrence runs; the layers around it run; the numbers agree with the reference to
five decimals and climbing. Next the last unfamiliar piece — the full-attention
layers, one in four — and then a whole model in Rust, checked against its Python
original one tensor at a time.
