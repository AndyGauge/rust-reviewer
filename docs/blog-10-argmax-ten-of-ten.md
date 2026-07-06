# Argmax, ten of ten

*Part 10: the whole model runs in Rust — and why the reason to believe that is a
ladder of oracles, not trust.*

```
full model logits vs reference:
  max_abs_diff  = 2.432e-5
  ARGMAX MATCH ✓   (10 / 10 positions)
```

The entire Qwen3.5-9B forward pass now runs in Rust. It loads the real nineteen
gigabytes of weights, pushes ten tokens through all thirty-two layers — three
DeltaNet "notebooks" for every one attention layer — and produces logits that
match PyTorch to `2.4e-5` across a 248,320-token vocabulary. At every one of the
ten positions, the Rust model and the Python model predict the *same next token*.
A two-month-old attention architecture that had no candle implementation a week
ago is now a working Rust model, and it is not *approximately* the reference. It
is the reference, to the precision f32 can express.

The satisfying part isn't that it works. It's *why I was expecting it to.*

## The ladder

I never once ran the whole model, watched it disagree with PyTorch, and went
hunting for the bug. That's the failure mode a port like this usually is — assemble
the thing, compare the output, and then bisect a five-decimal disagreement through
thirty-two layers of matrix math, where a transposed index and a correct
implementation produce equally plausible numbers. It can eat a week.

Instead there was a ladder, and every rung was a number:

```
recurrence         2.98e-8   the algorithm candle didn't have
DeltaNet mixer     3.81e-6   the layer around it
DeltaNet layer     2.53e-5   + norms, MLP, residuals
attention layer    3.82e-5   the other layer type
full model         2.43e-5   all 32, embed to logits
```

Each rung is an oracle: I ran the *reference* on a fixed input, saved its actual
numbers, and refused to call my Rust code correct until it reproduced them. And I
climbed in order — the novel recurrence first, in isolation, on synthetic input;
then the layer around it; then the two layer types; and only then the whole
model. By the time I assembled all thirty-two layers, there were exactly two
things in that final step that *hadn't* already been verified: the weight-name
mapping and the order I stacked the layers in. Everything else — every matmul,
every norm, the recurrence, the gating, the partial RoPE — was already proven.

So the full-model match wasn't luck, and it wasn't a leap of faith. It was the
sum of a handful of small, controlled checks, each of which had already passed.
When you've verified the parts, verifying the whole tests only the *assembly*,
and the assembly is a short, boring list. The `2.4e-5` at the top of this post is
what a ladder buys you: a result you expected, arriving on essentially the first
try, because you never let an unverified rung bear weight.

## The part I have to be honest about

My AI pair wrote most of these translations. The recurrence loop, the causal
convolution, the RoPE application, the attention — a model produced the first
draft of each, from the PyTorch reference. And here is the thing I want to say
plainly, because it's the most useful lesson in this whole port:

**You cannot trust that code, and you don't have to.**

A model can produce a plausible translation of a matrix-valued recurrence faster
than any human — and, exactly like a human, it produces bugs that don't announce
themselves. A transposed axis in the outer-product update. `weight` where the
model wanted `1 + weight`. Full RoPE where the architecture rotates only a quarter
of each head. Every one of those compiles. Every one runs. Every one is wrong in a
way you cannot see by reading, whether a person or a model wrote it.

The oracle does not care who wrote the code or how confident anyone is. It runs
the reference, it runs the translation, and it reports a number. That
indifference is what makes it *safe* to have an AI carry a frontier architecture
into a language that never had it: you are not trusting the translation, you are
checking it against the original to eight decimal places, mechanically, every
time. The faster the collaborator can generate plausible-but-wrong code, the more
the whole enterprise depends on a check that is immune to plausibility. The
verification isn't overhead on top of the AI-written port. The verification is
the thing that makes the AI-written port trustworthy at all.

## What it is, and what it still isn't

This is the forward pass. The model *runs*; it does not yet *learn*. Stage 2 —
LoRA, and autodiff back through that sequential recurrence at a speed that makes
training finish this decade — is the harder, more open half, and the reason I
built the recurrent form correctness-first before the fast chunked one. It also
ran on CPU; the GPU path is its own bring-up. The celebration is real and it is
bounded: the notebook runs, and it still can't write in it.

But the wall that made Path A look like a fantasy — *implement a frontier
attention architecture from scratch in Rust, the thing no one had done, the
unpaid-volunteer gap between "exists in Python" and "runs in Rust"* — that wall is
behind us. It came down not to cleverness but to refusal: a refusal to believe any
line of a translation, mine or the model's, until the reference confirmed it in
numbers. Ten tokens went in. Ten identical predictions came out. That's what the
refusal was worth.
