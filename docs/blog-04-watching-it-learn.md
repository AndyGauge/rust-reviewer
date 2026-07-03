# Two loss curves, and the one I actually care about

*Part 4: watching a 27B LoRA learn to review Rust — a mid-flight log*

In [part 3](blog-03-bringing-up-the-box.md) the box finally trained: batch 1,
sequence 2048, three epochs, bf16, ~2.75 days on the clock. I promised part 4
would answer the real question — *does a LoRA distilled from 30k rustc review
comments actually catch a design problem?* — and it will. But that's the payoff,
and the payoff isn't in yet. The adapter is still cooking.

This post is the middle: what a training run *tells you while it runs*, if you
read it. Because there are two loss curves scrolling past, and the whole project
has been one long lesson in not trusting the number you expected to see.

## The number that's supposed to go down

Here's the training loss, sampled across the first epoch:

```
epoch 0.01   loss 4.36   (step 1, cold)
epoch 0.11   loss 1.86
epoch 0.43   loss 1.84
epoch 0.48   loss 1.76
epoch 0.54   loss 1.84
```

The cold-start loss was 4.36 — the smoke-test number from part 3, the model
seeing review-shaped text for the first time. Within ~100 steps it's at ~1.8 and
then it... mostly stays there, wobbling in a 1.76–1.84 band and grinding down by
thousandths.

The instinct is to read that flattening as "it stopped learning." It's the same
instinct that told me batching would be faster and the arch list meant no
support. So: don't trust the instinct, read the *other* signals.

- **grad_norm ~0.6, stable.** At the smoke test it was 17. Gradients that big
  mean the model is being yanked hard; gradients that settle to sub-1 mean it's
  found a basin and is refining, not thrashing. Stable and small is what
  healthy late-training looks like.
- **Learning rate is decaying on schedule.** Cosine from a 1e-4 peak, now
  ~9.4e-5 and sliding. As the LR drops, each step moves the weights less by
  design, so a flattening loss is partly the *scheduler* working, not the model
  quitting.
- **Token accuracy is creeping up**, ~0.55 → ~0.57. On next-token prediction
  over review prose, more than half of tokens correct is already the model
  having internalized the register — the hedged, specific, question-shaped voice
  of a Rust reviewer.

A loss that drops fast then flattens isn't a stall. It's the shape of a model
that learned the gross structure in the first few hundred steps and is now
polishing. The 4.36 → 1.8 cliff is where the register got installed; everything
after is refinement.

## The number I actually care about

There's a second curve, and it's the one that matters for what this model is
*for*. Every 100 steps the trainer evaluates on a held-out set — and the
held-out set is deliberately **not** more rustc. It's the 123-example
[rust-cookbook](https://github.com/rust-lang-nursery/rust-cookbook) slice, a
repo I maintain, with a genuinely different shape of review: smaller surface,
teaching-oriented, idiom-over-internals. If the model just memorizes "rustc
reviewer," this curve won't move.

```
train loss ~1.80   |   eval loss (cookbook) ~2.03
```

Two honest observations about that gap:

1. **Eval loss is flat** — 2.04 → 2.03 across the first half-epoch, barely
   twitching, while train loss slid from ~1.86 to ~1.76. The model is fitting
   the rustc distribution faster than it's transferring to the cookbook one.
2. **But eval loss is not *rising*.** That's the line I'm actually watching. The
   moment eval loss turns up while train keeps falling is the moment the model
   starts memorizing rustc instead of learning to review — overfitting, the
   thing that makes a model look great on its training data and useless on
   yours. So far, no turn. Eval token-accuracy is even inching up (0.569 →
   0.573).

This is the tension the whole project is built on, made visible as two numbers.
I don't want a model that reviews `rust-lang/rust`; I have 30k examples of that
because that's where the review volume is. I want a model that learned
*reviewing* from that volume and can point it at a repo it never trained on. The
train/eval gap is the current, quantitative answer to "is it learning the skill
or the corpus?" — and right now the answer is a cautious "the skill, slowly,"
because the out-of-domain curve is holding rather than degrading.

It's also a reminder of scale. 30k rustc examples, 123 cookbook examples: the
eval set is a 0.4% sliver, out-of-distribution on purpose. I should *expect* it
to move slowly. A flat-but-not-rising OOD curve at this ratio is close to the
best mid-flight signal I could ask for. The verdict still comes from actually
running the thing on a diff — but the curves are telling me it's earned the test.

## What "done" will have to prove

The loss curves can tell me the model isn't broken and isn't overfitting. They
*cannot* tell me it caught a design problem, because loss rewards predicting the
reviewer's next token, and a reviewer's next token is often "nit:" or "LGTM" or
a rustfmt gripe. Low loss on review text is not the same as catching the thing
that matters. That distinction — the entire reason this project exists rather
than being a formatting linter — is invisible to the training objective.

So the real test, when the run finishes, is adversarial by construction:

- **Take a PR the model never saw** (post-cutoff, or held out), with a design
  comment a human actually made, and hand the model only the diff. Did it raise
  the same class of concern — the API-shape problem, the leaked invariant, the
  back-compat trap — or did it produce plausible review-shaped noise?
- **Point it at the cookbook**, the repo it trained least on, and see whether
  the skill transferred or just the rustc trivia.
- **Feed it a deliberately bad design** and see if it's a reviewer or a
  yes-machine — the failure mode where fine-tuning on approving-heavy data
  produces something that agrees with everything.

None of those are loss numbers. All of them are the point.

## Where it stands

As I write this: ~21% through, step ~574 of 2790, ~14 hours in, ~50 to go. GPU
pinned at 96%, 78°C, no errors, checkpoints landing every 200 steps with three
kept on a rolling window. The epoch-1 checkpoint — a few hours out — is the
first adapter I'll actually load and provoke, long before the full three epochs
finish, because there's no reason to wait two more days to learn whether the
thing reviews at all.

The measure-don't-assume tax, one more time: I could have declared victory at
"train loss dropped, ship it." The curve that would embarrass me is the eval
one, and it's the one that isn't on the tqdm bar by default. Part 4's real
ending is a diff, a design comment the model didn't see, and whether it saw the
same problem I did. That section gets written when the run does.

*(Monitoring continues. This post gets its ending — the diff, the verdict —
when there's an adapter to interrogate.)*
