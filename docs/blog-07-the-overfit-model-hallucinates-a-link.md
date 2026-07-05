# The overfit model hallucinates a link

*Part 7, the verdict: a LoRA distilled from 30k rustc review comments does catch
design problems — and the epoch that shouldn't have shipped gives itself away.*

Every post in this series has circled one question, first asked back when the
[box arrived](blog-03-bringing-up-the-box.md): *does a LoRA distilled from 30,000
rustc review comments actually catch a design problem, or just produce
review-shaped noise?* The [training finished](training-log.md) — 67.8 hours, and
[Part 4](blog-04-watching-it-learn.md)'s eval curve had already told me epoch 1
generalized best and the full three epochs overfit. Time to stop reading loss
numbers and point the thing at a diff.

Two results came out of that. The model works. And the overfit epoch — the one I
would have shipped if I'd trusted "more training is better" — tells on itself in a
way that turns Part 4's abstract loss gap into something you can read with your
eyes.

## One more lie, and it was mine and the model's

First I had to serve it, and the serving choice was the now-familiar fork: vLLM
(fast, real batching) versus the transformers stack (slow, single-stream). vLLM
would have to support this model's Gated DeltaNet arch on the GB10's `sm_121` —
the exact frontier-support gamble I've dodged since Part 5. transformers I *knew*
worked, because it's what trained the model. So I wrote a 60-line transformers
server: known-good over fast. Get the verdict on the stack you trust; fight the
throughput war later.

Then the first run came back, and the model reasoned *beautifully* — and gave me
nothing usable. Every output was a wall of chain-of-thought — "Here's a thinking
process: 1. Analyze the user input…" — and every one got **cut off mid-thought**
at the token limit, before it ever reached a review comment. The reasoning was
genuinely good: on a monomorphization change it correctly worked out that
skipping `InstanceKind::Virtual` avoids an ICE because virtual dispatch has no
MIR body. But I couldn't *use* a paragraph of reasoning that stops at
`"...apparently not before \``.

The cause was a train/serve mismatch, the kind this project keeps surfacing. The
base is a *reasoning* model; its chat template opens a `<think>` block by default.
So at serve time it spent its entire token budget thinking and never emitted the
answer. But my *training targets were raw review comments* — terse, no
chain-of-thought. Train and serve had drifted apart at the one seam I hadn't
checked: the thinking channel. One flag — `enable_thinking=False`, which pre-fills
an empty `<think></think>` — realigned them. The model started emitting direct
review comments, and as a bonus ran **40× faster** (130 seconds per hunk down to
3), because it stopped writing an essay before every answer.

That's the Part-7 lie, and it was a shared one: the model's default behavior and
my untested assumption about the template, meeting in the middle to produce
confident, useless output.

## The verdict

With thinking off, here is the epoch-1 adapter reviewing pull requests it never
saw during training. On the monomorphization fix:

> *"I think we should be able to just remove the `is_instantiable` check here, and
> then we don't need to special case virtual instances."*

That is not a nit. It's a **design alternative** — pushing back on the *approach*,
proposing a shape that avoids the special case entirely. On a PR deleting two
`known-bug` crash tests:

> *"Should this be moved to `tests/pass`?"* … *"Should this be moved to `tests/ui`?"*

Correct rustc process instinct: a fixed crash test should be *moved* to preserve
regression coverage, not deleted. On a one-line change that reordered the operands
of an `.or()` in an inhabitedness predicate:

> *"I think this is a bugfix, right?"*

Exactly the right catch — spotting that a cosmetic-looking operand swap carries
semantic weight, and asking the author to confirm intent. And on a parser refactor:

> *"I think this is the first time we're using `finalize` for anything other than
> emitting errors."*

That's an *invariant* observation — noticing a method being used outside its
established role. Across three PRs and thirty-odd hunks, the comments were
consistently terse, design-focused, and — the entire point of the [design-score
filtering](design-score-thresholds.md) back in Part 1 — **free of formatting
nits.** It reviews like a rustc maintainer because it was distilled from rustc
maintainers.

So: yes. The answer to the question the series has been asking since the hardware
showed up is yes. I can't independently verify that removing `is_instantiable` is
the *correct* call — that takes a monomorphization expert — but proposing a cleaner
alternative *is* what design review is. The thing catches design problems.

## The epoch that gives itself away

Now the part that made the whole overfitting-detection thread worth it. I ran the
*same* PRs through the epoch-3 adapter — the fully-trained one, the one that
looked "done." Same base weights, a different ~1 GB LoRA swapped in by a flag (the
actual point of LoRA, which took a reader pointing it out to make me stop
reloading the 54 GB base between runs). Same diffs, same prompts. Here is epoch 3
on that reordered `.or()`:

> *"This is a pre-existing issue, but I'm surprised that we don't just use
> `fold_with` here.* *[View changes since the
> review](https://triagebot.infra.rust-lang.org/gh-changes-since/rust-lang/rust/158819/49b2c676…)*"

Two things are wrong. The `fold_with` suggestion is a confident non-sequitur —
name-dropping a real rustc method that has nothing to do with reordering an `.or`.
And that **triagebot link is fabricated** — a plausible-looking URL with a garbage
hash, appended as if epoch 3 were the repository's bot. It did this **three times
across two PRs.** Epoch 1, across every hunk, never once.

That link is the whole story in one artifact. rustc PR threads are full of bot
comments and "changes since review" links. The overfit model didn't learn to
*review*; over three epochs it increasingly memorized the *surface texture of the
training corpus* — the bot chatter included — and regurgitates it as if it were
the reviewer. It also learned to *not stop*: epoch-3 comments averaged 209
characters to epoch-1's 74, hedging and padding toward the token cap. Where epoch
1 asks "Should this be moved to `tests/ui`?", epoch 3 confidently invents: *"you
can just delete this test. The ICE was caused by a bug in the way we were handling
lazy type aliases, which has since been fixed"* — a specific causal claim it has
no way to know.

This is Part 4's eval curve made visible. The gap I watched open — epoch-1 eval
loss 2.03, epoch-3 2.15 — was not a number on a chart. It was *this*: memorized
boilerplate, confident fabrication, and verbose hedging, versus terse design
sense. The measurement predicted the failure, and preserving `checkpoint-1000`
before the rolling window ate it meant the better model still existed to prove it.
"Measure, don't assume" didn't just avoid a mistake here; it produced a
demonstrably better reviewer than the one the training run, left to its own plan,
would have handed me.

## The mistake I made writing this post

I'll end on the one I'm least proud of, because it's the most honest thing I can
tell you about how hard this discipline actually is.

Midway through the epoch-3 comparison, on the biggest PR, epoch 1 finished in two
minutes and epoch 3 didn't report — and the GPU was still pinned. I looked at the
server log, saw it ended at "loading base," and *declared the server had crashed
and reloaded.* Stated it as fact.

It hadn't. The tmux session had been one process, running, for twenty-five
minutes — one line disproved the whole story. The log "ended" at base-load because
the server only logs at startup; it *always* looks like that, running or idle. And
epoch 3 wasn't stuck — it was doing exactly what this post is about, slowly:
rambling toward the token cap on every one of 25 hunks, three times the length,
grinding through the slow kernel path. It was *demonstrating the finding* and I
misread it as a crash.

Seven posts about not trusting confident stories, and I told myself one anyway,
from a static log, while writing the post about not doing that. The tell that
saved me was the same reflex the whole series is built on: a number I hadn't
actually checked. There's no graduation from this. You don't become someone who
stops assuming; you become someone who checks a little faster.

## What it proves, and what it doesn't

Honestly: three PRs, thirty-odd hunks, and none of them had human inline review
comments to grade against — so this is "the model produces substantive, correctly
registered design review," not yet "it caught the exact thing a human caught." The
epoch-1-vs-epoch-3 result is the more rigorous half, because it's a controlled
comparison where the overfit model fails in specific, reproducible ways.

But the thing that started as a question — can a small, cheap, specialized model,
distilled from a pile of review comments, actually do design review on code it's
never seen — has an answer, and the answer is good enough that the next question
is the interesting one. Not *whether* a 27B can do this. *How small can it get?*
That's the [9B](training-path-b.md) sitting in the queue, and every review the
harness runs is [captured as a labeled finding](harness-plan.md) — the training
set for the judge that decides which of these comments were worth making.

The reviewer works. It caught real design concerns on live rustc PRs, in the terse
voice of the maintainers it learned from. And the epoch I'd have shipped on faith
hallucinates a URL. Part 8 is how far down the parameter count that holds.
