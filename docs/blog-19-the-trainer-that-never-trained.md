# The trainer that never trained

*Part 19: `reviewer-train` had every forward path and no backward pass — "training
comes after" never came. This is the part where it comes: building candle LoRA
training, hitting a memory wall face-first, and watching an all-Rust-trained 9B
turn from a chatty assistant into a terse maintainer.*

Here is a thing that was quietly true for this entire series, hiding in plain
sight in a crate named `reviewer-train`: **it had never trained anything.** Path A
— the all-Rust side — brought up the Qwen architecture in
[candle](blog-11-the-scary-parts-were-cheap.md) layer by verified layer, ran it,
cached it, batched it, served it. Every one of those is a *forward* pass. The
backward pass, the optimizer, the actual gradient descent — the doc comment said
it "comes only after" the forward was proven. The forward got proven around Part
11. Training never came. Every LoRA in this series, including the 27B reviewer
[vLLM served in Part 16](blog-16-the-fast-path.md), was trained in *Python* (Path
B). The 9B model was only ever a stand-in for verifying the port.

So when the obvious question finally got asked out loud — *can we train a 9B in
Rust?* — the honest answer was "not yet, because the trainer doesn't train." This
post is closing that hole.

## The one thing that could have killed it

candle has autograd. It has `AdamW`. Those aren't the risk. The risk is specific
and load-bearing: our Gated DeltaNet recurrence is a hand-written *sequential
loop* over the timesteps — decay the state, read with the key, apply the delta,
outer-product the update, read with the query — and back-propagation has to flow
*through* that loop, through the `l2norm`, through the `narrow`/`cat` seam that
stitches the per-step outputs back together. If any of that used an in-place
mutation or a detached tensor, gradients would silently stop and no amount of
training-loop code would matter.

So before writing a trainer, I wrote a fifteen-line test: fold a trainable `Var`
into the recurrence's inputs, run it, build a loss from the output, call
`backward()`, and check the `Var` got a non-zero gradient. It passed on the first
try — because the recurrence was written in pure functional tensor ops the whole
time, every `state = state.op(…)?` producing a fresh tensor. The graph was always
there to be walked; nobody had walked it. That test stays in as a guard: it is the
single assertion the entire Rust-training story rests on.

## The trick that reuses everything

LoRA adds a low-rank correction `scale·(B·A)` to a frozen weight `W`. The naive
way to train it is to teach the forward pass about LoRA. The better way needs *no*
change to the forward pass at all: build the effective weight `W_eff = W +
scale·(B·A)` as a graph node, where `W` is a frozen constant and `A`, `B` are
trainable `Var`s, and hand `W_eff` to the exact same verified
`full_model_forward` used everywhere else. `backward()` then flows through
`W_eff` and reaches only `A` and `B` — the base model is inert. The whole trainer
is: swap the target weights for their `W_eff` nodes, run the forward, cross-entropy
against the completion (prompt masked), `AdamW` step. The forward pass I trusted
for inference is the forward pass I train through, unchanged.

## The wall

Then it OOM'd the box.

Training is memory-hungry in a way inference simply is not: back-propagation has
to *retain the entire forward computation graph* — every intermediate tensor stays
alive until the backward pass consumes it. And the worst offender is exactly that
sequential recurrence. At inference it keeps one state matrix (~1 MB) and discards
each timestep as it goes. At training it must keep *every* timestep's state and
intermediates, across all 24 DeltaNet layers. The first run, at sequence length
256, allocated **117 GB** of the box's 121 GB unified pool, fell into swap, and
sat in uninterruptible IO-wait for eighteen minutes doing effectively nothing.

The fix is unglamorous and the arithmetic is honest. Peak memory here fits
`≈ 35 GB + 0.32 GB × sequence_length` — a fixed cost (weights, the `W_eff`
copies, activations) plus a per-token term that *is* the recurrence unroll. The
only cheap lever is the sequence length, so I turned it down hard: at 96 tokens
the peak **plateaus at 104 GB** and holds. (I nearly killed that run too, seeing
memory climb and assuming runaway — it was just filling to its plateau. Watching a
number go up is not the same as watching it not stop.) The reference
implementation avoids all of this with a *chunked* parallel recurrence built for
training; our port is the simple sequential form, so we buy headroom with short
sequences instead. That is a real limitation, named plainly: **seq 96
front-truncates most of every diff**, so the model trains on the tail of each hunk,
not the whole thing.

## One epoch, in Rust

At ~33 seconds per step — the price of back-propagating through 96 sequential
recurrence steps across 24 layers — the full rust corpus would take days. The
cookbook slice would take an hour. So: one epoch, 106 examples, **65 minutes**,
memory rock-steady at 102 GB, LoRA on the attention and DeltaNet-mixer projections
(152 linears, rank 32). Out fell a 145 MB PEFT adapter. The loss bounced between
2.2 and 3.4 the whole way — batch size one, no gradient accumulation, heavily
truncated inputs — with only the faintest downward drift. Nothing that would pass
for a clean training curve. A proof of mechanism, not a trophy.

The real test wasn't the loss. It was whether the thing could *review*.

## From assistant to maintainer

The adapter — candle-trained, DeltaNet-mixer targets and all — loaded into vLLM
without a complaint, right next to the base model. So I ran the Part-7 PR through
both, and the difference is the entire point of fine-tuning made visible.

The **base 9B**, on the same hunk:

> *"This change looks good and addresses the ICE described in #158411. **Design &
> Correctness:** The logic is sound: `ty::InstanceKind::Virtual` correctly
> identifies…"* — and on for two more paragraphs.

The **Rust-trained adapter**, same hunk:

> *"I'd suggest to use `is_virtual` instead of `matches!`."*

That is the whole transformation in one line: a hedging, essay-writing assistant
turned into a terse maintainer who names one concrete thing and stops. It holds
across PRs it never saw — *"Please sort the list alphabetically,"* the same
invariant the 27B flagged on that PR; *"use `?` here instead of `// handled
separately below`,"* *"use the same error message for all malformed repr
attributes,"* *"keep `pred` on the left side of the `or`."* Twenty-five hunks in
ten and a half seconds, because terse output is few tokens and vLLM batches them.
After **one epoch, on 106 truncated cookbook examples**, the style transfer is
unmistakable.

## What it is, and what it isn't

It isn't the 27B. Some comments are filler (*"please add a comment"*); it saw a
hundred truncated examples for one noisy epoch, and it shows. A *sharp* Rust-trained
reviewer needs the two things this run couldn't afford: the sequence cap lifted —
which means truncated-BPTT or the chunked recurrence to bound that backward-graph
memory — and real training volume, which means living with ~33 s/step or making it
faster.

But the hole is closed. The crate called `reviewer-train` trains. It trains the
model this whole project is about, entirely in Rust, on a GPU no toolchain admitted
to supporting, and the adapter it produces loads onto the standard serving stack
and reviews real compiler PRs in the voice it was taught. Path A no longer has an
asterisk. The forward pass we spent eleven parts proving turned out to be most of a
trainer — it only ever needed someone to run it backward.
