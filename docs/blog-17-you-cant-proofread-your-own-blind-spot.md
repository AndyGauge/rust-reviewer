# You can't proofread your own blind spot

*Part 17: the base model judges the specialist's comments, catches the async
error the specialist can't see in itself — and why this judge is not the one
[Part 5](blog-05-one-box-in-the-diagram.md) talked me out of building.*

[Part 16](blog-16-the-fast-path.md) ended on a specific failure. Run the whole
async cookbook through the reviewer and it confidently flags Tokio's
`tokio::fs::read().await` and `recv().await` as *blocking* — applying a
synchronous-Rust reflex one layer too high. It's not a random slip. It's a
systematic blind spot: the model learned "recv and file I/O block a thread" from
sync Rust and never learned that async I/O is the exception. The same weights that
produce the wrong comment produce the confidence in it. Which is the whole
problem with asking a model to check its own work — a blind spot is, by
definition, the thing you can't see yourself having. You need a second reader.

## The judge I was told not to build

The idea of a second model grading the first isn't new to this series. It shows
up in [Part 5](blog-05-one-box-in-the-diagram.md) — in the list of confident,
AI-generated architecture suggestions I *threw away*:

> **"Make the critic aggressive; the judge filters the noise."** I spent the
> entire data pipeline teaching the model to be *selective*. Prompting it to "be
> exhaustive" pushes it off that distribution to manufacture noise, so you can
> build a second GPU pass to clean the noise back up. Two stages that cancel out.

That rejection still stands, and it's worth being precise about why what follows
doesn't reopen it. I did **not** touch the critic. It's the same selective
epoch-1 reviewer from [Part 7](blog-07-the-overfit-model-hallucinates-a-link.md),
running on its trained system prompt, emitting the handful of comments it thinks
are worth making. I did not prompt it to be aggressive and then hire a mop. The
judge here isn't filtering *noise the critic was told to manufacture* — it's
checking whether the comments the critic *chose* to make are actually correct.
Part 5's version was two stages that cancel; this is two stages that compose. The
difference is entirely in leaving the critic alone.

## What it is

A new `reviewer-run judge` subcommand. It reads the findings store, and for each
finding hands a judge model the diff hunk and the critic's comment on it, asking
for one word — ACCEPT, REJECT, or UNSURE — plus a sentence of reasoning. It runs
them in parallel, because the judge model is served right next to the critic on
vLLM (Part 16's payoff again: two models, one endpoint), and writes each verdict
into a **new `machine` field**, deliberately separate from the `human` field that
holds gold labels.

That separation is the one real design decision, and it's the same one Part 5's
instinct protects. Machine verdicts are a cheap first pass and a training signal;
they are *not* ground truth. Pour them into the `human` field and you quietly
start training your judge model on its own base model's opinions, dressed up as
human judgment. Two distinct streams, so either can be filtered.

The judge is the **base model** — `Qwen/Qwen3.6-27B`, the model the reviewer LoRA
was fine-tuned *from*. That's a deliberate choice: it has the same strong Rust
prior as the critic, but it hasn't been fine-tuned into the reviewer's groove, so
it isn't pattern-locked to the specialist's reflexes. It's a sibling who read the
same books but didn't develop the same tic.

## The run

Twelve async-cookbook critic comments, judged by the base model, eight at a time,
in **21 seconds**: **8 accept, 4 reject, 0 unsure.** The headline is the reject
I was hoping for:

> **fs/create** — critic: *"Should we add a comment that this is blocking?"*
> **base judge → REJECT:** *"The code uses `tokio::fs` functions which are
> non-blocking asynchronous operations, so the reviewer's claim that it is
> blocking is factually incorrect."*

There it is. The exact blind spot from Part 16, caught cold, automatically, by a
model that simply isn't fooled by the sync-Rust reflex. The critic cannot see this
error in itself — the confidence is baked into the same weights as the mistake —
but the sibling that never learned the tic calls it *factually incorrect* in one
line.

And it's a discriminating reader, not a rubber stamp. The other three rejects are
all sound:

- **fs/write** — critic wants a `fs::read` to verify the write; judge: *"the `?`
  already propagates write errors, so an explicit read is redundant for a basic
  example."*
- **fs/rw_traits** — critic wants an explicit `drop`; judge: *"the handles are
  dropped at the end of `main` already."*
- **fs/read** — critic wants a chunked-read variant; judge: *"out of scope for a
  basic `tokio::fs::read` example."*

While it *accepts* the comments that earn it: the missing `block_on` that would
let a spawned task actually run, the timeout example whose future completes
instantly so the timeout never fires, the `JoinSet` example with no error
handling, the recipe filed under the wrong module. That's exactly where you'd
want the accept/reject line drawn — wrong or redundant on one side, real design
gaps on the other.

## What it isn't

Two honest limits. First, the same choice that makes the base a *convenient*
judge — same family, already loaded — makes it an *imperfect* one: shared lineage
means correlated errors. A blind spot the base model *also* has, the judge will
wave through as confidently as the critic raised it. This is a cheap second
opinion, not an independent one; a truly independent judge would be a different
model from a different family, and that's a real endpoint away, not a free one.

Second, the judge did not make the critic *right*. It made the critic's
wrongness *visible and labeled*, which is a different and more honest service. The
reviewer still has an async blind spot after this run. What changed is that the
spot now has four rejects with reasons attached — which is to say, the beginnings
of exactly the training slice Part 16 said to go build: async examples, paired
with the verdict that the "blocking" reflex is wrong.

## Where it leaves it

There are three readers now, and they compose instead of cancel. The critic
*proposes*, selectively, the way [the data pipeline](blog-01-building-an-all-rust-reviewer.md)
taught it. The judge *triages*, cheaply, catching the factual misses. And the
human *confirms* — but now only the contested ones, because a run that used to be
"hand-judge all twelve" is now "the machine and I agree on eight; look at the four
it flagged." The expensive human attention gets spent exactly where two models'
opinions diverge, which is the only place it was ever worth spending.

Part 5 was right that a judge cleaning up after a deliberately-noisy critic is two
stages that cancel. It turns out a judge checking a *selective* critic for the
errors it's constitutionally unable to see in itself is a different machine
entirely — and it earns its place the moment the specialist has a blind spot,
which, as of Part 16, it measurably does. You can't proofread your own blind
spot. But the model you were fine-tuned from can read right over your shoulder.
