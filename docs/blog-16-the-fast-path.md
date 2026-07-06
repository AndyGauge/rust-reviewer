# The fast path was a door we assumed was locked

*Part 16: the frontier architecture we spent the whole project routing around
turns out to run on vLLM — serving our own LoRA, an order of magnitude faster
than the engine we built by hand. Then we point it at the async cookbook and
find the one thing it's confidently wrong about.*

For most of this project, the model we'd chosen — Qwen3.6-27B, a Gated DeltaNet
hybrid — was, in the words of
[Part 11](blog-11-the-scary-parts-were-cheap.md), *an architecture no toolchain
admits to supporting*, on a GPU (`sm_121`) new enough that whether any given stack
would run it was a coin flip. So the project took the long way: prove the
architecture by hand in candle, build a
[generation engine](blog-13-nineteen-for-nineteen.md) — tokenizer, chat template,
KV cache, batching — from scratch, in Rust, so the reviewer could run without
depending on anyone else supporting the arch. Every one of those posts was, in
part, insurance against a door being locked.

This post is me walking up to the door and finding it open.

## It runs

The command is unremarkable, which is the whole point:

```
vllm serve Qwen/Qwen3.6-27B \
  --enable-lora --lora-modules reviewer=…/checkpoint-1000-epoch1 \
  --max-lora-rank 32 --max-num-seqs 128 \
  --gpu-memory-utilization 0.6 --max-model-len 4096 --port 8001
```

vLLM 0.24 on the GB10 box — aarch64, CUDA 13, sm_121, the same unified-memory
machine that's carried this whole project. It came up serving two models:
`Qwen/Qwen3.6-27B` (the base) and `reviewer` (our epoch-1 LoRA). The Gated
DeltaNet layers run through Triton kernels — `fused_recurrent_gated_delta_rule` —
that JIT-compile for sm_121 on first use. The linear-attention state gets its own
pool of what vLLM calls Mamba cache blocks, one per in-flight decode sequence,
which is the reason `--max-num-seqs` is capped at 128 rather than the default 256.
And the LoRA loaded without a fight: vLLM's log even volunteers that
`Qwen3_5ForConditionalGeneration supports adding LoRA to the tower modules`.

Every scary thing we'd hedged against — the hybrid recurrence, the brand-new arch, the
adapter on top of it, on a GPU that didn't exist when the model was trained — is
a line in a startup log now — the same `sm_121` bring-up scare
[Part 11](blog-11-the-scary-parts-were-cheap.md) walked through for candle,
resolved here for vLLM's Triton kernels in one JIT pass. KV cache: 142,904
tokens. Status: READY.

## It's the *fast* path

The reason this matters isn't that vLLM works. It's that it's roughly an order of
magnitude faster than the engine we built.

Here are three real rustc PRs run through the served `reviewer` model, driven by
our own `reviewer-run` harness over the LAN — and this time with the adaptive
concurrency limiter *on*, because vLLM actually batches concurrent requests, so
for the first time there was a reason to let it:

| PR | hunks | wall time | limiter settled at |
|---|---|---|---|
| #158822 | 4 | 11.9 s | ~7 |
| #158819 | 1 | 8.8 s | ~6 |
| #158814 | **25** | **30.9 s** | ~5 |

That 25-hunk PR in 31 seconds is 0.81 hunks/second. The hand-built candle engine,
[measured in Part 14](blog-14-free-but-not-that-free.md), ran at 0.05 hunks/second
sequential and 0.07 in parallel. Same model, same GPU, same weights — roughly
**twelve times** the throughput.

It is important to say exactly *why*, because [Part 6](blog-06-learn-the-controller.md)
already told us and it would be easy to mislead here. vLLM is not faster
*per token*. Nothing is. Single-stream decode on this box is bandwidth-bound:
~273 GB/s over ~54 GB of weights touched per token is a hard ceiling around 5
tokens/second for *any* engine, ours included. vLLM's entire win is that while
one sequence waits on memory, others compute — continuous batching overlaps the
stalls. The candle engine decodes one stream at a time; vLLM decodes twenty-four.
The 12× isn't a better inner loop. It's the same inner loop, kept busy. Part 6's
thesis — *measure it, because the answer changes when the operation does* — holds
one more time: the per-token number didn't move, but the throughput number moved
an order of magnitude, and only one of those is what a review run feels.

## It's still the reviewer

Speed would be worthless if the served model weren't the thing we trained. It is.
On PR #158822, the first comment back was:

> *"I think this is a bug in `should_codegen_locally`, it should return `false`
> for virtual instances."*

That is the *exact* concern the epoch-1 reviewer raised in
[Part 7](blog-07-the-overfit-model-hallucinates-a-link.md), on the same PR,
before any of this serving infrastructure existed. Getting it back, verbatim in
spirit, out of vLLM with the LoRA loaded, is the proof that matters: the adapter
is genuinely applied *through* the Gated DeltaNet stack, not silently dropped.
The DeltaNet-linear LoRA — the part nobody could promise would work when we
started — works.

## Now that it's easy, point it at everything

Here's the part I didn't expect to be the interesting one. Once the reviewer is a
running endpoint, feeding it code stops being a project and starts being a loop.
A cookbook recipe is just code; the reviewer eats a diff; so you wrap the recipe
as a new-file diff and send it. The model was trained partly on
[rust-cookbook](blog-01-building-an-all-rust-reviewer.md), so this is fair game.

The whole `src/asynchronous/` set — 12 recipes with code — went through in **23.7
seconds**, eight at a time, vLLM batching them. The comments sorted themselves
into three piles, and all three are worth reporting honestly.

**The sharp ones** are unmistakably a cookbook maintainer:

> **fs/write** — *"Should we add a `fs::read` example too?"*
> **timeout** — *"a second example that uses `tokio::time::timeout_at`."*

That's the same reflex it showed on the `sort_struct` recipe, where it asked for
the missing `sort_by_key`. Recipes exist to show the idiomatic *set* of
operations, and the model keeps noticing when one of the pair is absent. It also
flagged structure — **rt/tokio-rt-macro** got *"this should be in the `tokio`
directory, not `rt`"*, the same placement instinct that told it, on a rustc PR, to
move a test to `tests/ui` — and on **fs/remove** it emitted an actual
`​```suggestion` block, the GitHub-review affordance it learned from real
maintainer comments, rewriting the recipe with `remove_file` / `remove_dir` /
`remove_dir_all`.

**The soft ones** — bounded channel, `ctrl_c`, the runtime builder — are real
review comments of the *"make this more realistic"* variety. Fine. Not incisive.

**And the wrong ones**, which are the actual finding:

> **channel/unbounded** — *"use `try_recv` here instead of `recv` to avoid
> blocking the main thread."*
> **fs/read** — *"add a note about the blocking nature of these functions."*

These are Tokio's *async* APIs. `recv().await` and `tokio::fs::read().await` do
not block a thread — yielding instead of blocking is the entire reason they
exist. The model is pattern-matching from synchronous Rust, where `recv` and file
I/O *are* blocking, and applying that reflex one layer too high. It's a
plausible-sounding, confidently-stated, and simply incorrect concern — and it
showed up on exactly the two recipes where sync-vs-async is the whole point.

## Where this leaves it

That async blind spot is the most useful thing in this post, and it's only
visible *because* serving got cheap. When a review took 26 seconds a hunk on the
hand-built engine, you ran a few careful PRs. When it takes a second a hunk on
vLLM, you run a whole topic area on a whim and the model's blind spots have
nowhere to hide. The epoch-1 reviewer has a real, diagnosable gap — it hasn't
learned that async I/O isn't blocking I/O — and now I know exactly what a slice of
the next training set should contain.

There's a symmetry worth sitting with. We built the candle engine and the
by-hand generation stack as insurance against vLLM never supporting this
architecture. vLLM supports it, and beats them by 12×. That could read as wasted
work, and it isn't: the hand-built path is *why* we can state, without
hand-waving, that the 12× is batching and not sorcery — Part 6's bandwidth wall
was measured on our own engine, and it's the reason vLLM's number is exactly as
large as it is and no larger. The long way around the door taught us the shape of
the room. But the door was open. The reviewer serves on the fast path, it's the
model we trained, and it just told me, twelve recipes at a time, precisely where
it still needs to learn.
