# Don't learn the number, learn the controller

*Part 6: adaptive concurrency, and why the batch size that won at training loses
at inference*

[Part 5](blog-05-one-box-in-the-diagram.md) ended promising Part 6 would be the
verdict: a diff the model never saw goes in one end, and we find out whether a
real design concern comes out the other. The model is still training — 29% as I
write this — so that's Part 7. This is what I built while waiting, and it turned
into the sharpest version of this project's recurring lesson yet: *even your own
prior measurement becomes an assumption the moment the operation changes.*

The trigger was a plain question. The harness reviews a PR hunk by hunk. I bought
a GB10 for parallelism. So — how many hunks at once?

There are two dumb answers. One at a time wastes the box. All eighteen at once is
a different kind of guess wearing a confident face. I needed the real number, and
I started to reach for one I already had.

## The conclusion I almost copied

Back in [Part 3](blog-03-bringing-up-the-box.md) I measured the GB10 carefully
and found it *compute-bound* on the 27B during training: batch size 1 was
fastest, and bigger batches just added padding waste. Clean result, hard-won.

So the obvious move was to carry it over. If the box is compute-bound and batch-1
won for training, keep the reviews sequential, right?

Wrong — and wrong in the exact shape this series keeps warning about. Training and
inference are *different operations on the same hardware*, and they live in
opposite regimes:

- **Training** does forward+backward over full sequences — big matmuls across
  thousands of tokens at once. That's compute-heavy. The GPU's arithmetic units
  are the bottleneck, so piling on more work just queues behind them. Compute-bound.
- **Inference decode** generates one token at a time. Each step is a *tiny*
  matmul, and its cost is dominated by reading all 54 GB of model weights out of
  memory to produce that single token. The arithmetic is trivial; the weight
  *fetch* is everything. Memory-bandwidth-bound.

And the GB10's memory is its soft spot: unified LPDDR5X, a few hundred GB/s —
roughly an order of magnitude under the HBM on a datacenter card. So decode isn't
just bandwidth-bound, it's bandwidth-*starved*. Which flips the batching
conclusion completely: when you're re-reading 54 GB of weights per token anyway,
serving one sequence or eight costs nearly the same memory traffic. Batching
amortizes that fixed cost across sequences — it's close to free throughput,
exactly the thing that was pure waste during training.

Same box. Opposite answer. The only thing that changed was the verb. My careful
Part-3 measurement, reused one operation later, would have been just another
confident assumption — the most seductive kind, because it *felt* earned.

## The parallelism isn't where you'd put it

The next reflex is to reach for threads on the laptop — fan the client out. But
the throughput doesn't live there. It lives in the inference server's
**continuous batching**: vLLM (or TGI) packs concurrent in-flight requests into
the same forward passes. The client's only job is to keep enough requests in
flight to *feed* that batch. Sequential requests starve it — each hunk decodes
alone, paying the full weight-read for a single sequence. The concurrency is I/O,
not compute; you don't need a thread per hunk, you need enough outstanding
requests that the server always has a full batch to chew.

## The number isn't a number

So: sweep the concurrency, find the best value, hardcode it. That was my next
plan, and my collaborator killed it with one sentence — *I want the machine to
learn its batch number through experimentation.*

He was right, and here's why the swept-and-hardcoded number is a trap. The
optimal concurrency depends on how much KV-cache each request needs, which
depends on prompt and output length — and those vary wildly hunk to hunk, PR to
PR. A number I measure on today's PR is stale on tomorrow's. The thing that's
actually constant isn't the batch size; it's the *rule for finding it.* Don't
learn the number. Learn the controller.

## Congestion control for inference

Once you say "controller that finds a working concurrency by probing and backing
off," you've described TCP. This is congestion control, and the decades of theory
transfer almost directly. I used a gradient (Vegas-style) limiter:

```
gradient   = min_rtt / recent_rtt          # 1.0 when unsaturated, <1 when queueing
new_limit  = limit * gradient + sqrt(limit) # grow while flat; the sqrt is probe headroom
```

The signal is latency, and continuous batching makes it beautifully readable.
Add a request to a batch that isn't full and it rides the same forward passes —
latency barely moves. Keep adding and eventually the batch saturates; new
requests *queue*, and latency climbs without buying throughput. That inflection —
latency rising off its floor — *is* the knee, and the gradient reads it directly:
while `recent_rtt ≈ min_rtt` it grows (probing up by `sqrt(limit)`), and the
moment latency lifts it shrinks. It settles just past the knee, holding the batch
full without piling on a queue.

One addition TCP taught: an **AIMD backstop**. On this unified-memory box the hard
wall isn't latency, it's KV-cache — push concurrency too far and the server
starts refusing requests (or the box OOMs, and we know how *that* goes here). So a
failed request isn't fatal; it's the strongest possible signal. It triggers a
multiplicative decrease — back off fast, re-probe gently — and the run continues
straight through the overload instead of dying at it.

## Proving a controller you can't run yet

Here's the honest part. The box is training; I can't run this against real
generation latencies for two more days. So how do I know the controller works?

I test the *mechanism*, not the outcome. A unit test simulates a server with a
known saturation knee at 8: flat latency below it, linear queueing above. From a
cold start of 2, the limiter climbs and settles at ~11 — the knee plus its probe
headroom — every time. Back-off, ceiling, and floor behaviors get their own
tests. What I'm proving isn't "the right number for the GB10 is N"; it's "given a
knee, this finds it." The number itself is the machine's to discover on real
hardware.

And I'm keeping myself honest about what I *haven't* shown. Run it today against
the stub critic and it reports "learned ~5" — which is **noise**. The stub has no
real latency, so the gradient is just chewing timing jitter. I could have quietly
shipped that 5 as if it meant something. It means nothing until there are real
generation costs to measure, and the writeup says so. A number that looks like a
result but isn't is worse than no number — it's the horizon effect in miniature,
cost mistaken for results, and this whole project is one long argument against it.

## The permit underneath

There's a harder problem sitting under all this, and I only get to gesture at it.
Every in-flight review holds a concurrency *slot* — a resource acquired when the
request launches and released when it settles. That release is a real obligation,
and async makes it genuinely thorny: if the whole operation is cancelled
mid-flight, a dropped future *cannot run async cleanup* — there's no async `Drop`,
and the code after the await never executes. Structured "closure over async"
shapes (`with_permit(|slot| async { … })`) make cleanup follow the call's
structure instead of a `Drop` impl you hope fires, which is a real improvement on
the success and error paths — but they don't save you from cancellation, where
you still need a synchronous best-effort backstop. That's live research, not a
thing I've solved; I just now understand *why* the slot, not the stateless HTTP
call, is where the safety question actually lives.

## The theme

This project's one lesson keeps changing costume. It arrived as *measure before
you optimize*, came back as *the label is not the operation*, and turned up again
as *watch the curve you didn't ask for*. Here it wears its sharpest form: **even
a measurement you earned becomes an assumption the moment the operation changes —
and the most honest answer to "what's the right number?" is often to build the
thing that never stops measuring it.**

I spent this project learning to distrust confident numbers, including my own.
The natural endpoint of that distrust isn't a better guess; it's handing the
measurement to a controller that re-earns it every run. I stopped trying to know
the batch size. I built the box a way to find out.

Part 7, at last, is the verdict — as soon as the GB10 finishes learning to review
and finally has the cycles to be asked.
