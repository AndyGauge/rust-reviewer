# Teaching a model to spot design problems — in pure Rust

*Part 1: building the data pipeline*

I help maintain Rust things. Review doesn't scale the way contributions do, and
I'm not comfortable just throwing more human reviewers at the problem. So I
started a side project: a model that catches **design problems** early — API
shape, leaky abstractions, broken invariants, edge cases, backwards-compat
footguns — on pull requests to `rust-lang/rust` and the
`rust-lang-nursery/rust-cookbook` (which I maintain).

Not to replace reviewers. To give them a first pass that says "wait, have you
considered…" before a human ever looks.

Two self-imposed rules made this interesting:

1. **It's a LoRA**, fine-tuned on top of a ~30B Qwen3 coder model, sized to run
   on the GB10 box (128 GB unified memory) I just ordered.
2. **It's all Rust.** No Python in the pipeline. Partly because I wanted an
   honest feel for what Rust can do across an ML workflow, partly because if
   there's a missing piece, that's a crate worth writing.

This post is about the data pipeline — which, it turns out, is 90% of the
project.

## How far does Rust actually get you?

Before writing a line, I wanted an honest map of where Rust is strong and where
it's pioneering. Here's the one I'm working from:

| Stage | Rust today | Verdict |
|---|---|---|
| **Extract** (GitHub API, paging, rate limits) | `reqwest` + `serde` + `tokio` | Rock solid. |
| **Prepare** (clean, score, chat-template, JSONL) | `serde_json`, `regex` | Rock solid. |
| **Inference** (run Qwen3 + a LoRA adapter) | `candle`, or `mistral.rs` | Strong, production-usable. |
| **LoRA training / SFT** | `candle`/`burn` + `candle-lora` | Doable, but you build the trainer. |
| **QLoRA (4-bit NF4 training)** | Essentially missing | **The gap.** |

The headline: everything *except* the training step is a clean, boring Rust win
today. The 128 GB on the GB10 conveniently lets me do bf16/8-bit LoRA on a 30B
model and sidestep the one true hole (4-bit quantized *training*). So for v1, the
all-Rust constraint costs me almost nothing — and the gaps I did find are exactly
the crate-shaped opportunities I was hoping to surface.

## The shape of the data

The gold for a reviewer model is GitHub's PR **review comments** endpoint. Every
review comment arrives already anchored to the exact lines it's about:

```
GET /repos/{owner}/{repo}/pulls/comments
-> { diff_hunk, path, body, user, in_reply_to_id, ... }
```

That `diff_hunk` is the whole game: a ready-made *(code change → what a
maintainer said about it)* pair. No reconstruction needed.

So the pipeline is three small crates:

```
reviewer-core      shared types
reviewer-extract   crawl the API -> data/raw/*.jsonl  (one comment per line)
reviewer-prepare   clean + score + format -> data/prepared/*.jsonl  (chat)
```

`reviewer-extract` is the unglamorous, important part: page through the `Link:
rel="next"` header, watch the `x-ratelimit-*` headers, sleep until reset when the
budget runs dry, and **checkpoint after every page** so a crash costs at most 100
comments. It resumes from the last `updated_at` it wrote. About 200 lines of
`reqwest` and it Just Works — this is Rust at its most boring and best.

## The interesting problem: most review comments are nits

Here's the trap. If you train on *every* inline comment, you don't get a design
reviewer — you get a nit bot, because most review comments are "typo", "rustfmt",
"trailing whitespace". The reframing from "a reviewer" to "a *design* reviewer"
barely touches the code. It almost entirely changes **data curation**.

So `reviewer-prepare` scores every comment for "design-ness" with a deliberately
simple, transparent heuristic — cheap signals I can explain and ablate:

- **Thread depth** — design discussions spawn reply threads. Strongest cheap
  signal.
- **Design vocabulary** — *invariant, abstraction, API, trade-off, instead of,
  have you considered, footgun*…
- **Substance ≈ length**, and **probing questions** (ends in `?`).
- **Nit penalty** — *typo, rustfmt, whitespace, indentation* drag the score down.

It emits standard chat JSONL (`{"messages":[…], "meta":{…}}`) with full
provenance in `meta` so every training example is traceable back to its comment.

## How's it going? (Live numbers)

I ran the whole pipeline on the cookbook first — small enough to validate fast.
**1,139 raw comments → a design-scored dataset.** Then I swept the threshold:

| `--min-design-score` | Examples | Keep-rate | Category mix |
|---|---|---|---|
| 0.3 | 220 | 19.3% | design 141, **other 55**, question 24 |
| **0.4** | **123** | **10.8%** | **design 117, question 6** |
| 0.5 | 43 | 3.8% | design 43 |

That `other` column is the punchline. The noisy bucket — pasted compiler errors,
side-channel "could you also clean up…" chatter — is **55 comments at 0.3 and
exactly 0 at 0.4**. The whole 0.3→0.4 step is the noise floor falling away. I'm
shipping **0.4** as the default.

And the stuff it keeps is genuinely good:

> "Can we avoid `unwrap()`? We are really trying to teach how to implement these
> functions in production code…"

> "Accurate, but worth adding the footgun std itself documents: `RwLock` gives no
> priority/fairness guarantee…"

> "I would extract only the link-checking logic into a separate function
> `check_link`…"

That's design feedback. That's the thing I want the model to learn.

Meanwhile the rustc backfill is running in the background as I write this —
checkpointing its way through 13 years of review comments, currently somewhere in
late 2013. It'll burn the rate-limit budget, nap until it resets, and carry on.
That's the corpus that'll actually have the volume to train on; the cookbook is
too small for training (43–220 examples is nowhere near enough for a 30B LoRA)
and will instead be my domain eval slice.

## Where this is honestly weak

I'd rather state the limits than oversell:

- **The heuristic has a ceiling.** It can't tell a *pasted compiler error* from
  *design discussion* — both are long and code-flavored — so a few still sneak in
  via thread depth. Fixing that is a v2 LLM-judge labeling pass (still pure Rust,
  over the API) that distills a cleaner score into the same `design_score` field.
- **No negatives.** Every example is a hunk that *got* a comment, so the model
  will learn to always find something. I need to mine approved/un-commented hunks
  as "looks good" examples.
- **Missing context.** Comments assume CI logs and prior discussion the model
  can't see. An "act-on" filter (did a later commit change the flagged lines?)
  should help keep only grounded feedback.

## Next

The rustc crawl finishes, I run `prepare` at 0.4 over the full corpus, and we
find out whether there's enough signal to train on. Then the actual LoRA — which
is where Rust stops being boring and I start finding out exactly how big those
"you build the trainer yourself" gaps really are.

That's part 2. The box hasn't even arrived yet.
