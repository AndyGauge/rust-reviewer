# Review harness plan

*How an actual PR review runs, end to end. Sketch — to be refined once there's an
adapter to measure.*

The design principle: **the pipeline is deterministic Rust with exactly one LLM
stage in the middle.** Fetch, parse, ground-check, distill, and render are all
plain code you can unit-test. The model's only job is `hunk → design comment` —
the one thing it was trained to do. Everything the model *doesn't* need to do, it
doesn't do.

```
 ┌─────────┐   ┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐
 │ 1 FETCH │──▶│ 2 SEGMENT│──▶│ 3 REVIEW    │──▶│ 4 GROUND │──▶│ 5 DISTILL│──▶│ 6 EMIT │
 │ (Rust)  │   │ (Rust)   │   │ (LLM, Spark)│   │ (Rust)   │   │ (Rust±LLM)│  │ (Rust) │
 └─────────┘   └──────────┘   └─────────────┘   └──────────┘   └──────────┘   └────────┘
      │                                                                            │
      └── existing human comments ────────────────────────────────────────────────┘
                              (overlap-marking in stage 5)
                                                                          ┌──────────────┐
                                                                          │ 7 HUMAN LOOP │
                                                                          │  accept/reject│──▶ feedback data
                                                                          └──────────────┘
```

## Stage 1 — Fetch (Rust, no model)

Reuses the GitHub client already in `reviewer-extract`. Given a PR number:

- **Metadata**: title, body, base ref, author, changed-files count.
- **The diff**: the PR diff is already **base-relative** (GitHub computes it
  against the merge-base), so "diff against main" is mostly free. Caveat: if the
  PR is stale, that diff is against an *old* main; reviewing against current
  `main` means recomputing the merge-base. Start with the PR-provided diff;
  revisit only if staleness bites.
- **Existing comments**: both `pulls/{n}/comments` (inline, position-anchored)
  and `issues/{n}/comments` (general). We grab these for two reasons: (a) mark in
  the final report which of our concerns a human already raised, and (b) they're
  the ground truth for eval — real human design comments to score the model
  against.

Output: a `FetchedPr` struct (metadata + unified diff + `Vec<ExistingComment>`).

## Stage 2 — Segment into hunks (Rust, no model)

Parse the unified diff into per-file, per-`@@`-block hunks. **The hunk is the unit
the model trained on**, so the format here must be *byte-identical* to what
`reviewer-prepare` produced (see the skew warning below). Each unit carries: file
path, the hunk text with context lines, and hunk header line numbers.

Optional pre-filter: skip hunks with no plausible design content (pure whitespace,
`Cargo.lock`, generated files, vendored code). Keep this **conservative and
off by default at first** — better to over-review and measure the false-positive
rate than to silently drop the hunk that had the real problem.

Output: `Vec<Hunk>`.

## Stage 3 — Agentic hunk review (LLM, on the Spark)

For each hunk, call the served adapter with the **exact SYSTEM prompt from
`reviewer-prepare`** and the hunk formatted exactly as in training. Output: zero
or more `RawComment { file, line, body }`.

On the word "agentic" — a real decision to make *by measurement, not assumption*:

- **Start one-shot (no tools).** The adapter was trained on `{hunk → comment}`,
  not on tool-use traces. It has no learned competence at calling a
  "fetch more context" tool, so bolting tools on first is off-distribution and
  likely adds noise, not signal.
- **Add a `fetch_context` tool only if the solo numbers demand it** — i.e. if
  measurement shows the model misses concerns specifically because a hunk
  references something not shown (a trait def, a caller). Then a *single*
  narrow tool ("show me the definition of X") is the minimal agentic step. Prove
  the need before paying the complexity.

Batching: recall from part 3 the box is compute-bound, so concurrency across
hunks buys throughput only up to GPU saturation. One request at a time, streamed,
is the honest default; measure before parallelizing.

## Stage 4 — Ground-check (Rust, no model) — the cheap judge

Deterministic filter, no second GPU pass: does each `RawComment`'s cited
file+line actually exist in the hunk it came from? Drop or flag the ones that
don't. This catches the most common hallucination (a comment about a line that
isn't there) for free, in ~30 lines of Rust, with zero model risk. This is the
half of "judging" that does *not* need an LLM.

The *semantic* half — "is this a real design concern or a nit" — is deferred. We
do **not** build an LLM judge until stage-3 measurement proves the solo critic's
false-positive rate is high enough to justify it, and if we do, it should be a
*different* model family so its blind spots aren't the critic's (see
[the actor-critic discussion](blog-04-watching-it-learn.md)).

Output: `Vec<GroundedComment>` + a `dropped` list (kept for traceability/feedback).

## Stage 5 — Distill (Rust, LLM optional)

Roll per-hunk comments up into a PR-level review:

- **Dedup / cluster** related concerns (the same API-shape issue flagged across
  three files becomes one concern with three anchors).
- **Overlap-mark** against stage-1 existing human comments: tag each concern
  `new` vs `already-raised-by-human`. This is what makes the artifact worth
  reading — it surfaces what the model saw that the humans *didn't*.
- Optional one LLM call for a prose PR-level summary, clearly separated from the
  grounded per-hunk concerns so a hallucinated summary can't launder itself as a
  cited finding.

## Stage 6 — Emit artifacts (Rust, no model)

Render locally on the Mac:

- **HTML report** (`maud`/`askama`/`tera`) opened via `file://` — code blocks,
  severity, per-concern hunk + anchor, `new`/`already-raised` badge, and the
  dropped/ungrounded list for transparency.
- **Structured JSON** — the same data typed, for traceability and diffing runs.

No network, no cloud — sovereign, as intended.

## Stage 7 — Human in the loop (closes the data flywheel)

You read the report and **accept/reject each concern.** Two payoffs:

1. It's a review *aid*, not an auto-reviewer — you stay the decision-maker
   (matches the explicit goal: not scaling up human reviewers, sharpening one).
2. **Every accept/reject is labeled feedback.** Logged, it becomes the
   gold-standard set for a v2 relabel and negatives-mining — the exact backlog
   items in the README. The harness generates its own next training data.

Posting accepted comments *back to the PR* is a separate, **write-gated,
off-by-default** action — an outward-facing publish that must ask first every
time. Given the stated goal (sharpen your own review, not post bot comments),
this likely stays off; it's here as an option, not a default.

## The train/serve skew trap (do this or the adapter silently underperforms)

The hunk formatting in stage 2/3 **must** be the same code that
`reviewer-prepare` used to build the training examples. If prepare formats a hunk
one way and the harness another — a different header, extra whitespace, a
different SYSTEM string — the model sees inputs unlike anything it trained on and
quietly degrades, with no error to tell you why. **Fix: extract the formatting +
SYSTEM prompt + message-shape into a shared `reviewer-core` crate** that both
`reviewer-prepare` and the harness depend on. One definition, no skew.

## Crate layout

```
crates/
  reviewer-core     NEW  shared types + hunk/message formatting + SYSTEM prompt
  reviewer-extract  ~    GitHub client (reuse; depend on -core)
  reviewer-prepare  ~    training-data prep (reuse; depend on -core for format)
  reviewer-run      NEW  the harness: fetch→segment→review→ground→distill→emit
```

`reviewer-run` is a plain CLI first: `reviewer-run --pr 12345 --repo rust-lang/rust
--out review.html`. Testable, scriptable, sovereign. Every deterministic stage
gets unit tests against recorded fixtures; only stage 3 needs the Spark.

## On "skills related?" — the seam, added last

The harness is a **Rust CLI first**, because that's the thing you can test and run
without any agent. Once it works, there are two optional wrappers for driving it —
build them *after* the CLI, not instead of it:

- **Claude Code skill** (`/review-pr 12345`): a thin skill that shells out to
  `reviewer-run` and reads back the artifact. Lowest-friction way to invoke the
  harness from the Claude Code sub you already use for dev — the seam between
  "the 20/mo account that writes the software" and "the local box that runs
  inference." Good ergonomics, ~no code.
- **MCP server**: expose `review_pr` as an MCP tool so *any* client — a Rig
  agent, Claude Code, whatever — can call it over the protocol. More portable
  than a skill, more work. Worth it only if something other than Claude Code
  needs to call the harness.

Recommendation: CLI now, Claude Code skill when the CLI works, MCP only if a
second client appears. Don't let the wrapper precede the thing it wraps.

## Build order (measure-first)

1. `reviewer-core`: pull the formatting/SYSTEM prompt out of `reviewer-prepare`
   into the shared crate (kills skew before it exists).
2. `reviewer-run` stages 1–2 + 6: fetch a real PR, segment it, render the diff
   to HTML — **no model yet.** Proves the deterministic spine end to end.
3. Wait for the adapter; wire stage 3 to the served endpoint with the training
   SYSTEM prompt; run one real hunk.
4. Stage 4 ground-check + stage 5 overlap-marking.
5. **Measure** on held-out PRs with real design comments (blog 4's test). Only
   *then* decide whether stages need the LLM judge or the agentic tool.
6. Stage 7 feedback logging; skill wrapper.

Steps 1–2 are buildable **today**, while the model cooks.
