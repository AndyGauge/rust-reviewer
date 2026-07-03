# Review: rust-train (code, approach, architecture, writing)

*Reviewed 2026-07-03 by Claude (Fable 5). Scope: all four crates, the `train/`
scripts, all six blog posts, and the supporting docs. `cargo test --workspace`
passes 15/15.*

**Overall:** a genuinely well-built project. The architecture is thoughtful,
the writing is strong, and the real issues cluster in the durability and
failure-path corners rather than the happy path.

---

## Architecture & approach

The three load-bearing decisions all hold up:

- **`reviewer-core` as the skew-killer.** Train and serve share the literal
  `SYSTEM` string and `user_prompt()` function, and `--dump-prompts` lets the
  invariant be verified empirically. What is usually a convention is here a
  compile-time dependency.
- **"The record is the artifact; HTML is a view."** The correct inversion, and
  the `CriticFinding` schema carries what a future judge model actually needs.
  Including `model_version` in the identity hash is a subtle, correct detail —
  findings from different checkpoints are different distributions.
- **The `Critic` trait boundary.** The one model-shaped hole stays swappable,
  and `StubCritic` lets the full capture → render → label loop run and be
  tested with no GPU.

The tests are good tests: the limiter's knee-convergence test proves the
mechanism rather than asserting a magic number, which matches the project's own
thesis. The adaptive limiter is honest about its limits (the non-decaying
`min_rtt` is documented as future work).

**One design tension worth naming:** the system prompt promises *"say it looks
good if there is nothing to raise,"* but the training data contains zero
"looks good" examples. The negatives backlog item is not just a data-quality
nicety — it is the gap between what the prompt asks for and what the model was
ever shown. Consider promoting it above the retracted-comment cleanup.

---

## Code

### Real issues

1. **A non-rate-limit 403 puts the extractor into an indefinite sleep/retry
   loop.** In `get_with_backoff` (`crates/reviewer-extract/src/main.rs:296`),
   any 403 with a parseable `x-ratelimit-reset` header is treated as rate
   limiting — but GitHub sends rate-limit headers on essentially every
   response, including permission-denied 403s ("Resource not accessible",
   SSO-blocked tokens). A genuinely forbidden request sleeps until reset,
   retries, gets 403 again, and loops forever instead of failing with a
   message. **Fix:** gate the rate-limit path on `x-ratelimit-remaining == 0`
   (or the body containing "rate limit"), and cap total retries.

2. **`finding_id` is built on `DefaultHasher`, which is not stable across Rust
   releases** (`crates/reviewer-run/src/findings.rs:150`). The std docs
   explicitly reserve the right to change the algorithm. Label preservation
   depends on this hash being reproducible *forever* — a toolchain upgrade
   could silently change every id, breaking dedup and orphaning all human
   verdicts. **Fix:** use a hash with a specified algorithm (sha2, xxh3 with a
   fixed seed, or a hand-rolled FNV).

3. **`findings::save` truncates the durable record in place**
   (`crates/reviewer-run/src/findings.rs:196`). `File::create` zeroes the file
   before rewriting; a crash or Ctrl-C mid-write destroys every finding and
   every label — the exact data the docs call irreplaceable ("you cannot train
   a judge on comments you didn't save"). **Fix:** write to a temp file and
   `rename()` over the original; same-filesystem rename is atomic.

4. **Failed hunk reviews vanish silently and are never retried**
   (`crates/reviewer-run/src/findings.rs:79-86`). On error the hunk gets an
   empty result and the run reports success. Consequences: (a) a review can be
   quietly incomplete — the missing finding might have been the real one, the
   same argument harness-plan.md makes against aggressive pre-filters; (b) a
   *systematic* failure (bad API key → every request 401s) produces a
   plausible-looking "0 findings" run. **Fix:** count failures and surface them
   in the summary and the HTML banner; ideally retry each failed hunk once
   after backoff, and only feed *overload* errors (429/503/timeout) to
   `limiter.on_error()` — a 401 is not a congestion signal.

### Smaller things

- **`label` hard-codes `is_design_problem: false` for `Unsure`**
  (`crates/reviewer-run/src/main.rs:238-241`). That bakes a confident negative
  label into the judge's training data for exactly the cases the human
  declined to judge. Make it an `Option`, or only ask on accept.
- `clean_body`'s comment claims it drops HTML comments, but the code only
  drops `>`-quoted lines (`crates/reviewer-prepare/src/main.rs:189`). GitHub
  review templates hidden in `<!-- -->` will leak into training targets.
- The concurrent extract mode has no checkpoint — a crash loses the whole
  parallel crawl. Worth one line in the `--workers` help text.
- `OptionUserExt` (`crates/reviewer-core/src/lib.rs:52`) is an extension trait
  for a single call site; inline `self.user.as_ref().map(|u| u.login.as_str())`
  is simpler.
- `HttpCritic::parse_completion` assumes assistant text lands in `content`. A
  vLLM reasoning parser can put output in `reasoning_content` with `content`
  empty — which this code treats as "no finding." Cheap insurance: warn when a
  completion parses to empty.

---

## Writing

The series has a real voice, and the recurring structure — confident wrong
belief, measurement, corrected model — is doing honest work rather than being
a rhetorical tic. Parts 2, 3, and 5 are the strongest; the w7-finishes-first
reversal in part 2 and the Firefox OOM in part 3 are the concrete details that
make technical writing land. The doc numbers check out: the score weights in
`design-score-thresholds.md` match the code, and the shard totals and prepare
stats in blog 2 sum exactly.

Issues, in order of importance:

1. **Blog 3 has a duplicated paragraph** — "The frontier-hardware stack had
   one more missing piece…" appears twice verbatim
   (`docs/blog-03-bringing-up-the-box.md:132-138`).
2. **README.md is stale** — it still says training is "Blocked on GB10
   arrival" while blogs 3–6 document the box training at 29%. The README is
   the front door; it contradicts the story.
3. **`capability-matrix.md` predates the Path B decision** and now conflicts
   with it: it recommends Qwen3-Coder-30B-A3B, and the parenthetical "there is
   no 'Qwen 3.6 35B'" confuses now that the project trains Qwen3.6-27B. Mark
   it superseded by `training-path-b.md` or update it.
4. **`docs/sft_qwen36.py` and `docs/requirements.txt` are stale duplicates**
   of the `train/` versions (the docs copy is missing the length-grouped
   sampler and has an old `--seq` default). Delete them.
5. **The "The theme" closer is becoming a formula.** Parts 5 and 6 both end
   with the growing recap list ("Part 2: … Part 3: … Part 4: …"); by part 7 it
   will be five items. The throughline is a strength; the mechanical
   restatement of it is not. Vary the ending.
6. **The unnamed collaborator.** "A collaborator caught me" / "my collaborator
   killed it with one sentence" recur across parts 5–6, while part 5 opens by
   dunking on confident AI-generated plans. If the collaborator is an AI,
   naming that would sharpen the honesty framing the series trades on; if a
   person, a name would read less evasive. The ambiguity is the one place the
   writing feels less candid than the rest.

---

## Priorities

1. **Atomic save** (code #3) and **stable hash** (code #2) — both threaten the
   labeled data the whole flywheel depends on.
2. **403 retry loop** (code #1) — turns an auth mistake into a hung crawl.
3. **Surfaced/retried failures + Unsure labeling** (code #4, label nit) —
   protects the integrity of both the review output and the judge's training
   set.
4. **Docs cleanup** (writing #1–4) — a ten-minute pass: dedupe the blog-3
   paragraph, refresh README status, reconcile or supersede the capability
   matrix, delete the stale `docs/` script copies.
