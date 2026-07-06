# Eighteen of nineteen

*Part 12: the reviewer generates its first real comment in Rust — and why the
one token that didn't match wasn't a bug.*

[Part 11](blog-11-the-scary-parts-were-cheap.md) ended on a deliberate
downer: the reviewer's fifty-four gigabytes of weights were correct, on the
GPU, merged with the fine-tune — and it couldn't say a word. A forward pass
produces logits, a probability over the next token. Turning that into a
sentence needs a tokenizer, a chat template, and a loop. This post is that
gap, closed, and a small detective story about the one place the Rust and
Python outputs disagreed.

## The fixture instead of two literals

The model was trained on a specific chat format: a system turn, a user turn,
and — because this is a reasoning model and reasoning burns the token budget —
an empty `<think></think>` primed into the assistant turn so it answers
instead of deliberating. Get any byte of that wrong and the model isn't
malfunctioning, it's just answering a question you didn't mean to ask.

The tokenizer ships a `chat_template.jinja` that renders the general case —
tools, images, multi-turn conversations, the works. I could have pulled in a
Jinja engine to render it faithfully. I didn't, because the reviewer never
uses the general case: it is always exactly one system turn and one user
turn, no tools, no images, generation-prompt on, thinking off. For that one
shape, the template collapses to a literal string, so `chat::render_prompt`
just writes it:

```rust
format!(
    "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n\
     <|im_start|>assistant\n<think>\n\n</think>\n\n",
    system.trim(), user.trim(),
)
```

That's a bet — "our shape is fixed, so the general renderer is overkill" —
and a bet like that is exactly the kind of thing [Part
9](blog-09-one-plus-the-weight.md) warned about: the code you're confident
about is the code you didn't read carefully. So it doesn't ship on
confidence. `reviewer-train dump-chat-fixture` writes the exact (system,
user) pair — built from the same `reviewer-core::SYSTEM` and `user_prompt()`
the real harness uses — to one JSON file. Both sides read *that file*: a
Python script renders it through the tokenizer's real
`apply_chat_template(enable_thinking=False)`, and `verify-chat-template`
renders it through my hand-written version and diffs the token ids.

```
rust ids   (214): [248045, 8678, 198, 2523, 513, 264, ...]
python ids (214): [248045, 8678, 198, 2523, 513, 264, ...]
  MATCH ✓
```

214 tokens, byte for byte. The fixture is the point as much as the match is —
it's a single source of truth, so "does the Rust template match the real
one" can never quietly drift into "do these two hand-copied strings happen to
agree today."

## The loop that reuses everything

Generation itself is almost insultingly simple, on purpose. [Part
11](blog-11-the-scary-parts-were-cheap.md) closed with a verified forward
pass; the fastest way to burn that trust is to bolt a clever inference engine
onto it before proving the dumb one works. So the first loop has no KV cache
at all:

```rust
for _ in 0..max_new_tokens {
    let logits = full_model_forward(w, &input_ids, &cos, &sin, cfg)?;
    let next = logits.narrow(1, s - 1, 1)?.argmax(D::Minus1)?;
    ids.push(next);
    if eos_ids.contains(&next) { break; }
}
```

Every new token re-runs the *entire* sequence through all sixty-four layers —
O(n²), and no faster than it sounds. It doesn't matter yet. This step exists
to prove one thing: that the verified `full_model_forward` from Parts 8
through 11 is sufficient, unmodified, to generate. No new model code, just a
loop around the old one.

One piece was missing, though. Every oracle so far had *handed* the RoPE
`cos`/`sin` tables to the Rust side, computed once in Python for one fixed
prompt. Generation grows the sequence one token at a time, so now Rust has to
compute its own rotary tables for arbitrary lengths — the first piece of this
whole port that isn't just "translate the reference," it's "derive it."
Reading the actual rotary module turned up a small gift: this model's RoPE is
technically *interleaved M-RoPE* — three separate position streams, for
temporal/height/width, because the same tokenizer serves an image-text model.
For text-only input, though, all three streams carry identical position ids,
which makes the interleaving a no-op. It reduces to plain partial RoPE over
the first quarter of each head. Convenient — but "convenient" is exactly the
kind of claim that gets checked, not assumed, which is where this story
actually starts.

Fed a real rustc hunk through the whole pipeline — tokenize, template,
generate, decode — the milestone from Part 11's to-do list finally lands:

```
comment:
I'm not sure if this is the ...
```

The Rust reviewer, in Rust, said something about a real diff.

## One token didn't match

Verification for this stage means the same thing it's meant since Part 8:
run the real `generate(do_sample=False)` in Python on base-plus-LoRA, run the
same fixture through the Rust loop, diff the ids. Both sides load the actual
27B in bf16, so this isn't cheap — each run is a couple of minutes just to
place fifty-four gigabytes on the GPU. Python's greedy decode, unprompted,
produced a genuinely good review comment and stopped itself at EOF:

```
"I think this is a bugfix, but I'm not sure if it's intentional."
```

The Rust loop matched it for the first generated token, then diverged on the
second. Eighteen of nineteen.

Every previous mismatch in this series turned out to be a real bug — a
transposed axis, a forgotten `(1 + weight)`, a dtype that silently upcast.
So I did not shrug this one off. But re-running the full 27B for every
hypothesis would cost minutes per guess, and the actual question didn't need
the model at all: is the RoPE table I now compute myself *correct*, or is
something further downstream wrong? That's a question you can answer without
touching a single weight.

`train/step_oracle.py` dumps the *real* rotary module's cos/sin for the exact
215-token sequence where the runs disagreed — teacher-forced, no generation,
just the tensors. `reviewer-train verify-rope` loads that and compares it to
my self-computed table, on the CPU, no model in sight:

```
rope cos table vs reference:
  max_abs_diff  = 1.951e-3
  MATCH ✓ (within one bf16 ULP)
```

The RoPE derivation was right. So the divergence lives in the forward pass
itself — except the same script that dumped the rotary tables also dumped
the actual logits at that position, and they answered the question outright:

```
top5 @ last position: [(2688, 18.125), (1683, 18.125), (1459, 17.875), ...]
```

Two different tokens, the same bf16 value, to three decimal places past the
point where bf16 has any more decimal places to give. Candle's kernels and
PyTorch's kernels do the same matrix multiplications in a different order,
accumulate in a different sequence, and round at a different point — and
here, for this one token, that ordinary and unavoidable noise landed two
genuinely tied candidates on exactly the same sixteen-bit float. Whichever
implementation's argmax breaks the tie first wins, and the two frameworks
don't agree on which. That isn't a bug in the port. It's the bf16 floor,
visible for the first time because generation is the first place a single
flipped token becomes the *entire rest of the sequence*, autoregressively —
one bit, cascading forward through everything that depends on it.

## The theme, once more

This is the same lesson as Part 9, wearing different clothes. There, the
danger was code I was confident about *because* it was familiar. Here, the
danger was a mismatch I could have been tempted to write off as *not really a
bug, must be bf16, don't worry about it* — a hand-wave that would have been
right this time and wrong the next. The oracle method doesn't let you assert
that; it makes you show it. `verify-rope` and `step_oracle.py` exist because
"probably bf16" isn't a finding, it's a guess, and the whole discipline of
this series has been refusing to ship guesses. Two cheap, model-free checks
turned a guess into a number: the tie is real, it's exactly one ULP wide, and
it's the same tie a second run of PyTorch against itself on different
hardware could just as easily land on the other side of.

The reviewer talks now — slowly, one O(n²) forward pass per word, but
correctly, verified the same way every layer before it was. What's left is
the part [Part 11](blog-11-the-scary-parts-were-cheap.md) flagged as the real
work: a KV cache for the attention layers, and — because this is a hybrid
architecture — a *second* kind of cache for the DeltaNet recurrent state,
so the notebook doesn't have to reread itself from page one for every new
word it writes.
