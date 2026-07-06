# The scary parts were cheap

*Part 11: the reviewer's weights now run in Rust on the GPU — and how a verified
forward pass turned three intimidating changes into almost-boring ones.*

[Part 10](blog-10-argmax-ten-of-ten.md) ended with the 9B model running in Rust,
argmax-identical to PyTorch, on the CPU. Between there and here are three changes
that each *sounded* like a project: move it to the GPU — a two-month-old compute
architecture no toolchain admits to supporting; scale from 9B to the 27B, three
times the size; and merge in the actual fine-tune, the LoRA that reviewed rustc.
Every one of them went fast. This post is about why the scary parts were cheap.

## The GPU that wasn't supposed to work

candle, like PyTorch before it, ships its own CUDA kernels, and the GB10's
compute capability — `sm_121` — is new enough that "does the stack support this
GPU" was a real coin-flip. This is the [Part 3](blog-03-bringing-up-the-box.md)
situation exactly: back then PyTorch's arch list swore it didn't support this
card, and the matmul ran anyway.

So I set the compute-cap flag, pointed cargo at the CUDA toolkit, and built. It
compiled — candle's kernels, no arch complaints, 56 seconds. But building isn't
running (the recurring lesson), so I ran the full 9B forward on the GPU. It
matched the reference to `1.05e-5`, argmax ten of ten — *tighter* than the CPU
run, because GPU-f32 rounding lines up more closely with the reference's own GPU
math. **candle's CUDA backend works on this frontier GPU.** The warning label was
wrong again, and the matmul was right again.

## Three times the model, and no new code

The 27B is the same architecture as the 9B — same Gated DeltaNet notebooks, same
hybrid stack — just bigger: 5,120-wide instead of 4,096, sixty-four layers
instead of thirty-two, forty-eight value heads instead of thirty-two. All I did
was lift the hardcoded 9B dimensions into a config read from the model's own
`config.json`, and thread it through. No new algorithm. No new layer. The port
became a function of numbers it reads at startup.

Then I ran the 27B on the GPU and it matched — argmax ten of ten, sixty-four
layers of it. The only friction was two bugs, and here's the part worth noticing:
they were both *loud*. Running the 27B in bf16 (it's 54 GB; it doesn't fit in
f32) surfaced two dtype mismatches — a normalization that upcast to f32 and forgot
to cast back, a causal mask I'd built in f32 while the attention ran in bf16. Both
of them **errored immediately.** `dtype mismatch in add, lhs: BF16, rhs: F32`.

That's the good kind of bug. Compare it to the silent `1 + weight` from
[Part 9](blog-09-one-plus-the-weight.md), which ran fine and poisoned the output.
A dtype mismatch can't hide — the type system stops it at the door. I fixed both
in minutes, because the compiler told me exactly where they were.

## The moment it became the reviewer

The last step is the one I'd been building toward for eleven posts. Everything so
far was a generic port — *a* model in Rust. Then I merged the LoRA.

A LoRA adapter is a small correction to each weight: `W += (α/r)·B·A`, where `B`
and `A` are the skinny learned matrices. My epoch-1 adapter — the one the
[overfitting saga](blog-07-the-overfit-model-hallucinates-a-link.md) picked as the
keeper — touches 496 of the model's linear layers. I loaded it, computed each
correction, added it in. Four hundred ninety-six merges, and the generic 27B
became *my* 27B: the reviewer that flagged the virtual-dispatch ICE and caught the
deleted crash tests, now sitting in candle on my own GPU.

And it matched — the merged model in Rust against the reference base-plus-adapter,
argmax ten of ten. Not a model that reviews like mine. *Mine.*

## Why it was cheap

None of these three steps was hard, and that is the whole point. Each touched a
small, well-defined surface — a build flag, a config struct, a weight merge — and
every one was checked the exact same way: run the reference, run the Rust, compare
the argmax. The scary-sounding changes were cheap because **the thing underneath
them was already proven.** I'd spent [Parts 8](blog-08-the-model-keeps-a-notebook.md)
through 10 verifying the forward pass to eight decimals, component by component.
Once that foundation was solid, putting it on a new GPU, scaling it 3×, and
merging a fine-tune weren't archaeology — they were arithmetic. The rigor
front-loaded the risk. The back half collected the dividend.

This is the case for the oracle method that I couldn't have made at the start:
verification doesn't just catch bugs, it makes the *next* change cheap, because
you're always extending something you already trust. It compounds.

## Runs, not reviews

The honest boundary, because it matters. The reviewer's weights are correct in
Rust, on the GPU, all fifty-four gigabytes of them. But it cannot yet review a
single line of code. A forward pass produces *logits* — a probability over the
next token — not a review. Turning that into an actual comment needs an inference
engine: an autoregressive loop that generates token by token, a KV cache (which
for this hybrid arch means caching both the attention keys/values *and* the
DeltaNet recurrent state — the reason I built the slow recurrent form first), a
tokenizer, and the chat template. That's the gap between "the weights are right"
and "the reviewer works," and it's the next build.

But the wall is down. The reviewer exists in Rust — fifty-four gigabytes of base
and a gigabyte of fine-tune, merged and verified, running on a GPU that never
admitted it could. It just can't talk yet.
