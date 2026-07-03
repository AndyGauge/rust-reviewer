# The box arrived, and everything lied to me a little

*Part 3: bringing up a GB10 for training*

The [data pipeline](blog-01-building-an-all-rust-reviewer.md) has been done for a
while. [Part 2](blog-02-going-parallel.md) ended with 29,745 design-review
examples and a promise: when the hardware lands, train the LoRA. The ASUS GX10
(NVIDIA GB10, Grace Blackwell, 128 GB unified) arrived. This post is about the
gap between "the box is on my desk" and "a training step actually runs" — which
turned out to be a series of small, instructive lies from every layer of the
stack.

The recurring lesson of this whole project has been *measure, don't assume.* I
apparently need to learn it once per blog post. Here's this one's tuition.

## The box is a good citizen (mostly)

DGX OS is preloaded — Ubuntu 24.04.4 on aarch64, kernel `6.17-nvidia`, 20 Grace
cores, 121 GiB of unified memory, 822 GB free on NVMe. Key auth over SSH, add an
alias, `rsync` the project over. All boring, all fine.

Phase 0 verification looked great:

```
NVIDIA GB10, driver 580.159.03, CUDA 13.0
compute_cap: 12.1   (sm_121 — exactly what Blackwell GB10 needs)
```

And then the lies started.

## Lie #1: "nvcc: command not found"

The CUDA *toolkit* was installed — 13.0.88, sitting right there in
`/usr/local/cuda` — but not on `PATH`. `nvcc` reported missing when it wasn't. A
one-line `.bashrc` append fixed it, but it set the tone: the box knows things it
doesn't advertise.

## Lie #2: "there's no PyTorch for this hardware"

Every 2026 guide I found said the same thing: no official ARM64 + CUDA PyTorch
wheels for Blackwell; use the NGC container, or a community wheel, or build from
source. I braced for a bad afternoon.

Then I actually asked pip:

```
torch-2.12.1+cu130-cp312-cp312-manylinux_2_28_aarch64.whl
```

The official `cu130` index *does* ship an aarch64 wheel now. The guides were
stale. `pip install torch --index-url https://download.pytorch.org/whl/cu130` and
done. The frontier moves fast enough that six-week-old advice is archaeology.

## Lie #3: "your GPU architecture isn't supported"

This one's my favorite, because it's the exact shape of the trap. After
installing torch:

```python
>>> torch.cuda.get_arch_list()
['sm_80', 'sm_90', 'sm_100', 'sm_110', 'sm_120']
```

No `sm_121`. My GPU is sm_121. By the arch list, this build can't run on my card.
If I'd trusted the metadata, I'd have gone off to build PyTorch from source for a
day.

Instead I ran the actual operation:

```python
>>> a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
>>> (a @ a).sum()   # works fine
```

It runs. sm_121 is forward-compatible with the sm_120 kernels (plus PTX JIT), so
the advertised list is pessimistic and the hardware is fine. **The arch list is
metadata; the matmul is truth.** Test the thing you actually care about, not the
label on the box.

## The real bug: unified memory fooled the framework

With torch working and the 55 GB of Qwen3.6-27B weights downloaded, I wrote a
smoke test — the cheapest possible insurance: load the model, attach LoRA, run
*one* forward+backward, confirm the loss is finite. Ten minutes to save a wasted
night.

It failed on the backward pass:

```
Some parameters are on the meta device because they were offloaded to the cpu.
RuntimeError: Function MmBackward0 returned an invalid gradient at
index 1 - expected device meta but got cuda:0
```

`device_map="auto"` — the standard, recommended incantation — had decided the
model didn't fit and *offloaded half of it to CPU*. On a box with 121 GB of room
for a 54 GB model.

Why? Unified memory. On GB10 the CPU and GPU share one pool, and `nvidia-smi`
reports GPU memory as `N/A` (there's no discrete VRAM to report). `accelerate`
reads that, concludes the GPU is tiny, and helpfully offloads to "CPU" — which is
the *same physical memory it was trying to avoid.* Then training dies because
gradients land on the GPU while their parameters sit on the meta device.

The fix is one line — `device_map={"": 0}`, i.e. "put all of it on GPU 0, I know
what I'm doing" — but you only find it because you ran one step before betting the
whole run. That smoke test is the highest-leverage ten minutes in the project.

With the fix:

```
loss     : 4.3627  (finite)
grad norm: 17.4408  (grads flowing)
trainable: 124,730,880 / 27,481,459,440  (0.45%)
PASS
```

A 27-billion-parameter model, learning through a 125-million-parameter LoRA
keyhole, on hardware that had insisted three separate times it couldn't do this.

## Lie #4: "the fast path is ready"

One warning remained: the Gated DeltaNet layers — Qwen3.6's fancy hybrid
linear-attention — were running a slow torch fallback because their fast Triton
kernels weren't available. So I installed `flash-linear-attention`, and Triton
tried to JIT-compile a helper and face-planted:

```
fatal error: Python.h: No such file or directory
```

The frontier-hardware stack had one more missing piece — the Python dev headers
Triton needs to build its CUDA shims. `apt install python3.12-dev`, and the fast
path compiles.

The frontier-hardware stack had one more missing piece — the Python dev headers
Triton needs to build its CUDA shims. `apt install python3.12-dev`, and the fast
path compiles.

## Lie #5: the OOM that a browser caused

With the fast path built, I fired a throughput probe: 25 real steps, so I could
measure steady-state tokens/sec before betting a night. It ran one step... and
died. No traceback — which is itself a clue, because a silent death means
*killed*, not *crashed*. The kernel log:

```
oom-killer invoked ... task=firefox
Out of memory: Killed process 5281 (firefox)
NVRM: ... Out of memory [NV_ERR_NO_MEMORY]
```

The box ran out of memory training a 54 GB model on 121 GB of RAM — because a
**Firefox** was open on a desktop nobody was looking at. Unified memory again:
the GNOME session and the browser draw from the *same pool* as the model, and
the marginal few gigabytes at step two were the difference. The kernel ate the
browser; the training died anyway.

Two fixes. First, kill the desktop: `systemctl set-default multi-user.target`
reclaims the GUI's memory for the pool where it belongs. Second — and this one's
a correctness bug hiding as a warning — the log had been quietly telling me:

```
Packing gathers multiple samples into a single sequence ... may lead to
cross-contamination between samples
```

Sequence *packing* concatenates short examples to fill the context window, but
without a flash-attention variant (not built for sm_121 yet), attention leaks
across the packed boundaries — example A attends to example B. Free throughput,
silently corrupted gradients. Off it went (`packing=False`) until flash-attn
exists for this card.

## The number, at last

Relaunched, GUI dead, packing off. It ran. GPU at 78%, steady state around **2
seconds per example** at batch size 1 — ~16 hours per epoch, a ~2-day run for
three epochs. Exactly the "weekend" my back-of-envelope predicted for a dense
27B, so nothing was wrong. But batch-1 leaves the GPU at 78%, and I *assumed*
batching would fill it and buy back the weekend. This box has taught me what
"assume" gets you, so I measured instead. Three configs:

| Config | Per-example | GPU |
|---|---|---|
| batch 1 | ~2.5s | 78% |
| batch 8, naive | ~9.6s | 96% |
| batch 8, length-grouped | ~3–3.5s | 96% |

Batching made it **worse.** The dataset's lengths are wildly skewed — median 377
tokens, tail to ~16k — so a naive batch pads every short example up to the one
long one in it, and the GPU spends its cycles multiplying zeros. Length-grouped
batching (sorting similar lengths together, which I had to hand-wire because
transformers 5.x dropped the `group_by_length` flag) fixed the padding but *still*
lost to batch-1. The tell is that 78%: the GB10 is **compute-bound** on a 27B, not
memory-bandwidth-bound like I'd assumed. Batching buys throughput only when you're
starved on memory bandwidth with spare compute — here there's no spare compute to
fill, so batching just adds padding overhead and O(n²) attention on longer packed
sequences. Batch-1 wins outright.

(This is the lie I told *myself*, and the only one in the series I caught before
it cost me — because by now I probe first out of reflex.)

So: batch 1, sequence length capped at 2048 to tame the O(n²) tail, three epochs,
bf16, torch fallback for the DeltaNet layers (the *other* half of the fast path,
`causal-conv1d`, is one more thing not yet built for sm_121 — a future speedup, not
a blocker). It's running now, in a tmux session so a dropped SSH won't kill it, and
it'll cook for about two days.

## The theme

Nothing here was a Rust problem, or even a hardware problem. Every obstacle was a
*layer confidently reporting something that wasn't operationally true*: PATH said
no compiler, pip guides said no wheel, the arch list said no support,
`accelerate` said no memory, a Firefox nobody opened ate the training run, a
cheerful "packing enabled!" was quietly poisoning the loss, and my own conviction
that a bigger batch must be faster was backwards. Each was caught the same way:
run the smallest real test of the actual thing and *read what it says* — the
matmul, the backward pass, the kernel log, the training warnings, the stopwatch.

Seven lies, a handful of one-line fixes, and one smoke test standing between me
and a wasted night — and the last one was mine, caught only because this box has
finally trained me to probe before I believe. It's training now, two days on the
clock. Part 4 is the model itself: does a LoRA distilled from 30k rustc review
comments actually catch a design problem?
