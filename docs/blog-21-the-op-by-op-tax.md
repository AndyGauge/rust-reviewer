# The op-by-op tax

*Part 21: NVIDIA ships a tile-based Rust GPU kernel library the same week the
series needs a faster inner loop. Getting it to run on the GB10 takes a CUDA
toolkit the box doesn't have and three errors that don't say what's wrong. Then
the Gated DeltaNet recurrence becomes a single cuTile kernel that matches the
oracle, gets wired into the real model, and generates tokens: prefill gets up to
2.9× faster, decode doesn't change, and the number that matters turns out to be
the one the kernel benchmark can't see.*

I heard about [cuTile Rust](https://github.com/NVlabs/cutile-rs) the way most
people did: it came out of RustConf 2026, where Melih Elibol's talk
"Fearless Concurrency on the GPU" was on the schedule, and NVIDIA published it
on 8 September as one half of "CUDA Rust" (the other half, `cuda-oxide`, is
the classic one-thread-per-lane SIMT model). cuTile is the tile half. You write
a kernel as ordinary Rust over *tiles* of a tensor, and a `#[cutile::module]`
macro captures the AST, lowers it through CUDA Tile IR, and JIT-compiles it to
a cubin for whatever GPU you're on. Mutable tensors are partitioned into
disjoint tiles before launch, so a kernel can't race with itself by
construction. NVIDIA's own numbers are for a B200: about 7 TB/s on
element-wise ops, about 2 PFlop/s on f16 GEMM.

This series has a Rust trainer, [Path A](blog-19-the-trainer-that-never-trained.md),
whose forward pass is a pile of candle tensor ops. Every one of those ops is a
separate kernel launch and a separate trip through memory. A library that lets
you write the fused version in Rust, and has already shipped inside
Hugging Face's Grout and mistral.rs, is exactly the thing to measure against.
So: does it run on the GB10, and is it actually faster than what the trainer
does today?

**The short version, so you don't have to read to the end:**

- **It runs on the GB10**, but only after unpacking a newer CUDA toolkit into my
  home directory and chasing three errors, one of which (`tileiras` quietly
  needing `libnvvm`) says nothing useful.
- **On simple ops it's fast, but candle already has the cheap win.** cuTile's
  RMSNorm and softmax are about 6× faster than the op-by-op versions the model
  uses, yet candle's own fused `rms_norm` is within 1.05 to 1.33× of cuTile. For
  RMSNorm you don't need a new toolchain; you need to call a function that
  already exists. For softmax at wide rows, cuTile genuinely wins (2.9× at
  N=32768).
- **The Gated DeltaNet recurrence is where it pays.** As a single cuTile kernel
  it matches the PyTorch oracle (4.5e-8) and runs about 35× faster than the
  candle loop in isolation.
- **Inside the real model, generating tokens, the win is smaller and
  lopsided.** Time to first token on the 9B drops from 25.7 s to 8.9 s on a
  4,400-token prompt (2.9×), but only about 1.3× on a 350-token one. Decode
  speed doesn't change at all. The whole-model logits check passes, 10 of 10.
- **The reason the win shrinks is the useful finding.** Once the recurrence is
  gone, about 4.3 seconds of prefill remains that doesn't depend on prompt
  length. That's the next bottleneck, and no kernel in this post touches it.
- **Caveats up front:** it's inference only (no backward pass, so no training),
  I only measured the 9B, and on the two longest prompts the output text differs
  from the loop's. I think that's bf16 rounding in the loop and not a bug in the
  kernel, but I haven't proved it.
- **What comes next,** in order: find out what that 4.3-second floor is, build a
  decode path that doesn't pay for the bridge, run the 27B once the box has the
  memory, and write the backward kernel that training needs.

The post goes in that order: the setup and a baseline on RMSNorm and softmax,
then the recurrence kernel and how it was checked, then the kernel running
inside the model and generating tokens, then what to distrust about all of it.

## Getting it to run

The README says GB10 (`sm_121`) is supported and has a tutorial for it. The
box had other opinions.

**The toolkit was too old.** cutile-rs needs CUDA 13.2 at minimum and
recommends 13.3. The box has 13.0, and `sudo` on it wants a password. The apt
repository NVIDIA configures there already lists 13.3 packages, though, so
instead of installing anything I downloaded the debs with `apt-get download`
and unpacked them into a directory under my home with `dpkg -x`. The system
CUDA and the gpt-oss server that was running on the box at the time never
noticed. The set I ended up needing: `cuda-tileiras`, `cuda-nvcc`, `cuda-crt`,
`cuda-cudart` and its `-dev`, `cuda-driver-dev`, the `curand`/`cublas`/
`cusolver`/`cusparse`/`nvrtc` dev packages, and `libnvvm`. That last one is
where the time went.

**Three failures, in order:**

1. `fatal error: 'stddef.h' file not found`, from bindgen while building
   `cuda-bindings`. libclang couldn't find its own builtin headers. Pointing
   `BINDGEN_EXTRA_CLANG_ARGS` at gcc's include directory fixed it.
2. `fatal error: 'curand.h' file not found`. The bindings include it, and my
   first unpack hadn't. More packages.
3. The interesting one. The crate built, the kernel lowered to Tile IR, and then:

   ```
   tileiras failed while compiling Tile IR bytecode.
   stderr: error: failed to compile Tile IR program
   ```

   Even `saxpy` failed, at every optimization level, so it wasn't the kernel.
   Running `tileiras` by hand under `strace` showed it looking for
   `nvvm/lib64/libnvvm.so` next to itself and not finding it. `libnvvm` is a
   separate package from the compiler driver, and `tileiras` needs it at
   runtime without saying so. After `apt-get download libnvvm-13-3`, `hello_world`
   printed its line and `saxpy` printed `2 * 31 + 31 = 93`.

The lesson is the same one [Part 20](blog-20-sixteen-of-sixty-four.md) ended
on, from the other direction: the error message described the symptom, and the
cause was in a file lookup you could only see by watching the process. The
whole environment is now a sourceable script on the box, `~/cutile-env.sh`,
alongside Rust 1.99.0 (the crate wants 1.89 or newer; I updated the box's
toolchain while I was in there).

## The baseline

cutile-rs ships criterion benchmarks. Two of them map straight onto the
trainer: RMSNorm and softmax, on a 4096-row f16 tensor with the row width N
swept from 1024 to 32768. I ran those as they are, with the GPU idle and the
clocks left unlocked.

For the comparison I wrote `bench_ops`, a small binary in `reviewer-train`
that times the same shapes in candle on the CUDA backend and reports
throughput the same way: one read plus one write of the whole tensor. I
checked that definition by reproducing cutile-rs's own reported GB/s from its
reported times before trusting it. Three implementations:

- **cuTile**: the cutile-rs kernels.
- **candle op-by-op**: `rmsnorm` copied from `model.rs` (square, mean, affine,
  sqrt, divide, multiply, each its own kernel, plus a tiny one for the weight), and `candle_nn::ops::softmax`.
  This is what the trainer runs today.
- **candle fused**: `candle_nn::ops::rms_norm` and `softmax_last_dim`, one
  kernel each, which the trainer does not currently use.

<iframe src="assets/cutile-gb10.html#embed" title="Throughput of RMSNorm and softmax on the GB10: cuTile, candle fused and candle op-by-op" style="width:100%;height:480px;border:0;" loading="lazy"></iframe>

*Hover a point for the numbers. [Open the chart on its own page](assets/cutile-gb10.html).*

| | N | cuTile | candle fused | candle op-by-op | cuTile vs fused | cuTile vs op-by-op |
|---|---|---|---|---|---|---|
| RMSNorm | 1024 | 98 µs | 131 µs | 508 µs | 1.33× | 5.2× |
| RMSNorm | 4096 | 296 µs | 310 µs | 1871 µs | 1.05× | 6.3× |
| RMSNorm | 32768 | 2338 µs | 2523 µs | 14203 µs | 1.08× | 6.1× |
| Softmax | 1024 | 109 µs | 59 µs | 631 µs | 0.54× | 5.8× |
| Softmax | 4096 | 350 µs | 446 µs | 2230 µs | 1.27× | 6.4× |
| Softmax | 32768 | 2554 µs | 7339 µs | 16095 µs | 2.9× | 6.3× |

## What the numbers say

**The op-by-op tax is about 6×, at every width.** The model's RMSNorm moves
37 GB/s. cuTile moves 230. That isn't a clever-kernel result; the op-by-op
version launches a kernel per step and makes about five full passes over a tensor
that doesn't fit in cache, and a bandwidth-bound op pays for each pass. Softmax is the same
story at 33 GB/s against 210.

**cuTile tops out near 230 GB/s on this box.** That's about what a
bandwidth-bound kernel should reach on the GB10's unified LPDDR memory, which
the spec puts at roughly 273 GB/s (that figure is from memory and I haven't
checked it against the datasheet; the chart marks it as such). The B200's
7 TB/s is a different regime and the number to carry from this box is the
relative one.

**Candle's fused RMSNorm is nearly as good, which is the surprise.** On
RMSNorm the gap between cuTile and a single fused candle kernel is 1.05 to
1.33×. Almost all of the 6× is recoverable by calling `ops::rms_norm` instead
of hand-rolling the six ops, with no new toolchain, no JIT, no CUDA 13.3. I
didn't expect that, and it's the most immediately useful result here.

**Fused softmax doesn't scale; cuTile's does.** Candle's `softmax_last_dim` is
the fastest of the three at N=1024 (59 µs against 109 µs), roughly even at
2048, and then falls off: 73 GB/s at N=32768 while cuTile climbs to 210. That
looks like a kernel that doesn't use enough parallelism per row as rows get
wide. Whether that matters depends on what widths the model really sees, and
the answer is probably "narrower than 32768".

## The real target: the recurrence

RMSNorm and softmax were the warm-up. The reason to want cuTile in this repo is
the Gated DeltaNet recurrence, because it's where the trainer's forward pass
spends its time and it has no fused candle op to fall back on. [Part 8](blog-08-the-model-keeps-a-notebook.md)
ported it to candle as a loop: for each timestep, decay the state, read it with
the key, compute the delta, apply an outer-product update, read it with the
query. Per head, the state is a 128 × 128 matrix. Per timestep that's about
twenty small candle ops, each its own kernel launch, and the state round-trips
through memory between all of them.

The shape of the problem is a good fit for a tile language. The recurrence is
sequential in time but independent across (batch, head) pairs, so one program
per pair, with the whole `[Dk, Dv]` state held in registers for the entire
loop. Inputs are read once, the state never touches global memory, and the
L2-norm of `q` and `k` fuses into the same kernel. The core of it:

```rust
let mut state: Tile<f32, { [DK, DV] }> = constant(0.0f32, shape![DK, DV]);

for t in 0i32..S {
    // ... load q_t, k_t, v_t, g_t, beta_t; l2-normalize q and k; scale q ...

    state = state * exp(g_s).broadcast(shape![DK, DV]);              // 1. decay
    let kv_mem = reduce_sum(state * k_n, 0i32);                      // 2. read with key
    let delta = (v_row - kv_mem.reshape(shape![1, DV]))
        * b_s.broadcast(shape![1, DV]);                              // 3. delta
    state = state + k_n * delta.broadcast(shape![DK, DV]);           // 4. outer-product update
    let out_t = reduce_sum(state * q_n, 0i32);                       // 5. read with query
    out_part.store(out_t.reshape(shape![1, 1, DV]), [0i32, t, 0i32]);
}
```

Those five commented steps are line for line the five in `delta.rs`. The
kernel is the recurrence with the loop moved onto the GPU.

**Four things the compiler told me, in order.** The library is young and its
errors are literal, which made them quick to fix:

1. The launcher zips at most six operands, and my first signature had seven
   (output, five tensors, and a scale). The scale became `rsqrt` of the key
   dimension computed inside the kernel.
2. Partitioned (mutable) tensors can't have rank above three, so batch and head
   collapse into one axis: everything is `[B*H, S, D]`, and the grid is one
   program per row of that.
3. `constant()` takes only literals, so a value derived from a const generic
   has to go through `convert_scalar` and then `broadcast`.
4. A cast nested inside an expression needs its type spelled out on a `let`.

None of those are the kind of thing that costs a day. They're a half-hour of
reading error text, which is a reasonable price for a first kernel.

**Checking it.** The oracle that `verify-delta` uses, `delta_synth.safetensors`
(a transformers `torch_recurrent_gated_delta_rule` call on a 1 × 5 × 2 × 4 × 4
input), loads into the new crate directly:

```
cuTile gated delta recurrence vs transformers oracle (1x5x2x4x4):
  max_abs_diff  = 4.470e-8
  mean_abs_diff = 1.069e-8
  MATCH ✓
```

That's well inside the `1e-4` bar the candle port is held to, and the same
order as the candle port's own `2.98e-8` in [Part 8](blog-08-the-model-keeps-a-notebook.md).
A four-element
state is a tiny test, so I also ran four shapes against a plain-Rust CPU
reference of the same math, growing up to the real one (1 × 200 × 48 heads ×
128 × 128). The worst difference across all four was 1.2e-7. This is the
same ladder-of-oracles habit as [Part 10](blog-10-argmax-ten-of-ten.md): match at
toy scale first, then at the shape that matters.

**The benchmark.** Real Qwen3.6-27B dimensions: 48 value heads, `Dk = Dv =
128`, batch 1, f32. The cuTile column times the kernel only (inputs uploaded
once, output left on the device). The candle column is the real
`recurrent_gated_delta_rule` from `delta.rs`, on the GPU, including the
transposes it does on the way in and out.

| Sequence length | cuTile | candle | speedup |
|---|---|---|---|
| 64 | 0.45 ms | 15.2 ms | 34× |
| 256 | 1.76 ms | 61.2 ms | 35× |
| 1024 | 7.05 ms | 251 ms | 36× |
| 2048 | 14.1 ms | 502 ms | 36× |

Both scale linearly with sequence length, as a sequential recurrence should.
cuTile costs about 6.9 µs per timestep; candle costs about 245 µs. That ratio
is flat from 64 tokens to 2048, which is the signature of a per-step overhead,
not of anything that depends on the data. Candle's side is launch-bound: twenty
small kernels per step, each too small to keep the GPU busy, so most of the 245
µs is the CPU enqueueing work and not the GPU doing it. Some of the gap is the
Grace CPU's launch latency, which a faster host would shave.


## Generating tokens with it

A fast kernel in its own binary proves nothing about the model. The real test is
the model doing inference with it, so I wired it into `reviewer-train` behind a
cargo feature (`--features cutile`) and a run-time switch
(`REVIEWER_DELTA=cutile`). Everything else in the model is the same code.

**The bridge.** Candle owns all the memory and cuTile borrows it. cutile-rs can
wrap a foreign device allocation as a tensor without a copy
(`Tensor::from_foreign`), so a candle CUDA buffer becomes a cuTile tensor by
implementing a small trait that returns its device pointer, and by keeping the
candle tensor alive inside the wrapper so the memory can't be freed mid-kernel.
Candle allocates the output buffer, cuTile writes into it, and candle reads it
afterwards. The pointers work in both libraries because they address the same device
memory, and the bridge synchronizes the device on either side of the launch so
the two never touch a buffer at once.

**What changed in the kernel** to make it usable for inference:

- **State in and out.** Decode needs to seed the recurrence from the previous
  step's state and hand back the new one. The launcher allows only one mutable
  output, so the output buffer is `[B·H, 128 + S, Dv]`: the first 128 rows are
  the final state and the rest are the per-step outputs. My first attempt took
  two mutable views of that buffer and the borrow checker said no, so the state
  is written one row at a time at the end.
- **A runtime step count.** This one was a real find. The first version took `S`
  as a compile-time constant, so I rounded prompts up to a multiple of 64 to
  limit recompiles. Then the JIT times showed up in the logs: 2.5 s to compile
  for 384 steps, 3.3 s for 576, 6.4 s for 1216. Compile time was growing with
  the sequence length, which means the compiler was unrolling the loop. Reading
  the step count from the tensor's shape at run time fixed it: every size now
  compiles in 0.6 to 0.8 s, once, and no padding is needed.
- **A guard against training.** The kernel has no backward pass, so a training
  run with the switch on would silently get no gradients. `train` now refuses to
  start if `REVIEWER_DELTA=cutile` is set.

**Is it still right?** Two checks. The bridge has its own verifier,
`verify-delta-cutile`, that runs the kernel through candle against both the
transformers oracle and the candle loop, with a nonzero initial state, an
awkward sequence length, bf16 and f32, and a decode step continuing from the
kernel's own state. Nine comparisons, nine matches: 4.5e-8 against the oracle,
about 1e-7 in f32 against the loop, and under 8e-3 in bf16 (where the loop does
its arithmetic in bf16 and the kernel keeps the state in f32, so some difference
is expected). The stronger one is `verify-model`, the whole 9B model's logits
against the PyTorch dump, with all 24 DeltaNet layers going through the kernel:

| | max logit diff | mean logit diff | argmax agreement |
|---|---|---|---|
| candle loop | 0.266 | 0.0284 | 10 / 10 |
| cuTile kernel | 0.323 | 0.0267 | 10 / 10 |

The reference is f32 and the model runs in bf16, so both are dominated by bf16
noise, and neither is meaningfully closer. The kernel doesn't make the model
worse.

**The inference benchmark.** Qwen3.5-9B in bf16 with the reviewer adapter merged,
greedy decoding, batch 1, on the GB10. The prompts are six real hunks from the
training data, 351 to 4,419 tokens. Each is run twice in one process and I report
the second pass, so every kernel size is already compiled. (The gpt-oss server
was resident on the box but idle, as it was for every measurement in this post.)

<iframe src="assets/cutile-inference.html#embed" title="Prefill time for six prompts, split into DeltaNet recurrence and the rest of the model, candle loop against cuTile" style="width:100%;height:520px;border:0;" loading="lazy"></iframe>

*Hover a bar for the split. [Open the chart on its own page](assets/cutile-inference.html).*

| Prompt tokens | candle loop | cuTile | speedup | recurrence alone |
|---|---|---|---|---|
| 351 | 5.48 s | 4.35 s | 1.26× | 1.43 s → 36 ms |
| 375 | 5.62 s | 4.40 s | 1.28× | 1.53 s → 37 ms |
| 511 | 6.31 s | 4.38 s | 1.44× | 2.08 s → 47 ms |
| 513 | 6.24 s | 4.36 s | 1.43× | 2.10 s → 47 ms |
| 1,188 | 9.41 s | 4.68 s | 2.01× | 4.87 s → 182 ms |
| 4,419 | 25.65 s | 8.92 s | 2.87× | 18.27 s → 1.09 s |

Time to first token drops by about a fifth to a third on the short prompts and
by two thirds on the longest. Across the six prompts, total prefill falls from
58.7 s to 31.1 s. The recurrence itself got 40× faster on short prompts and 17×
on the longest, consistent with the standalone kernel benchmark; the bridge's
own work (casts, transposes, the output buffer) takes a bigger share of what's
left.

**The part the kernel benchmark couldn't tell me.** The recurrence is only part
of prefill, and it is a growing part. With the candle loop it was 26% of prefill
at 351 tokens and 71% at 4,419. Remove it and what's left is the rest of the
model, which is about 4.3 seconds *regardless of prompt length* up to 513 tokens
(4.1 to 4.4 s before and after, so the kernel isn't what moved it). A floor that
flat says the rest of prefill is dominated by something other than the amount of
work, probably a long chain of small ops. That's the next bottleneck and this
kernel does nothing about it. This is Amdahl's law with real numbers in it: a
36× speedup on a quarter of the time is a 1.3× speedup.

**Decode didn't move.**

| | candle loop | cuTile prefill only | cuTile prefill and decode |
|---|---|---|---|
| decode tokens/s (351 to 513 tokens) | 10.2 to 10.3 | 10.1 to 10.2 | 9.6 |
| decode tokens/s (4,419 tokens) | 8.3 | 8.2 | 7.8 |

At one token there is nothing to fuse: the loop's twenty tiny kernels and the
bridge's thirty-odd candle ops around one cuTile launch cost about the same,
and sending decode through cuTile as well is about 6% *slower*. So the shipped
policy is hybrid: sequences of 2 or more steps go to the kernel, single-token
decode stays on the loop. `REVIEWER_DELTA_MIN_STEPS=1` forces cuTile everywhere,
which is how I measured the third column.

**Same words out?** Not always. Across both passes, the output is identical to
the candle loop's on the four prompts of 513 tokens or fewer, and different on
the two longest (1,188 and 4,419 tokens). I think that's numerics: the loop does
its arithmetic in bf16, so rounding accumulates over thousands of steps, while
the kernel keeps its state in f32. That would also make the kernel the more
accurate of the two, but I haven't proved it. I didn't run a reference
implementation on those prompts, so I can say they differ and not which one is
right. With decode also on cuTile, four of twelve outputs differ from the loop,
which says the same thing from a different angle.

## What I'd distrust about this

- **First run lied.** The initial candle numbers had fused softmax at N=1024
  reading 478 GB/s, well above what the memory can deliver. An 8 MB tensor read
  repeatedly sits in L2. I changed `bench_ops` to cycle through 8 distinct
  input tensors, which brought it down to 285 GB/s, still a touch above the
  273 spec line. The N=1024 softmax point is the one I trust least.
- **Different harnesses.** cuTile's numbers are criterion medians. Candle's are
  the median of seven batches of 100 calls, synchronizing the device once per
  batch. They measure the same thing but they aren't the same code.
- **Synthetic shapes.** For RMSNorm and softmax, N goes up to 32768; the 27B's
  hidden size is 5120, and in training the row count is tokens-per-batch, not
  4096. Those tables tell you the shape of the curves, not the speedup you'll
  see in a training step. (The recurrence benchmark below does use the real
  head count and dimensions.)
- **One model.** The inference numbers are the 9B. The 27B in bf16 is 54 GB, which
  doesn't fit beside the gpt-oss server that holds most of the box's memory, and
  I wasn't going to stop it for a benchmark. The 27B has 48 value heads to the
  9B's 32 and 48 linear layers to its 24, so the recurrence should be a bigger
  share of its prefill, not a smaller one. That's a prediction, not a result.
- **Six prompts, one pass.** The warm numbers are a single second pass over six
  prompts. They were stable between the two passes (the first-pass prefills
  differ by a few percent plus the one-time compile), but six prompts is a small
  sample and I didn't compute error bars.
- **Speed only, for the first two ops.** I haven't checked that the RMSNorm or
  softmax kernels agree numerically with the PyTorch oracles.
  [The rule for Path A](path-a-stages-4-5-todo.md) is correctness before speed,
  and a kernel that hasn't been through a `verify-*` stage hasn't earned a place
  in the forward pass. The recurrence kernel below has been through one.

## What it still doesn't do

- **No backward pass.** Path A trains a LoRA adapter *through* the recurrence with
  candle's autograd, which differentiates the loop for free. The kernel is
  inference-only, and the training guard makes that loud. A backward kernel is
  the hard half of making this useful for training, and I haven't started it.
- **The bridge is heavy at one token.** A single decode step spends more time in
  candle casts and layout ops around the launch than in the kernel. A decode-
  specific path that skips the f32 round trip would probably let cuTile win
  there too. I didn't build it.
- **bf16 in, f32 inside.** The kernel computes in f32 and the bridge casts at the
  edges. The loop computes in whatever dtype the model uses.
- **The baseline is the weakest one.** The sequential loop is what the trainer
  runs, so it's the honest comparison for "what changes if I swap this in." It
  is not the best known way to compute a gated delta rule. Chunked parallel
  forms, the kind production DeltaNet kernels use, process many timesteps at once
  with matrix multiplies, and a cuTile kernel written that way would be a very
  different benchmark. The speedups above are over *my loop*, not over the state
  of the art.

## Where this leaves the plan

Three separate things came out of this.

The cheap one: swap the hand-rolled `rmsnorm` in `model.rs` for
`candle_nn::ops::rms_norm`, re-run the `verify-*` stages, and bank most of the
6× on that op. That's an afternoon and it doesn't need cuTile. The caveat is the
`(1 + weight)` scale Qwen uses: the fused op takes the weight directly, so the
weight needs the `+1` applied once up front instead of every call. I haven't
made this change.

The one that pays now: the recurrence kernel is real, it is verified against the
PyTorch dump, and it takes time to first token from 25.7 s to 8.9 s on a
4,400-token prompt. It stays behind a switch and a feature flag until it's been
through more than a 10-token model check, but for inference it works.

The one that matters next is the floor. Once the recurrence is gone, about 4.3
seconds of prefill is something else, and it doesn't depend on how long the
prompt is. Finding out what that is, with the same attribution trick that found
the recurrence's share, is worth more now than another kernel. After that, in
order: a decode path that doesn't pay for the bridge, the 27B once the box has the
memory, and, since training is what Path A is for, the backward kernel.

The benches are `crates/reviewer-train/src/bin/bench_ops.rs` and
`bench_delta.rs`, the standalone kernel crate is `experiments/cutile-deltanet/`,
and the bridge is `crates/reviewer-train/src/cutile_delta.rs`. To reproduce the
inference runs, on the box: build with `--features cutile` (after
`source ~/cutile-env.sh`), then run `reviewer-train bench --sequential-only
--jsonl <prompts> --weights <9B snapshot> --bf16 --adapter <adapter>` with and
without `REVIEWER_DELTA=cutile`; add `REVIEWER_DELTA_TIMING=1` for the
recurrence-only attribution. For the earlier numbers, `bench_ops` and
`bench_delta` with `--features cuda`, and `cargo bench -p cutile-benchmarks --
rmsnorm` (and `softmax`) in a cutile-rs checkout.
