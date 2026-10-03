# cutile-deltanet

The Gated DeltaNet recurrence (`reviewer-train/src/delta.rs`) as a single
[cuTile Rust](https://github.com/NVlabs/cutile-rs) kernel. One program per
(batch, value-head); the `[Dk, Dv]` f32 state lives in registers and the
timestep loop runs inside the kernel, with the q/k L2-norm fused in.

Not a workspace member: it needs CUDA 13.2+ and only builds on the GB10 box
(`source ~/cutile-env.sh` there; see Part 21 of the book for the toolchain setup).

```sh
# correctness: same oracle `reviewer-train verify-delta` uses
cargo run --release -- verify --oracle ~/rust-train/train/delta_synth.safetensors
# larger shapes (up to 48 heads, Dk=Dv=128) vs a plain-Rust CPU reference
cargo run --release -- random
# kernel-only timings at Qwen3.6-27B shapes (B=1, H=48, Dk=Dv=128, f32)
cargo run --release -- bench
# candle baseline, same shapes (the real delta.rs, on the GPU)
CUDA_COMPUTE_CAP=121 cargo run --release -p reviewer-train --features cuda --bin bench_delta
```

## Status

This crate is the standalone, benchmark-only version (random inputs, f32, `S` fixed per compile).
The inference-capable kernel (initial/final state, runtime `S`) lives in
`crates/reviewer-train/src/cutile_delta.rs` behind `--features cutile` and
`REVIEWER_DELTA=cutile`; see Part 21.

The notes below describe this standalone crate.

Forward only, f32 only, zero initial state, no final-state output, and `S` is a
const generic (the kernel is JIT-compiled once per sequence length). So it can
verify and benchmark the recurrence, but it can't yet replace `delta.rs` in
prefill-with-cache, decode, or training (no backward).

Results on the GB10, B=1, H=48, Dk=Dv=128, f32 (kernel only for cuTile; the
candle number is the full `recurrent_gated_delta_rule` call including its
transposes, which is launch-bound):

| S | cuTile | candle | speedup |
|---|---|---|---|
| 64 | 0.45 ms | 15.2 ms | 34x |
| 256 | 1.76 ms | 61.2 ms | 35x |
| 1024 | 7.05 ms | 251 ms | 36x |
| 2048 | 14.1 ms | 502 ms | 36x |

Correctness: max abs diff 4.5e-8 against the transformers oracle
(`delta_synth.safetensors`, 1x5x2x4x4), and <= 1.2e-7 against the CPU reference
on four shapes up to 1x200x48x128x128.
