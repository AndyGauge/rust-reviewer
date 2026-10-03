//! The Gated DeltaNet recurrence as a cuTile kernel, callable on candle CUDA tensors.
//! **Inference only** — there is no backward pass, so gradients do not flow through it.
//!
//! Semantics match `delta::recurrent_gated_delta_rule_loop` with `qk_l2norm = true`
//! (including `initial_state`, which is what makes a decode step "one more step").
//! One program per (batch, value-head) keeps the `[DK, DV]` f32 state in registers for
//! the whole timestep loop. Candle's buffers are borrowed by cuTile with no copy
//! (`Tensor::from_foreign`); candle allocates and owns the output buffer.
//!
//! Enabled at run time with `REVIEWER_DELTA=cutile` (and built with `--features cutile`).

use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Storage, Tensor};
use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::cuda_core::Stream;
use cutile::prelude::*;

#[cutile::module]
mod gdn {
    use cutile::core::*;

    /// `out` is `[B*H, R, DV]` f32 with `R >= DK + S`: rows `0..DK` receive the final
    /// state and rows `DK..DK+S` the per-step outputs (one `&mut` output is all the
    /// launcher allows). `gb` is `[B*H, 2, S]`: plane 0 is `g`, plane 1 is `beta`.
    /// The step count `S` is a *runtime* value (read from `q`'s shape): a compile-time
    /// trip count made the JIT unroll the loop, so compile time grew with `S`.
    #[cutile::entry()]
    fn gated_delta_rule<const DK: i32, const DV: i32, const R: i32>(
        out: &mut Tensor<f32, { [1, R, DV] }>,
        q: &Tensor<f32, { [-1, -1, DK] }>,
        k: &Tensor<f32, { [-1, -1, DK] }>,
        v: &Tensor<f32, { [-1, -1, DV] }>,
        gb: &Tensor<f32, { [-1, 2, -1] }>,
        h0: &Tensor<f32, { [-1, DK, DV] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b*h, 0, 0)

        let q_part = q.partition(shape![1, 1, DK]);
        let k_part = k.partition(shape![1, 1, DK]);
        let v_part = v.partition(shape![1, 1, DV]);
        let gb_part = gb.partition(shape![1, 1, 1]);
        let h0_part = h0.partition(shape![1, DK, DV]);
        let mut out_part = out.partition_mut(shape![1, 1, DV]);

        let eps: Tile<f32, { [1, 1] }> = constant(1e-6f32, shape![1, 1]);
        let dk_f: f32 = convert_scalar(DK);
        let dk_t: Tile<f32, { [1, 1] }> = dk_f.broadcast(shape![1, 1]);
        let q_scale: Tile<f32, { [1, 1] }> = rsqrt(dk_t, ftz::Disabled);

        let h0_t: Tile<f32, { [1, DK, DV] }> = h0_part.load([pid.0, 0i32, 0i32]);
        let mut state: Tile<f32, { [DK, DV] }> = h0_t.reshape(shape![DK, DV]);

        let steps: i32 = q.shape()[1];
        for t in 0i32..steps {
            let q_t: Tile<f32, { [1, 1, DK] }> = q_part.load([pid.0, t, 0i32]);
            let k_t: Tile<f32, { [1, 1, DK] }> = k_part.load([pid.0, t, 0i32]);
            let v_t: Tile<f32, { [1, 1, DV] }> = v_part.load([pid.0, t, 0i32]);
            let g_t: Tile<f32, { [1, 1, 1] }> = gb_part.load([pid.0, 0i32, t]);
            let b_t: Tile<f32, { [1, 1, 1] }> = gb_part.load([pid.0, 1i32, t]);

            let q_col: Tile<f32, { [DK, 1] }> = q_t.reshape(shape![DK, 1]);
            let k_col: Tile<f32, { [DK, 1] }> = k_t.reshape(shape![DK, 1]);
            let v_row: Tile<f32, { [1, DV] }> = v_t.reshape(shape![1, DV]);
            let g_s: Tile<f32, { [1, 1] }> = g_t.reshape(shape![1, 1]);
            let b_s: Tile<f32, { [1, 1] }> = b_t.reshape(shape![1, 1]);

            let q_ss: Tile<f32, { [1] }> = reduce_sum(q_col * q_col, 0i32);
            let k_ss: Tile<f32, { [1] }> = reduce_sum(k_col * k_col, 0i32);
            let q_inv: Tile<f32, { [1, 1] }> =
                rsqrt(q_ss.reshape(shape![1, 1]) + eps, ftz::Disabled);
            let k_inv: Tile<f32, { [1, 1] }> =
                rsqrt(k_ss.reshape(shape![1, 1]) + eps, ftz::Disabled);
            let q_n: Tile<f32, { [DK, DV] }> =
                (q_col * q_inv.broadcast(shape![DK, 1]) * q_scale.broadcast(shape![DK, 1]))
                    .broadcast(shape![DK, DV]);
            let k_n: Tile<f32, { [DK, DV] }> =
                (k_col * k_inv.broadcast(shape![DK, 1])).broadcast(shape![DK, DV]);

            state = state * exp(g_s).broadcast(shape![DK, DV]);
            let kv_mem: Tile<f32, { [DV] }> = reduce_sum(state * k_n, 0i32);
            let delta: Tile<f32, { [1, DV] }> =
                (v_row - kv_mem.reshape(shape![1, DV])) * b_s.broadcast(shape![1, DV]);
            state = state + k_n * delta.broadcast(shape![DK, DV]);
            let out_t: Tile<f32, { [DV] }> = reduce_sum(state * q_n, 0i32);
            out_part.store(out_t.reshape(shape![1, 1, DV]), [0i32, DK + t, 0i32]);
        }
        // Final state -> rows 0..DK, one row at a time (a second mutable view of `out`
        // is rejected by the borrow checker).
        let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
        for r in 0i32..DK {
            let ri: Tile<i32, { [] }> = scalar_to_tile(r);
            let row: Tile<f32, { [1, DV] }> = extract(state, [ri, zero]);
            out_part.store(row.reshape(shape![1, 1, DV]), [0i32, r, 0i32]);
        }
    }
}

/// `REVIEWER_DELTA=cutile` turns the cuTile path on (checked once).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("REVIEWER_DELTA").map(|v| v == "cutile").unwrap_or(false))
}

/// Shortest sequence routed to the kernel (`REVIEWER_DELTA_MIN_STEPS`, default 2).
/// Below it the candle loop runs: at S=1 (decode) there is nothing to fuse and the
/// bridge's own overhead outweighs the loop's twenty tiny launches. Set it to 1 to
/// force cuTile for decode as well.
pub fn min_steps() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("REVIEWER_DELTA_MIN_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(2)
    })
}

fn stream() -> Result<&'static Arc<Stream>> {
    static STREAM: OnceLock<std::result::Result<Arc<Stream>, String>> = OnceLock::new();
    STREAM
        .get_or_init(|| {
            Device::new(0)
                .and_then(|d| d.new_stream())
                .map_err(|e| format!("cutile device/stream init: {e:?}"))
        })
        .as_ref()
        .map_err(|e| candle_core::Error::Msg(e.clone()))
}

fn ce<E: std::fmt::Debug>(e: E) -> candle_core::Error {
    candle_core::Error::Msg(format!("cutile: {e:?}"))
}

/// A candle CUDA tensor's buffer, lent to cuTile. Holding the `Tensor` keeps the
/// allocation alive for as long as cuTile's view of it exists.
struct CandleBuf {
    _keep: Tensor,
    ptr: CUdeviceptr,
    len_bytes: usize,
}

// SAFETY: `ptr` is the live device allocation behind `_keep` (contiguous, f32), valid
// for `len_bytes` for as long as `_keep` is held, which is as long as this value.
unsafe impl DeviceAllocation for CandleBuf {
    fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    fn device_id(&self) -> usize {
        0 // single-GPU box
    }
}

/// Borrow a contiguous f32 candle CUDA tensor as a cuTile tensor (no copy).
///
/// SAFETY: the caller guarantees nothing else touches the buffer while the returned
/// tensor is in use (candle is synchronized before launch and idle until it finishes).
unsafe fn borrow(t: &Tensor) -> Result<Tensor_> {
    assert_eq!(t.dtype(), DType::F32);
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        candle_core::bail!("cutile_delta: tensor must be contiguous");
    }
    let Storage::Cuda(cs) = &*storage else { candle_core::bail!("cutile_delta: not a CUDA tensor") };
    let slice = cs.as_cuda_slice::<f32>()?;
    let (base, _guard) = slice.device_ptr(slice.stream());
    let ptr = base + (layout.start_offset() * 4) as CUdeviceptr;
    let dims: Vec<i32> = t.dims().iter().map(|&d| d as i32).collect();
    let mut strides = vec![1i32; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    let owner = Arc::new(CandleBuf { _keep: t.clone(), ptr, len_bytes: t.elem_count() * 4 });
    Ok(unsafe { Tensor_::from_foreign(owner, dims, strides) })
}
type Tensor_ = cutile::prelude::Tensor<f32>;

/// `[B,S,H,D]` (any float dtype) -> `[B*H, S, D]` f32 contiguous.
fn prep(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
    let (b, h, s, d) = x.dims4()?;
    x.reshape((b * h, s, d))
}

/// The output buffer's row count is a compile-time shape (`R`), so it is rounded up to
/// one of a few sizes: the kernel compiles once per size, not once per prompt length.
const CAPS: [usize; 6] = [1, 64, 256, 1024, 4096, 16384];

/// Same contract as `delta::recurrent_gated_delta_rule_loop(.., qk_l2norm = true, ..)`:
/// `q,k: [B,S,H,Dk]`, `v: [B,S,H,Dv]`, `g,beta: [B,S,H]` -> `([B,S,H,Dv], [B,H,Dk,Dv])`.
pub fn recurrent_gated_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let dev = q.device().clone();
    let in_dtype = q.dtype();
    let (b, s, h, dk) = q.dims4()?;
    let dv = v.dim(3)?;
    let cap = *CAPS.iter().find(|&&c| c >= s).unwrap_or(&s);
    let bh = b * h;

    let qf = prep(q)?;
    let kf = prep(k)?;
    let vf = prep(v)?;
    // [B,S,H] x2 -> [B*H, 2, S]
    let gf = g.to_dtype(DType::F32)?.transpose(1, 2)?; // [B,H,S]
    let bf = beta.to_dtype(DType::F32)?.transpose(1, 2)?;
    let gb = Tensor::stack(&[gf, bf], 2)?.contiguous()?.reshape((bh, 2, s))?;
    let h0 = match initial_state {
        Some(st) => st.to_dtype(DType::F32)?.contiguous()?.reshape((bh, dk, dv))?,
        None => Tensor::zeros((bh, dk, dv), DType::F32, &dev)?,
    };
    let rows = dk + cap;
    let out_buf = Tensor::zeros((bh, rows, dv), DType::F32, &dev)?;

    let stream = stream()?;
    dev.synchronize()?; // candle's work is done before cuTile reads/writes its buffers
    // SAFETY: see `borrow`; candle does nothing with these buffers until `sync_on` returns.
    let (out_t, q_t, k_t, v_t, gb_t, h0_t) = unsafe {
        (borrow(&out_buf)?, borrow(&qf)?, borrow(&kf)?, borrow(&vf)?, borrow(&gb)?, borrow(&h0)?)
    };
    let out_part = out_t.partition([1, rows, dv]);
    gdn::gated_delta_rule(out_part, Arc::new(q_t), Arc::new(k_t), Arc::new(v_t), Arc::new(gb_t), Arc::new(h0_t))
        .generics(vec![dk.to_string(), dv.to_string(), rows.to_string()])
        .sync_on(stream)
        .map_err(ce)?;

    let state = out_buf.narrow(1, 0, dk)?.reshape((b, h, dk, dv))?.to_dtype(in_dtype)?;
    let out = out_buf
        .narrow(1, dk, s)?
        .reshape((b, h, s, dv))?
        .transpose(1, 2)?
        .contiguous()?
        .to_dtype(in_dtype)?;
    Ok((out, state))
}
