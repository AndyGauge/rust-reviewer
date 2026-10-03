//! Gated DeltaNet recurrence as a cuTile kernel, checked against the same
//! `train/delta_synth.safetensors` oracle that `reviewer-train verify-delta` uses.
//!
//!   source ~/cutile-env.sh
//!   cargo run --release -- verify --oracle ~/rust-train/train/delta_synth.safetensors
//!   cargo run --release -- random            # bigger shapes vs a plain-Rust CPU reference
//!   cargo run --release -- bench             # Qwen3.6-27B shapes, timings as JSON lines

use anyhow::{Context, Result, bail};
use cutile::prelude::*;
use std::sync::Arc;
use std::time::Instant;

#[cutile::module]
mod gdn {
    use cutile::core::*;

    /// One program per (batch, value-head). The whole `[DK, DV]` state lives in
    /// registers and the timestep loop runs inside the kernel, so q/k/v are read
    /// once and the state never touches global memory.
    ///
    /// Per step (matches `reviewer-train/src/delta.rs`):
    ///   q,k <- l2norm; q *= 1/sqrt(DK)
    ///   S *= exp(g);  kv = S^T k;  d = (v - kv) * beta;  S += k (x) d;  out = S^T q
    #[cutile::entry()]
    fn gated_delta_rule<const S: i32, const DK: i32, const DV: i32>(
        out: &mut Tensor<f32, { [1, S, DV] }>, // [B*H, S, DV]
        q: &Tensor<f32, { [-1, -1, DK] }>,    // [B*H, S, DK]
        k: &Tensor<f32, { [-1, -1, DK] }>,
        v: &Tensor<f32, { [-1, -1, DV] }>,
        g: &Tensor<f32, { [-1, -1, 1] }>, // [B*H, S, 1]
        beta: &Tensor<f32, { [-1, -1, 1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id(); // (b*h, 0, 0)

        let q_part = q.partition(shape![1, 1, DK]);
        let k_part = k.partition(shape![1, 1, DK]);
        let v_part = v.partition(shape![1, 1, DV]);
        let g_part = g.partition(shape![1, 1, 1]);
        let beta_part = beta.partition(shape![1, 1, 1]);
        let mut out_part = out.partition_mut(shape![1, 1, DV]);

        let eps: Tile<f32, { [1, 1] }> = constant(1e-6f32, shape![1, 1]);
        let dk_f: f32 = convert_scalar(DK);
        let dk_t: Tile<f32, { [1, 1] }> = dk_f.broadcast(shape![1, 1]);
        let q_scale: Tile<f32, { [1, 1] }> = rsqrt(dk_t, ftz::Disabled);
        let mut state: Tile<f32, { [DK, DV] }> = constant(0.0f32, shape![DK, DV]);

        for t in 0i32..S {
            let q_t: Tile<f32, { [1, 1, DK] }> = q_part.load([pid.0, t, 0i32]);
            let k_t: Tile<f32, { [1, 1, DK] }> = k_part.load([pid.0, t, 0i32]);
            let v_t: Tile<f32, { [1, 1, DV] }> = v_part.load([pid.0, t, 0i32]);
            let g_t: Tile<f32, { [1, 1, 1] }> = g_part.load([pid.0, t, 0i32]);
            let b_t: Tile<f32, { [1, 1, 1] }> = beta_part.load([pid.0, t, 0i32]);

            let q_col: Tile<f32, { [DK, 1] }> = q_t.reshape(shape![DK, 1]);
            let k_col: Tile<f32, { [DK, 1] }> = k_t.reshape(shape![DK, 1]);
            let v_row: Tile<f32, { [1, DV] }> = v_t.reshape(shape![1, DV]);
            let g_s: Tile<f32, { [1, 1] }> = g_t.reshape(shape![1, 1]);
            let b_s: Tile<f32, { [1, 1] }> = b_t.reshape(shape![1, 1]);

            // L2-normalize q and k over the key dim, then scale q.
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

            // 1. decay
            state = state * exp(g_s).broadcast(shape![DK, DV]);
            // 2. read with key: kv_mem[v] = sum_k S[k, v] * k[k]
            let kv_mem: Tile<f32, { [DV] }> = reduce_sum(state * k_n, 0i32);
            // 3. delta
            let delta: Tile<f32, { [1, DV] }> =
                (v_row - kv_mem.reshape(shape![1, DV])) * b_s.broadcast(shape![1, DV]);
            // 4. outer-product update
            state = state + k_n * delta.broadcast(shape![DK, DV]);
            // 5. read with query
            let out_t: Tile<f32, { [DV] }> = reduce_sum(state * q_n, 0i32);
            out_part.store(out_t.reshape(shape![1, 1, DV]), [0i32, t, 0i32]);
        }
    }
}

/// `[B,S,H,D]` -> `[B,H,S,D]` on the host.
fn bshd_to_bhsd(x: &[f32], b: usize, s: usize, h: usize, d: usize) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                let src = ((bi * s + si) * h + hi) * d;
                let dst = ((bi * h + hi) * s + si) * d;
                o[dst..dst + d].copy_from_slice(&x[src..src + d]);
            }
        }
    }
    o
}

struct Case {
    b: usize,
    s: usize,
    h: usize,
    dk: usize,
    dv: usize,
    // all `[B,H,S,*]`
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

/// Plain-Rust CPU reference of the recurrence, `[B,H,S,DV]` out. Same math as
/// `delta.rs` / transformers' `torch_recurrent_gated_delta_rule`.
fn cpu_reference(c: &Case) -> Vec<f32> {
    let (b, s, h, dk, dv) = (c.b, c.s, c.h, c.dk, c.dv);
    let mut out = vec![0f32; b * h * s * dv];
    let scale = 1.0 / (dk as f32).sqrt();
    for bh in 0..b * h {
        let mut st = vec![0f32; dk * dv];
        for t in 0..s {
            let qi = (bh * s + t) * dk;
            let vi = (bh * s + t) * dv;
            let l2 = |x: &[f32]| {
                let n = (x.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
                x.iter().map(|a| a / n).collect::<Vec<_>>()
            };
            let q: Vec<f32> = l2(&c.q[qi..qi + dk]).iter().map(|a| a * scale).collect();
            let k = l2(&c.k[qi..qi + dk]);
            let decay = c.g[bh * s + t].exp();
            st.iter_mut().for_each(|a| *a *= decay);
            let mut kv = vec![0f32; dv];
            for i in 0..dk {
                for j in 0..dv {
                    kv[j] += st[i * dv + j] * k[i];
                }
            }
            let delta: Vec<f32> =
                (0..dv).map(|j| (c.v[vi + j] - kv[j]) * c.beta[bh * s + t]).collect();
            for i in 0..dk {
                for j in 0..dv {
                    st[i * dv + j] += k[i] * delta[j];
                }
            }
            for j in 0..dv {
                out[vi + j] = (0..dk).map(|i| st[i * dv + j] * q[i]).sum();
            }
        }
    }
    out
}

/// Device-resident inputs, so the benchmark can time the kernel without the host copies.
struct DevInputs {
    q: Arc<Tensor<f32>>,
    k: Arc<Tensor<f32>>,
    v: Arc<Tensor<f32>>,
    g: Arc<Tensor<f32>>,
    beta: Arc<Tensor<f32>>,
}

fn upload_case(stream: &Arc<cutile::cuda_core::Stream>, c: &Case) -> Result<DevInputs> {
    let upload = |v: &Vec<f32>, shape: &[usize]| -> Result<Arc<Tensor<f32>>> {
        Ok(api::copy_host_vec_to_device(&Arc::new(v.clone()))
            .sync_on(stream)?
            .reshape(shape)
            .map_err(|e| anyhow::anyhow!("{e:?}"))?
            .into())
    };
    let (bh, s) = (c.b * c.h, c.s);
    Ok(DevInputs {
        q: upload(&c.q, &[bh, s, c.dk])?,
        k: upload(&c.k, &[bh, s, c.dk])?,
        v: upload(&c.v, &[bh, s, c.dv])?,
        g: upload(&c.g, &[bh, s, 1])?,
        beta: upload(&c.beta, &[bh, s, 1])?,
    })
}

/// Launch the kernel and return the `[B*H, S, DV]` output tensor (device-resident).
fn launch(stream: &Arc<cutile::cuda_core::Stream>, c: &Case, d: &DevInputs) -> Result<Tensor<f32>> {
    let (bh, s) = (c.b * c.h, c.s);
    let out: Partition<Tensor<f32>> =
        api::zeros(&[bh, s, c.dv]).sync_on(stream)?.partition([1, s, c.dv]);
    let res = gdn::gated_delta_rule(
        out, d.q.clone(), d.k.clone(), d.v.clone(), d.g.clone(), d.beta.clone(),
    )
    .generics(vec![s.to_string(), c.dk.to_string(), c.dv.to_string()])
    .first()
    .unpartition()
    .sync_on(stream)?;
    Ok(res)
}

fn run_gpu(stream: &Arc<cutile::cuda_core::Stream>, c: &Case) -> Result<Vec<f32>> {
    let d = upload_case(stream, c)?;
    Ok(launch(stream, c, &d)?.to_host_vec().sync_on(stream)?)
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| (x - y).abs()).collect();
    (d.iter().cloned().fold(0., f32::max), d.iter().sum::<f32>() / d.len() as f32)
}

/// xorshift, so the check needs no rand dependency.
struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        // Box-Muller
        let (u1, u2) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn random_case(b: usize, s: usize, h: usize, dk: usize, dv: usize, seed: u64) -> Case {
    let mut r = Rng(seed | 1);
    let n = |r: &mut Rng, len| (0..len).map(|_| r.normal()).collect::<Vec<f32>>();
    Case {
        b, s, h, dk, dv,
        q: n(&mut r, b * h * s * dk),
        k: n(&mut r, b * h * s * dk),
        v: n(&mut r, b * h * s * dv),
        g: (0..b * h * s).map(|_| -r.unit()).collect(), // exp(g) in (0,1]
        beta: (0..b * h * s).map(|_| r.unit()).collect(),
    }
}

fn verify(oracle: &str) -> Result<()> {
    let bytes = std::fs::read(oracle).with_context(|| format!("reading {oracle}"))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)?;
    let get = |name: &str| -> Result<(Vec<f32>, Vec<usize>)> {
        let t = st.tensor(name)?;
        let data = t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok((data, t.shape().to_vec()))
    };
    let (q, qs) = get("q")?; // [B,S,H,Dk]
    let (v, vs) = get("v")?;
    let (k, _) = get("k")?;
    let (g, _) = get("g")?; // [B,S,H]
    let (beta, _) = get("beta")?;
    let (expect, _) = get("out")?; // [B,S,H,Dv]
    let (b, s, h, dk, dv) = (qs[0], qs[1], qs[2], qs[3], vs[3]);
    let case = Case {
        b, s, h, dk, dv,
        q: bshd_to_bhsd(&q, b, s, h, dk),
        k: bshd_to_bhsd(&k, b, s, h, dk),
        v: bshd_to_bhsd(&v, b, s, h, dv),
        g: bshd_to_bhsd(&g, b, s, h, 1),
        beta: bshd_to_bhsd(&beta, b, s, h, 1),
    };
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let got_bhsd = run_gpu(&stream, &case)?;
    // back to [B,S,H,Dv] to compare with the oracle's layout
    let mut got = vec![0f32; got_bhsd.len()];
    for bi in 0..b { for hi in 0..h { for si in 0..s {
        let src = ((bi * h + hi) * s + si) * dv;
        let dst = ((bi * s + si) * h + hi) * dv;
        got[dst..dst + dv].copy_from_slice(&got_bhsd[src..src + dv]);
    }}}
    let (max, mean) = diff(&got, &expect);
    println!("cuTile gated delta recurrence vs transformers oracle ({b}x{s}x{h}x{dk}x{dv}):");
    println!("  max_abs_diff  = {max:.3e}");
    println!("  mean_abs_diff = {mean:.3e}");
    println!("  {}", if max < 1e-4 { "MATCH ✓" } else { "MISMATCH ✗" });
    if max >= 1e-4 { bail!("mismatch vs oracle"); }
    Ok(())
}

fn random() -> Result<()> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let mut ok = true;
    for &(b, s, h, dk, dv) in &[(1, 5, 2, 4, 4), (2, 33, 3, 16, 16), (1, 64, 4, 64, 64), (1, 200, 48, 128, 128)] {
        let case = random_case(b, s, h, dk, dv, 0x9E3779B97F4A7C15 ^ s as u64);
        let want = cpu_reference(&case);
        let got = run_gpu(&stream, &case)?;
        let (max, mean) = diff(&got, &want);
        let pass = max < 1e-3;
        ok &= pass;
        println!("B={b} S={s} H={h} Dk={dk} Dv={dv}: max_abs_diff={max:.3e} mean={mean:.3e} {}",
                 if pass { "MATCH ✓" } else { "MISMATCH ✗" });
    }
    if !ok { bail!("mismatch vs CPU reference"); }
    Ok(())
}

fn bench() -> Result<()> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    // Qwen3.6-27B: 48 value heads, Dk = Dv = 128. Batch 1. Kernel only: inputs are
    // uploaded once and the output stays on the device.
    for &s in &[64usize, 256, 1024, 2048] {
        let case = random_case(1, s, 48, 128, 128, 7);
        let dev = upload_case(&stream, &case)?;
        launch(&stream, &case, &dev)?; // JIT + warmup
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = Instant::now();
            for _ in 0..5 { launch(&stream, &case, &dev)?; }
            times.push(t.elapsed().as_secs_f64() * 1e6 / 5.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(r#"{{"impl":"cutile","s":{s},"us":{:.1}}}"#, times[3]);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("verify") => verify(args.get(3).context("--oracle <path>")?),
        Some("random") => random(),
        Some("bench") => bench(),
        _ => bail!("usage: cutile-deltanet verify --oracle <safetensors> | random | bench"),
    }
}
