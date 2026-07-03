# DGX (GB10) Rust training setup

Bring-up plan for training the design-review LoRA on the **ASUS Ascent GX10**
(NVIDIA GB10 Grace Blackwell, 128 GB unified memory, ~273 GB/s, ARM64). All-Rust
training path, with a documented Python escape hatch.

> The box ships with **DGX OS** (Ubuntu 24.04 on aarch64) preloaded — NVIDIA
> driver, CUDA, and container stack already installed. You are not starting from
> bare metal; you are starting from "drivers work, now build your stack."

See also: [training-plan.md](training-plan.md) (time/size estimates),
[capability-matrix.md](capability-matrix.md) (Rust ML gaps).

---

## Actual bring-up results (ASUS GX10, 2026-07-02)

What the real box looked like, and the frontier-hardware gotchas that bit:

- **OS:** Ubuntu 24.04.4 LTS, aarch64, kernel `6.17.0-nvidia`. 20 cores, 121 GiB
  unified RAM, 916 GB NVMe (822 GB free).
- **GPU:** NVIDIA GB10, driver 580.159.03, **CUDA 13.0**. `compute_cap` reports
  **12.1 (sm_121)** — as expected. `nvidia-smi` shows memory as `N/A` (unified
  memory has no discrete VRAM figure) — normal, not a fault.
- **Gotcha 1 — CUDA not on PATH.** The toolkit (nvcc 13.0.88) was installed at
  `/usr/local/cuda` but not on `PATH`. Fix: append `/usr/local/cuda/bin` to PATH
  and `/usr/local/cuda/lib64` to `LD_LIBRARY_PATH` in `~/.bashrc`.
- **Gotcha 2 — PyTorch wheel.** The **official** `cu130` index *does* now ship an
  aarch64 wheel: `torch-2.12.1+cu130 ...manylinux_2_28_aarch64`. No NGC container,
  community wheel, or source build needed (contra older 2026 guides).
  Install: `pip install torch --index-url https://download.pytorch.org/whl/cu130`.
- **Gotcha 3 — sm_121 not in the arch list, but kernels run anyway.**
  `torch.cuda.get_arch_list()` shows `sm_120` (not `sm_121`), yet a real bf16
  matmul on the GPU **succeeds** (sm_121 is forward-compatible with sm_120 / PTX
  JIT). Lesson: test with an actual GPU op, not the advertised arch list.
- **Model arch recognized natively.** transformers 5.12.1 loads
  `Qwen/Qwen3.6-27B` as `model_type=qwen3_5`,
  class `Qwen3_5ForConditionalGeneration` (multimodal) — **no `trust_remote_code`**.
  Load with `AutoModelForImageTextToText`, not `AutoModelForCausalLM`.

Stack that ended up working: rustc 1.96.1 · torch 2.12.1+cu130 · transformers
5.12.1 · trl 1.7.0 · peft 0.19.1 · datasets 5.0.0 · accelerate 1.14.0.

---

## Phase 0 — Verify the box (≈30 min)

ARM64 + Blackwell is new; confirm the stack before building on it.

```sh
nvidia-smi          # driver alive? GB10 visible? ~128 GB unified shown?
nvcc --version      # CUDA toolkit: need 12.8+ / 13 for Blackwell
uname -m            # aarch64
nproc; free -h      # Grace CPU cores + memory
df -h               # NVMe free space (need ~150 GB for base + merged + data)
```

**Critical gotcha:** GB10 compute capability is **sm_121**. Anything that
compiles CUDA kernels (candle included) must target it:

```sh
export CUDA_COMPUTE_CAP=121
```

If the driver/toolkit is too old for sm_121, fix that first — it is the single
most likely blocker.

---

## Phase 0.5 — OS, headless mode, and SSH access

DGX OS is **Ubuntu 24.04 LTS** based and ships preinstalled with a **GNOME
desktop** (the GX10 is a personal dev machine). You do **not** pick Server vs
Desktop at setup, and you should **not** reinstall plain Ubuntu Server — you'd
throw away NVIDIA's preconfigured Blackwell driver + CUDA stack and have to
rebuild it on brand-new ARM64/sm_121 hardware. Keep DGX OS; just run it headless.

### Why headless matters here (unified memory)

The GB10 has **unified memory**: CPU and GPU share one 128 GB pool. A running
GNOME session (compositor, browser, etc.) consumes directly from that pool — i.e.
out of your training budget. On a discrete-GPU box the desktop lives in system
RAM and never touches VRAM; on GB10 it is the same memory. So for serious runs,
don't start the graphical session.

```sh
# Boot to console (no GUI) — reclaims the desktop's memory for the model
sudo systemctl set-default multi-user.target
# Switch back to desktop later if wanted:
#   sudo systemctl set-default graphical.target
# Stop the GUI for the current session without rebooting:
#   sudo systemctl isolate multi-user.target
```

### SSH server setup

You'll drive the box from your laptop. DGX OS usually has `openssh-server`, but
enable + harden it explicitly.

```sh
# On the DGX (first time, at the physical console or initial desktop):
sudo apt update && sudo apt install -y openssh-server
sudo systemctl enable --now ssh
sudo systemctl status ssh          # confirm active (running)
ip -4 addr show | grep inet        # note the box's LAN IP
```

Key-based auth (do this instead of passwords):

```sh
# On your LAPTOP: create a key if needed, then copy it to the box
ssh-keygen -t ed25519 -C "dgx"     # if you don't already have one
ssh-copy-id <user>@<dgx-ip>        # installs your pubkey on the DGX
ssh <user>@<dgx-ip>                # confirm key login works
```

Harden the daemon (after confirming key login works):

```sh
# On the DGX: /etc/ssh/sshd_config.d/10-hardening.conf
PasswordAuthentication no
PermitRootLogin no
# then:
sudo systemctl restart ssh
```

### Make long training runs survive disconnects

SSH sessions drop; training runs are hours. Use a multiplexer so a dropped
connection doesn't kill the run:

```sh
sudo apt install -y tmux
tmux new -s train        # start a session, launch training inside it
# detach: Ctrl-b then d   |   reattach later: tmux attach -t train
```

### Quality-of-life

```sh
# On your LAPTOP ~/.ssh/config — alias + keepalive so idle sessions don't drop
Host dgx
    HostName <dgx-ip>
    User <user>
    ServerAliveInterval 60
    ServerAliveCountMax 10
# then just: ssh dgx
```

For monitoring during a run: `nvidia-smi -l 5` (refresh every 5s), or
`watch -n5 nvidia-smi`, in a second tmux pane.

---

## Phase 1 — Prove inference first (the cheap gate)

Before any training, prove you can *run* `Qwen3-Coder-30B-A3B` in Rust. This
validates the entire CUDA/Blackwell/ARM path at near-zero risk. If it fails, you
have found your Blackwell problem cheaply instead of mid-training.

```sh
curl https://sh.rustup.rs -sSf | sh && source "$HOME/.cargo/env"

# Option A: mistral.rs (most batteries-included; supports LoRA adapters + quant)
# Option B: candle-transformers example (lower-level)
# Pull the model and generate one prompt's worth of tokens.
```

Gate: if a Rust engine loads the model and generates text, the GPU stack is good.

---

## Phase 2 — Choose the training path

| | Rust-native (candle/burn + candle-lora) | Python escape hatch (Unsloth/Axolotl) |
|---|---|---|
| Blog story | The whole point | Breaks the no-Python rule |
| Blackwell/ARM risk | Real — possible missing/immature kernels | Lower, also catching up |
| What you build | An `SFTTrainer`-equivalent (crate opportunity) | A config file |

**Plan:** attempt Rust-native, time-boxed, validated on a toy model first
(Phase 3). Keep Python documented as a fallback so a single kernel gap can't sink
the project — but you may never need it.

---

## Phase 3 — De-risk the trainer on Qwen3-0.6B BEFORE the 30B

Do **not** debug a training loop at 30B scale. Make the full loop work on a tiny
model + a 500-example slice, then scaling up is config + patience, not debugging.

Loop to prove end-to-end:

1. Model load — safetensors via `hf-hub`
2. Tokenizer + chat-template application — `tokenizers` + `minijinja`
3. LoRA injection — `candle-lora`
4. Forward → loss with **prompt-token masking** (train only on the assistant turn)
5. Backward + **AdamW** (candle built-in) + checkpoint write
6. Eval loss on the held-out cookbook slice (`data/prepared/cookbook-0.4.jsonl`)

Gate: when this converges on 0.6B, the trainer is proven.

---

## Phase 4 — Train for real, then serve

- Run `Qwen3-Coder-30B-A3B`, **bf16** LoRA, rank 16–32, 2–3 epochs, watching eval
  loss (stop on uptick). No quantization needed — 128 GB fits bf16 (~70 GB
  resident). See [training-plan.md](training-plan.md).
- Output: a ~200 MB adapter.
- Serve it back through the Phase-1 inference path (base + adapter), point it at a
  fresh PR diff, and read what it says.

---

## Rust crate shopping list

```toml
# inference + training core
candle-core         # cuda feature; build with CUDA_COMPUTE_CAP=121
candle-nn
candle-transformers
candle-lora         # LoRA layers
# data + model plumbing
tokenizers          # HF tokenizer + chat template
safetensors
hf-hub              # model/dataset download
minijinja           # render tokenizer_config.json chat template
# inference engine (alt to candle for serving)
# mistralrs
```

Plus your own thin `reviewer-train` crate: prompt masking, sequence packing, eval
loop, checkpointing. That trainer is the piece that does not exist ergonomically
in Rust yet — i.e. the crate worth publishing.

---

## Build/run env summary

```sh
export CUDA_COMPUTE_CAP=121          # GB10 / Blackwell
export HF_HOME=/data/hf              # large model cache on NVMe, not home
# candle picks up CUDA from the toolkit; ensure nvcc is on PATH
cargo build --release -p reviewer-train --features cuda
```

---

## Gates recap (each is go/no-go, not a formality)

| Phase | Gate | If it fails |
|---|---|---|
| 0 | `nvidia-smi` + sm_121 toolkit | update driver/CUDA |
| 1 | Rust inference generates tokens | Blackwell kernel issue — investigate before training |
| 3 | 0.6B trainer converges | fix the loop cheaply at small scale |
| 4 | 30B eval loss drops, no overfit | tune LR/epochs/rank |

Either outcome at a gate is a good blog post: "it just worked" or "here is
exactly which kernel was missing."
