#!/usr/bin/env python3
"""One-step smoke test before the full run.

Proves the whole training path works on the GB10: model loads as
image-text-to-text, LoRA attaches, chat template applies, and a single
forward+backward produces a finite loss within the memory budget. Cheap
(~minutes) insurance against discovering a problem 4 hours into training.
"""
import json
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "Qwen/Qwen3.6-27B"
DATA = "../data/prepared/rust-0.4.jsonl"
N = 8

t0 = time.time()
print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL)

print("loading model (bf16, ~54GB, takes a few min)...")
# device_map={"": 0} forces the whole model onto GPU 0. Do NOT use "auto" here:
# on GB10 unified memory accelerate misreads GPU capacity (nvidia-smi shows mem
# as N/A) and offloads layers to the meta/CPU device, which breaks backward.
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
)
print(f"  loaded in {time.time()-t0:.0f}s; dtype={model.dtype}")

# LoRA. all-linear is the safe default; will narrow to language-only later.
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules="all-linear", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
model.train()
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# Build one batch of N examples via the chat template.
rows = []
with open(DATA) as f:
    for _ in range(N):
        rows.append(json.loads(f.readline())["messages"])
texts = [tok.apply_chat_template(m, tokenize=False) for m in rows]
enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
dev = next(model.parameters()).device
input_ids = enc["input_ids"].to(dev)
attn = enc["attention_mask"].to(dev)
labels = input_ids.clone()
labels[attn == 0] = -100  # ignore padding in loss
print(f"batch: {tuple(input_ids.shape)} (batch x seq)")

print("forward + backward...")
out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
loss = out.loss
loss.backward()

# Did any LoRA grad actually flow?
gnorm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
           if p.requires_grad and p.grad is not None) ** 0.5

peak = torch.cuda.max_memory_allocated() / 1e9
print("\n=== SMOKE TEST RESULTS ===")
print(f"loss           : {loss.item():.4f}  ({'FINITE ok' if torch.isfinite(loss) else 'NON-FINITE BAD'})")
print(f"grad norm      : {gnorm:.4f}  ({'grads flow ok' if gnorm > 0 else 'NO GRADS BAD'})")
print(f"peak GPU mem   : {peak:.1f} GB / 121 GiB")
print(f"total time     : {time.time()-t0:.0f}s")
print("PASS" if (torch.isfinite(loss) and gnorm > 0) else "FAIL")
