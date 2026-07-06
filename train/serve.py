#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for the reviewer LoRA, on the transformers
stack (the known-good path — the same stack that trained the model, so no
dependency on whether vLLM supports this arch on sm_121 yet).

Serves `POST /v1/chat/completions` so `reviewer-run --endpoint` can drive it
unchanged. Loads a base model (+ optional LoRA adapter), auto-detecting the
loader class the way sft.py does. Not fast — transformers generate, no paged
attention — but reliable, and plenty for evaluating a handful of PRs.

    pip install fastapi uvicorn
    python serve.py --base Qwen/Qwen3.6-27B \
        --adapter out/keep/checkpoint-1000-epoch1 --port 8000

Then, from anywhere on the LAN:
    reviewer-run review --repo rust-lang/rust --pr 158822 \
        --endpoint http://<box-ip>:8000/v1 --model reviewer \
        --model-version reviewer-27b@epoch1
"""
import argparse
import time

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)


def is_multimodal(model: str) -> bool:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    archs = " ".join(getattr(cfg, "architectures", None) or [])
    if any(t in archs for t in ("ConditionalGeneration", "ImageText", "VL", "Vision")):
        return True
    return any(hasattr(cfg, a) for a in ("vision_config", "visual_config", "vision_tower"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        help="LoRA adapter as name=path (repeatable). The request's `model` field "
        "selects which adapter to activate — base loaded ONCE, no reload to switch. "
        "A bare path (no name=) is loaded as 'default'.",
    )
    ap.add_argument("--model-name", default="reviewer", help="name reported to the client")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    return ap.parse_args()


ARGS = parse_args()

print(f"loading tokenizer for {ARGS.base} ...")
TOK = AutoTokenizer.from_pretrained(ARGS.base, trust_remote_code=True)

mm = is_multimodal(ARGS.base)
loader = AutoModelForImageTextToText if mm else AutoModelForCausalLM
print(f"loading base ({'image-text' if mm else 'causal'}, bf16, ~min) ...")
MODEL = loader.from_pretrained(
    ARGS.base, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa"
)

# Load every adapter onto the ONE base and switch with set_adapter() per request
# — this is the actual point of LoRA (frozen base shared, ~1 GB deltas swapped),
# so comparing epoch-1 vs epoch-3 costs a pointer flip, not a 54 GB reload.
ADAPTERS = []  # names, in load order; ADAPTERS[0] is the default
for spec in ARGS.adapter:
    name, _, path = spec.partition("=")
    if not path:  # bare path -> "default"
        name, path = "default", name
    from peft import PeftModel

    if not ADAPTERS:
        print(f"attaching adapter {name} <- {path} ...")
        MODEL = PeftModel.from_pretrained(MODEL, path, adapter_name=name)
    else:
        print(f"attaching adapter {name} <- {path} ...")
        MODEL.load_adapter(path, adapter_name=name)
    ADAPTERS.append(name)
if ADAPTERS:
    MODEL.set_adapter(ADAPTERS[0])
    print(f"adapters: {ADAPTERS} (default: {ADAPTERS[0]})")
MODEL.eval()
if TOK.pad_token_id is None:
    TOK.pad_token = TOK.eos_token
print("ready.")

app = FastAPI()


class Msg(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str | None = None
    messages: list[Msg]
    temperature: float | None = None
    max_tokens: int | None = None


@app.get("/v1/models")
def models():
    ids = ADAPTERS or [ARGS.model_name]
    return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}


def build_inputs(messages):
    # enable_thinking=False pre-fills an empty <think></think> so the model emits
    # the review comment directly, matching our no-CoT training targets. Without
    # it, this reasoning-capable base spends the whole token budget "thinking" and
    # never reaches the answer. Fall back if a tokenizer lacks the kwarg.
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        return TOK.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        return TOK.apply_chat_template(messages, **kw)


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    # Select the adapter by the request's `model` field (no reload — set_adapter
    # is a pointer flip). Single-stream only: set_adapter mutates shared state, so
    # concurrent requests for different adapters would race — fine here, the
    # harness drives this server sequentially.
    if ADAPTERS and req.model in ADAPTERS:
        MODEL.set_adapter(req.model)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    inputs = build_inputs(messages)
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    temp = req.temperature if req.temperature is not None else ARGS.temperature
    gen_kwargs = dict(
        max_new_tokens=req.max_tokens or ARGS.max_new_tokens,
        pad_token_id=TOK.pad_token_id,
    )
    if temp and temp > 0:
        gen_kwargs.update(do_sample=True, temperature=temp)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = MODEL.generate(**inputs, **gen_kwargs)
    text = TOK.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

    return {
        "id": f"chatcmpl-{int(time.time()*1000)}",
        "object": "chat.completion",
        "model": req.model or ARGS.model_name,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": out.shape[1] - prompt_len},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")
