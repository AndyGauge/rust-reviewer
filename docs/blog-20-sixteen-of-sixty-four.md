# Sixteen of sixty-four

*Part 20: a routine task — get a second machine talking to the GB10 box — turns
into an audit of the vLLM server that's been quietly underselling itself since
[Part 16](blog-16-the-fast-path.md): a 4096-token context on a model that natively
does 262k, and a tool-calling flag that's necessary but not sufficient.*

This one didn't start as a model post. It started as plumbing: get a laptop
talking to the GB10 box over the LAN, point [Zed](https://zed.dev)'s remote
development at it instead of typing an IP address every time. SSH config, mDNS
hostname instead of a DHCP IP that can drift, `ssh_connections` block in Zed's
settings. Boring, correct, done in ten minutes.

Then, checking whether the laptop even needed SSH for the part that actually
mattered — talking to the model — the answer was no. The `vllm serve` command
[from Part 16](blog-16-the-fast-path.md) binds `0.0.0.0:8001` with no
`--api-key`. It was never behind SSH to begin with; anything on the LAN can hit
it in plaintext HTTP already. That's a deliberate trade for a home network: an
open front door is fine when you trust everyone who can reach it, and the
day this box sits behind a router port-forward is the day that flag goes back
on. Worth naming, not worth fixing today.

What *was* worth fixing was sitting in that same `vllm serve` line, unchanged
since Part 16: `--max-model-len 4096`.

## The context limit that was never real

4096 tokens was never a deliberate ceiling — it was whatever got the server
running fastest while I was proving vLLM could serve the LoRA at all. Asked
point-blank whether the box's 128 GB could support something like 150k tokens,
my instinct was to reach for the usual dense-transformer arithmetic: KV cache
scales with layers × heads × head_dim × 2 (K and V) × tokens, and for a 27B
model that arithmetic gets ugly fast.

Except Qwen3.6-27B isn't a dense transformer. `config.json` says so directly,
in a field I'd glossed over every time I'd looked at this model before:

```
"layer_types": [
  "linear_attention", "linear_attention", "linear_attention", "full_attention",
  "linear_attention", "linear_attention", "linear_attention", "full_attention",
  ...
]
```

Every fourth layer is full attention. The other three are linear-attention —
the Gated DeltaNet layers this whole series has been building around since
[Part 11](blog-11-the-scary-parts-were-cheap.md) — and those carry a
*constant-size recurrent state* that doesn't grow with sequence length at all.
Sixty-four layers, sixteen of them actually keep a KV cache. The other
forty-eight are already paying a fixed price, regardless of whether the prompt
is 400 tokens or 150,000.

Run the arithmetic on just the sixteen that matter — 4 KV heads, 256-dim heads,
bf16 — and it's 4 KB per token per layer, 64 KB per token total. A 150,000-token
sequence costs **9.4 GB** of KV cache. On a model whose config also states
`max_position_embeddings: 262144`, meaning 150k isn't even close to a
rope-scaling stretch — it's comfortably inside what the model was trained to
handle natively. The number I'd been treating as a hard architectural
constraint was a leftover default from a smoke test three parts ago.

New command: `--max-model-len 200000`, `--gpu-memory-utilization 0.85` (up from
0.6, since a bigger context budget wants more headroom for the KV pool on top
of the ~52 GB the weights already take). Killed the tmux session, relaunched,
watched the log for the two failure modes that would've mattered —
`CUDA out of memory` and the process dying outright — and fifty seconds later
it was serving again. `nvidia-smi` still reports GPU memory as `N/A` on this
box, same as it did [bringing the machine up in Part 3](blog-03-bringing-up-the-box.md)
— unified memory has no separate VRAM figure to give. That line hasn't gotten
any less funny.

## The flag that was necessary but not sufficient

Somewhere in that same session, a tool-calling request came back unsupported,
and another agent's advice was `--enable-auto-tool-choice`. Reasonable-sounding,
and wrong in the specific way that stale advice usually is: half right. vLLM
won't even *start* with that flag alone — `--enable-auto-tool-choice` tells it
to attempt tool parsing, but it also needs `--tool-call-parser <name>` to know
what the model's tool-call output actually looks like, and there is no default.

Nineteen parser names are registered — `hermes`, `qwen3_xml`, `qwen3_coder`,
`llama3_json`, and so on — because "supports tool calling" isn't one format.
Different chat templates emit different syntax for a function call, and the
parser has to match the template or it silently fails to extract anything.
Guessing `hermes` because it's the old Qwen2.5 answer would have started the
server, accepted the `tools` parameter, and quietly never parsed a single call
— the worst kind of broken, the kind that looks like it's working.

The actual answer was sitting in the model's own `chat_template.jinja`:

```
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
```

Nested XML, not Hermes's flat JSON. That's `qwen3_xml` specifically — a parser
that didn't exist for older Qwen releases, added because this exact template
format is new. `--enable-auto-tool-choice --tool-call-parser qwen3_xml`, one
more restart, and the two flags that looked like a single fix turned out to be
a fix and a lookup.

## The theme, again

Nothing here was hard. The hybrid-attention math is four numbers multiplied
together. The tool-parser mismatch is one `grep` through a template file. What
made both worth a blog post is the same lesson [Part 3](blog-03-bringing-up-the-box.md)
already taught and I still had to relearn: a default that's been sitting
unquestioned since the first time something worked is not the same as a limit.
4096 tokens was a smoke-test artifact wearing the costume of a hardware
ceiling. `--enable-auto-tool-choice` was advice that was true as far as it
went, and not far enough. In both cases the fix was the same move — stop
trusting the number that's already there, go read the file that actually
defines the behavior — and in both cases the file said the box could do more
than the command line had ever asked of it.
