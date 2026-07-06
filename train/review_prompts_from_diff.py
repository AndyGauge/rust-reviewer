#!/usr/bin/env python3
"""Turn a local git diff into the reviewer's prompt jsonl (same wire format as
`reviewer-run review --dump-prompts`), so the Rust reviewer can review code that
isn't a GitHub PR — e.g. its own generation engine. One {system,user,path,
hunk_header} object per hunk.
"""
import json
import subprocess
import sys

SYSTEM = (
    "You are a senior reviewer for the Rust project. You look for design "
    "problems — API shape, abstractions, invariants, edge cases, "
    "backwards-compatibility, and maintainability — not formatting nits. Given a "
    "diff hunk from a pull request, write the review comment a maintainer would "
    "leave, or say it looks good if there is nothing to raise."
)
REPO = "AndyGauge/rust-reviewer"


def user_prompt(path, hunk):
    return f"Repository: {REPO}\nPull request: #\nFile: {path}\n\n```diff\n{hunk.rstrip()}\n```"


def hunks(diff):
    path, header, body = None, None, []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@"):
            if header is not None:
                yield path, header, "\n".join(body)
            header, body = line, [line]
        elif header is not None and not line.startswith("diff --git") and not line.startswith("index "):
            if line.startswith("--- ") or line.startswith("+++ "):
                continue
            body.append(line)
        if line.startswith("diff --git") and header is not None:
            yield path, header, "\n".join(body)
            header, body = None, []
    if header is not None:
        yield path, header, "\n".join(body)


rng = sys.argv[1] if len(sys.argv) > 1 else "c93e51b^..ea71ed6"
paths = sys.argv[2] if len(sys.argv) > 2 else "crates/reviewer-train/src/"
diff = subprocess.run(
    ["git", "diff", rng, "--", paths], capture_output=True, text=True, check=True
).stdout

out = []
for path, header, hunk in hunks(diff):
    if hunk.count("\n") < 3:  # skip trivial hunks
        continue
    out.append({"system": SYSTEM, "user": user_prompt(path, hunk), "path": path, "hunk_header": header})

with open("new_code_prompts.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
print(f"wrote {len(out)} hunks -> new_code_prompts.jsonl")
for o in out:
    print(f"  {o['path']}  {o['hunk_header'][:60]}")
