#!/usr/bin/env python3
"""Benchmark the suffix N-gram proposer against off/simple/cache arms.

Attaches to already-running endpoints, one per arm; the proposer is selected via
each server's [models.speculative] config. Requires a >=2-stage Skippy split —
single-node serving runs an in-process llama.cpp path that never invokes the
Cache/Suffix proposers (draft_n stays 0). Reads decode tok/s and acceptance from
the server timings (predicted_per_second, draft_n / draft_n_accepted).

    ./skippy-suffix-proposer-bench.py --model <id> \
        --arm off=http://127.0.0.1:9401 --arm simple=http://127.0.0.1:9402 \
        --arm cache=http://127.0.0.1:9403 --arm suffix=http://127.0.0.1:9404
"""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

EDIT_FILE = """def parse_config(path):
    with open(path) as handle:
        raw = handle.read()
    data = json.loads(raw)
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = value.strip()
        else:
            result[key] = value
    return result"""

WORKLOADS: dict[str, str] = {
    "edit": (
        "Here is a Python function:\n\n```python\n"
        + EDIT_FILE
        + "\n```\n\nRe-emit the entire function verbatim, changing only the name "
        "`parse_config` to `load_config`. Output just the code."
    ),
    "chat": (
        "Explain, in two short paragraphs, why generating a token is more "
        "expensive than verifying one in speculative decoding."
    ),
}


def load_agent_loop() -> str | None:
    corpus = HERE / "skippy-coding-agent-loop.jsonl"
    if not corpus.exists():
        return None
    with corpus.open() as handle:
        first = handle.readline().strip()
    if not first:
        return None
    return json.loads(first).get("prompt")


def http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class Sample:
    tok_s: float
    predicted_n: int
    draft_n: int
    draft_accepted: int


def run_once(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> Sample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    response = http_json(base_url.rstrip("/") + "/v1/chat/completions", payload, timeout)
    timings = response.get("timings", {}) or {}
    return Sample(
        tok_s=float(timings.get("predicted_per_second", 0.0)),
        predicted_n=int(timings.get("predicted_n", 0)),
        draft_n=int(timings.get("draft_n", 0)),
        draft_accepted=int(timings.get("draft_n_accepted", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=URL",
        help="A speculative arm: e.g. suffix=http://127.0.0.1:9404 (repeatable)",
    )
    parser.add_argument("--model", required=True, help="Served model id (see /v1/models)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    arms = dict(item.split("=", 1) for item in args.arm)
    workloads = dict(WORKLOADS)
    agent = load_agent_loop()
    if agent:
        workloads["tool_loop"] = agent

    print(f"arms={list(arms)}  workloads={list(workloads)}  runs={args.runs}\n")
    table: dict[str, dict[str, float]] = {}
    for arm_name, url in arms.items():
        table[arm_name] = {}
        for wl_name, prompt in workloads.items():
            toks, drafted, accepted = [], 0, 0
            for _ in range(args.runs):
                sample = run_once(url, args.model, prompt, args.max_tokens, args.timeout)
                toks.append(sample.tok_s)
                drafted += sample.draft_n
                accepted += sample.draft_accepted
            mean_tok = statistics.mean(toks)
            table[arm_name][wl_name] = mean_tok
            accept = f"  accept={accepted / drafted:.2f} (drafted {drafted})" if drafted else ""
            print(f"  {arm_name:8} {wl_name:10} {mean_tok:7.1f} tok/s{accept}")

    print("\n=== mean decode tok/s (rows: arm, cols: workload) ===")
    cols = list(workloads)
    print("arm".ljust(10) + "".join(c.ljust(12) for c in cols))
    for arm_name, row in table.items():
        print(arm_name.ljust(10) + "".join(f"{row[c]:<12.1f}" for c in cols))


if __name__ == "__main__":
    main()
