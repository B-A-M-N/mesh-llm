# WAN Split Transport - Status and Handoff

Status notes for the WAN split-serving transport work on branch
`wip/wan-direct-prediction-return` (PR #1028). This branch lands the **transport
core** only: it is proposer-agnostic and carries no draft-model code. Companion
to [WAN_SPLIT_PERF.md](WAN_SPLIT_PERF.md) (the throughput/latency model) and
[PIPELINED_VERIFY_WINDOW.md](PIPELINED_VERIFY_WINDOW.md) (the verify-window
pipeline).

## Goal

Make a 2-stage split serve reliably over a WAN link, and give the verify-window
pipeline the transport primitives it needs to hide inter-stage latency. The
throughput lever (committed tokens per ring traversal / traversal time) is left
to the proposer; this branch only makes the traversal correct, fast to warm, and
pipeline-capable.

## What is proven (on this branch)

Validated live on a 2-node WAN split (Apple Silicon coordinator <-> remote
GPU worker, ~24 ms RTT):

- The split **serves** end-to-end. A per-stream RTT gate mismatch previously
  turned every request into a 502; the split now serves correctly, and when a
  direct-return sink is unavailable it falls back to the serial VerifyWindow
  path over the forward lane (slower, not broken).
- **Direct prediction return works over WAN** after raising the return-sink
  ready-handshake timeout (5s -> 20s) to cover cold WAN bridge setup, and
  pre-warming a pool of return-sink sockets. Warm setup dropped from ~12.7s to
  ~0.6s.
- **Pipelining engages**: with a confirmed direct-return route, verify-window
  `max_in_flight` reached depth 4 over the WAN split.

## Landed pieces (keep set)

- `c340f741` - serve without direct-return + per-stream RTT operational-stream
  gate fix. Root-cause fix; turns hard 502s into served requests.
  Note: the warn-and-proceed behavior also changes failure semantics for LAN
  splits - a degraded stream now resolves via re-election rather than a fast
  stream failure.
- `46108cfc` - 20s return-sink ready timeout (matches the forward-lane budget).
- `2b76ba4e` - pre-warmed return-sink pool + implicit pipeline confirmation
  gated on a real direct reply (closes a 300s-hang hazard). Best-tested code on
  the branch. Known limit: the pool covers stage-0 -> immediate-downstream only,
  so 3+-stage chains still cold-open the deeper links.
- `73766d27` - refactor: extract verify-window resolution (clippy line-count).
- `ced1812a` - test fixture: `verify_window_pipeline_force` in the defaults UI
  schema reference.
- `86ff55eb` - per-source accepted-prefix **survival telemetry** (native MTP and
  n-gram). This is the instrument for judging *any* proposer's depth behavior.
- `44d31e3b` - continuous n-gram refill of the speculative horizon (Phase 1).
  Sound mechanism (sealed-tail invariant). It only pays with a proposer that can
  keep the horizon full; expect little from it with the current 4-token n-gram
  window until PR #1037's suffix proposer lands.
- `e48d4583` - forced-pipeline mode (`verify_window_pipeline_force`). This is a
  **benchmark/CI lever only** to bypass the adaptive profitability gate; it is
  not a production default.

Protocol: no wire change in this set. Older peers stay on the serial path;
mixed-version meshes remain safe.

## Key finding: fixed depth is diagnostic, not the win

The one controlled A/B on this branch showed that forcing deeper fixed-depth
speculation *lost* throughput (about 45 -> 37 tok/s) because stale/divergent
windows exploded (2 -> 16). Depth only helps when accepted-prefix survival stays
high at that depth. So `verify_window_pipeline_force` + fixed depth is a
measurement tool, not a serving policy.

The go-forward payoff plan is therefore:

- **Proposer:** ride PR #1037's suffix-ngram (prompt-lookup) proposer, which
  matches verbatim spans up to 64 tokens and holds high acceptance on the
  re-emissive text that dominates agent/coding workloads (file edits, tool-output
  echo). The 4-token llama.cpp n-gram window is too short for this.
- **Depth:** make it **adaptive**, gated on the Phase 0 survival curve per
  source - deep while suffix matches hold, dropping to a single serial traversal
  on a miss. Fixed depth is what lost throughput here.
- Fix PR #1037's two blockers before benchmarking it over WAN: the
  prefix-cache + speculative-checkpoint 502 (`chain_restore_hit`), and
  greedy-equivalence (verified speculative decode must be byte-identical to
  baseline decode).

## Deferred: draft-model speculation

Draft-*model* speculation was explored on this branch and **cut** per reviewer
triage. Once the WAN is hidden, the draft model becomes the loop's cost center
(shard's numbers put it at ~94% of the loop), so a synchronous draft on the
critical path cannot win. That work is parked on branch `wip/wan-draft-ahead`
(not merge-ready). If ever revived, it must be **async draft-ahead** that hides
draft compute inside the WAN RTT, and only for novel-prose workloads where
n-gram survival is near zero - the MTP + suffix composite likely covers the gap
without an external draft.

## Bringup config trap (read before a 2-node split)

The most common failure bringing up a 2-node split is the split silently never
forming: the coordinator sits in `standby`, `peers=1`, forever. Causes seen:

- **Worker `--max-vram` >= model size.** The worker reports it can hold the
  whole model alone, so placement correctly decides no split is needed. Cap
  **both** nodes below model size (e.g. two nodes at 12 GB each for an ~18 GB
  model). If you see `Split waiting for peers ... found 1 eligible
  [worker:>=model-size]`, fix `--max-vram` instead of waiting out the timeout.
- **Same-machine loopback shares one identity.** Two `mesh-llm serve` processes
  on one host load the same `~/.mesh-llm/key` and appear as a single node, so the
  split cannot form. Give the second process a distinct identity with
  `MESH_LLM_EPHEMERAL_KEY=1` (see `docs/design/TESTING.md`, "Forced split on one
  machine"). For a genuine WAN test, use two real machines.
