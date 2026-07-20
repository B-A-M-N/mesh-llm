# Pipelined VerifyWindow Decode

## Purpose

This document describes Skippy's internal speculative-decode subsystem for
native multi-token prediction (MTP) and the optional MTP-anchored N-gram
extender. It covers the wire protocol, target-verification invariant,
asynchronous scheduling, operating modes, and diagnostic telemetry.

The staged-runtime protocol deliberately has no compatibility path for the
retired synchronous `VerifySpan` message. Public mesh gossip and the
OpenAI-compatible API retain their normal compatibility guarantees.

## Terms

| Term | Meaning |
|---|---|
| Target | The full staged model, authoritative for every emitted token. |
| Native MTP | Model-provided typed draft attached to a target reply. GLM 4.7 Flash currently supplies a narrow `N+1` candidate. |
| N-gram sidecar | A request-local llama.cpp `ngram-cache` proposer over target-committed tokens. |
| Composite proposal | Native-MTP prefix plus an optional N-gram suffix. |
| VerifyWindow | Versioned target request that verifies a candidate span at one session position. |
| Free target token | Target's next token after a fully verified span. |
| Stale window | Optimistic in-flight window invalidated by an earlier divergence. |

## Safety Invariant

MTP and N-gram are candidate sources, never authorities. The target verifies
each candidate sequence, and Skippy commits only the longest target-matching
prefix. A target correction is committed after a rejection.

```mermaid
flowchart LR
  MTP["Native MTP draft"] --> C["Composite candidate"]
  NGRAM["N-gram continuation"] --> C
  C --> W["VerifyWindow to target"]
  W --> R{"Target result"}
  R -->|"full accept"| A["Commit verified candidate\nand free target token"]
  R -->|"partial accept"| P["Commit matching prefix\nthen target correction"]
  R -->|"no proposal"| T["Ordinary target decode"]
```

This invariant also applies when multiple windows are in flight.

## Wire Protocol

`STAGE_STATE_VERSION` is `9`. `VerifyWindow` is wire message kind `21`; the
legacy kind `10` is rejected. An old/new staged-runtime pairing therefore fails
clearly instead of silently interpreting requests with different semantics.

```mermaid
sequenceDiagram
  participant S0 as "Stage 0 / OpenAI frontend"
  participant ST as "Downstream stages + target"
  Note over S0,ST: "Typed native MTP draft travels in target-reply sideband"
  S0->>ST: "VerifyWindow(id, position, current + candidates)"
  ST-->>S0: "PredictedTokens(window id, target tokens, next MTP draft)"
  S0->>S0: "Classify longest matching candidate prefix"
  S0->>S0: "Commit target-verified tokens only"
```

Pipelined decode requires direct prediction return. The target reply must reach
stage zero through the upstream-opened return sink; configuration fails when
that sink is unavailable.

## Composite Proposals

The sidecar exists only inside the native-MTP composite strategy. Skippy uses
llama.cpp's request-local `ngram-cache` rather than a second Rust history
scanner. It reads after the provisional MTP prefix and may provide the entire
candidate on a step where the model returns no MTP token.

```mermaid
flowchart TD
  D["Receive typed MTP draft"] --> Q{"MTP tokens?"}
  Q -->|"no"| P["Cache candidate\nfrom accepted context"]
  Q -->|"yes"| H["Read cache after\nprovisional MTP prefix"]
  H --> X["Append useful suffix only"]
  P --> C["Composite proposal"]
  X --> C
  C --> V["One VerifyWindow"]
```

For MTP `[a, b]`, a valid historical continuation must start `[a, b, ...]`.
The composite proposal becomes `[a, b, c, d]`, not two independent requests.
A one-token N-gram tail is discarded. A rejected tail does not count as an MTP
prefix rejection; it only backs off the sidecar.

The cache is never shared between requests and is updated only after target
tokens commit. Drafting with `[a, b]` is read-only, so a rejected VerifyWindow
cannot affect a later lookup. This permits a cache tail to follow MTP even when
the cache would not independently predict `[a, b]`.

## Adaptive Sidecar Policy

The sidecar begins with the smallest useful tail. A fully accepted tail widens
the next tail by one token, up to the configured maximum. A rejected tail resets
the width and enters sidecar cooldown. With no MTP token, pure N-gram can use
the available N-gram budget.

```mermaid
stateDiagram-v2
  [*] --> InitialTail
  InitialTail --> WiderTail: "full tail accepted"
  WiderTail --> WiderTail: "full tail accepted below maximum"
  InitialTail --> Cooldown: "tail rejected"
  WiderTail --> Cooldown: "tail rejected"
  Cooldown --> NativeOnly: "cooldown proposal"
  NativeOnly --> Cooldown: "cooldown remains"
  NativeOnly --> InitialTail: "cooldown exhausted"
```

## Serial Native-MTP Mode

Native MTP alone uses serial VerifyWindow processing. A window is opened,
verified, classified, and committed before the next window begins.

```mermaid
sequenceDiagram
  participant F as "Frontend"
  participant T as "Target"
  F->>T: "Window 41: current + MTP candidate"
  T-->>F: "Window 41 reply"
  F->>F: "Commit verified prefix"
  F->>T: "Window 42: next candidate"
  T-->>F: "Window 42 reply"
```

This is the native-MTP parity path. It is not decode parallelism by itself.

## Pipelined Composite Mode

`verify_window_pipeline_depth > 1` is a maximum rather than a command to keep
that many windows in flight. The request-local scheduler first measures full
acceptance, stage-zero compute time, and downstream wait by verify width. It
admits a dependent window only after enough observations show that expected
downstream overlap exceeds the cost of stale work. Otherwise the same composite
proposal uses one synchronous batched VerifyWindow. A deeper admitted proposal
is partitioned into FIFO windows. The target's free-advance candidate is
reserved as the next window's optimistic current token, preventing duplicate
KV positions.

Profiles are independent by verify width and retain only recent observations.
An observation counts as a continuation only when the verified window and its
free target both match the buffered candidate. Admission compares expected
downstream overlap with the larger of local-compute or downstream stale-work
cost, including a safety margin. This lets WAN or downstream-heavy topologies
use configured depth while a stage-zero-heavy split remains on the profitable
synchronous batched path.

```mermaid
sequenceDiagram
  participant F as "Stage-zero frontend"
  participant T as "Target"
  Note over F: "Composite proposal: [m1, n1, n2, n3]"
  F->>T: "Window 100 verifies [m1, n1]"
  F->>T: "Window 101 verifies [n2, n3]"
  T-->>F: "Window 100 reply"
  F->>F: "Commit verified prefix"
  T-->>F: "Window 101 reply"
  F->>F: "Commit only if prefix remains valid"
```

Replies complete in FIFO window-id order. An earlier divergence invalidates
later optimistic windows. Skippy drains and records them as stale. The next
decode or verify message carries the corrected absolute position, and every
stage rewinds its local attention KV to that position before executing it.
There is no rollback control message or repair replay.

```mermaid
flowchart LR
  W0["Earlier window\npartial accept"] --> C["Commit matching prefix\n+ correction"]
  W0 --> D["Discard later windows\nas stale"]
  D --> R["Next message names corrected position"]
  R --> N["Each stage rewinds locally and continues"]
```

## Verification Outcomes

| Target result | Committed output | Next action |
|---|---|---|
| Full accept | Candidate plus free target token where applicable | Continue; adaptive width may grow. |
| Tail rejection | MTP prefix and matching tail prefix, then correction | Back off sidecar only. |
| Prefix rejection | Matching prefix, then correction | Handle native MTP rejection and discard stale windows. |
| EOG | Verified prefix through EOG | Stop. |
| No candidate | Ordinary target token | Continue decode. |

## Running On The Two-Host Lab

Use the package-qualified model reference. The normal mesh runtime owns split
planning; do not replace it with a direct `gguf://` reference for this flow.

The package owns a tested declarative default. `mesh-llm` resolves that package
plan once at launch, applies model-level settings before global defaults, and
passes the resulting typed configuration to `skippy-server`. The server does
not read `SKIPPY_NATIVE_MTP_*`, `SKIPPY_NGRAM_CACHE_*`, or
`SKIPPY_VERIFY_WINDOW_*` from its request hot path. Those variables are retired
from supported operation.

### Package Strategy Shape

`model-package.json` names reusable proposers and strategies. A GLM 4.7 Flash
package can expose native MTP plus a request-local cache sidecar as follows:

```json
{
  "generation": {
    "speculative_decoding": {
      "default": "mtp-cache",
      "proposers": {
        "mtp": {
          "type": "native-mtp",
          "prediction_depth": 1,
          "layer_indices": [47]
        },
        "cache": {
          "type": "ngram-cache",
          "ngram_min": 2,
          "ngram_max": 4,
          "max_proposal_tokens": 10,
          "history_scope": "request"
        }
      },
      "strategies": {
        "mtp-cache": {
          "type": "composite",
          "primary": "mtp",
          "extender": "cache",
          "extension_policy": {
            "initial_tokens": 2,
            "max_tokens": 8,
            "tail_backoff_proposals": 5
          }
        }
      }
    }
  }
}
```

Use the following stable strategy names:

| Strategy | Composition | Benchmark condition |
|---|---|---|
| `mtp` | Native MTP proposer | MTP |
| `mtp-cache` | Native MTP primary plus request-local cache N-gram tail | MTP + N-gram cache |

`disabled` is an operator control rather than a package strategy; it supplies
the no-MTP baseline. Every listed strategy is still target-verified.

### Operator Configuration

Choose a package strategy with `speculative.strategy`. `auto` uses the package
default; `disabled` turns speculation off; `mtp` preserves the direct native
MTP path. A named strategy such as `mtp-cache` is valid only when the selected
package declares it. Packages provide the recommended bounds and topology for
a model, while an explicit direct-GGUF configuration enables the request-local
cache by combining native MTP with valid N-gram bounds.

```toml
[defaults.speculative]
strategy = "auto"

[[models]]
model = "meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M"

[models.speculative]
strategy = "mtp-cache"
ngram_max_proposal_tokens = 10
extension_initial_tokens = 2
extension_max_tokens = 8
extension_tail_backoff_proposals = 5
verify_window_min_tokens = 1
verify_window_max_tokens = 6
verify_window_pipeline_depth = 2
```

### No MTP Baseline

```bash
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

Use `[models.speculative] strategy = "disabled"` to make this an explicit
baseline instead of relying on environment variables.

### Native MTP Only

```bash
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

Use `[models.speculative] strategy = "mtp"` to force this control.

### MTP With Cache-backed N-gram Extension

```bash
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft
```

Use `[models.speculative] strategy = "mtp-cache"` with the bounded settings
above when the package declares that recommendation. For a direct GGUF, use
the built-in request-local cache proposer explicitly:

```toml
[models.speculative]
strategy = "mtp"
ngram_min = 2
ngram_max = 4
ngram_max_proposal_tokens = 6
extension_max_tokens = 6
```

With native MTP and an N-gram proposer present, mesh-llm creates the bounded
composite plan. The package remains the preferred way to publish tested values.

### Invocation Overrides

`mesh-llm serve` may temporarily override a package-selected strategy without
editing `config.toml`. CLI settings have highest precedence, then the selected
model entry, then `[defaults.speculative]`; unspecified CLI fields retain the
lower-layer value. Named package strategies remain package-declared. For a
direct GGUF, the CLI may enable only the request-local cache extension, and
only when it supplies valid N-gram bounds alongside native MTP.

```bash
mesh-llm serve meshllm/GLM-4.7-Flash-MTP-GGUF:Q4_K_M --split --no-draft \
  --speculative-strategy mtp \
  --speculative-ngram-min 2 \
  --speculative-ngram-max 4 \
  --speculative-extension-max-tokens 8 \
  --speculative-verify-window-pipeline-depth 2
```

The supported tuning flags are `--speculative-ngram-{min,max}`,
`--speculative-ngram-max-proposal-tokens`,
`--speculative-extension-{initial,max}-tokens`,
`--speculative-extension-tail-backoff-proposals`,
`--speculative-native-mtp-{reject-cooldown-tokens,suppress-cooldown-drafts,suppress-cooldown-draft-limit}`,
and `--speculative-verify-window-{min,max}-tokens` / `--speculative-verify-window-pipeline-depth`.
Use `--speculative-native-mtp-allow-cooldown-drafts` to explicitly override a
configured suppression policy to `false`.

### Standalone Skippy Server

`skippy-server` does not resolve layer-package recommendations. For isolated
stage-server operation it accepts a complete, already resolved JSON
`SpeculativeDecodeConfig` via `serve-binary --openai-speculative-config` or
`serve-openai --speculative-config`. The file is validated as one typed plan
before serving starts. This is intentionally not a second policy-merging path;
normal mesh serving always resolves the package and policy in `mesh-llm`.

```mermaid
flowchart LR
  C["SPEED-Bench client\non micstudio"] --> S0["micstudio :9337\nOpenAI frontend / stage 0"]
  S0 --> L["Persistent direct-LAN\nbinary stage lanes"]
  L --> S1["studio54\nstage 1"]
  S1 --> S0
  S0 --> C
```

The normal planner currently selected `micstudio 0..47` and `studio54 47..48`.
Record layer ranges, direct RTT, lane count, context size, and binary commit
with every benchmark. That shape proves normal split serving but is not directly
comparable to historic 22/26 benchmark rows.

## Telemetry And Interpretation

The OpenAI response `timings` object provides aggregate evidence; debug
telemetry supplies per-window and per-stage detail.

| Question | Counters |
|---|---|
| Is decode faster? | `predicted_per_second`, `predicted_n`, `predicted_ms` |
| Which plan actually ran? | `llama_stage.spec.requested_strategy`, `llama_stage.spec.effective_strategy` |
| Are proposals accepted? | `draft_n`, `draft_n_accepted` |
| Did the sidecar widen MTP? | `native_mtp_hybrid_native_tokens`, `native_mtp_hybrid_ngram_tokens`, `native_mtp_hybrid_proposed_tokens` |
| Were tails useful? | `native_mtp_hybrid_accepted_tail_tokens`, `native_mtp_hybrid_ngram_tail_rejections`, `native_mtp_hybrid_ngram_sidecar_backoffs` |
| Was it pipelined? | `verify_window_depth`, `verify_window_opened`, `verify_window_max_in_flight`, `verify_window_stale_discarded` |
| Did the proposer keep supplying horizon? | `verify_window_horizon_refill_attempts`, `verify_window_horizon_refill_successes`, `verify_window_horizon_refill_tokens`, `verify_window_horizon_refill_misses` |
| Why was depth used or suppressed? | `verify_window_policy_observed_windows`, `verify_window_policy_continuation_windows`, `verify_window_policy_profitable_widths`, `verify_window_policy_permit_checks`, `verify_window_policy_permits`, `verify_window_policy_suppressed` |
| Where was time spent? | `verify_window_downstream_wait_ms`, `verify_window_forward_write_ms`, `verify_window_stage0_compute_ms`, `verify_window_verify_elapsed_ms` |

A useful hybrid run needs more than an increased `draft_n`: it needs accepted
tail tokens, high anchor agreement, bounded stale-window work, and completion
throughput higher than the native-MTP control.

Pipeline-policy telemetry contains only bounded numeric counts and stage timing;
it does not include prompts, completions, token IDs, paths, endpoints, or node
identifiers. Debug OTLP export remains explicitly configured by the operator.
Horizon refills query speculative suffixes only in request-local memory; the
telemetry exports counts, never those suffixes or their token IDs.
