# MLX partial-layer staged execution

## Status

Dense Llama-family MLX stages now run as separate OS processes from partial
SafeTensors artifacts and communicate over Skippy's existing binary stage wire.
The first proof uses `HuggingFaceTB/SmolLM2-135M-Instruct` split at layer 15.

This is a production-shaped bridge with an explicit host control path, not yet
the default automatic mesh launch path:

- `skippy-engine` owns the engine-neutral `StageEngine` contract and residual
  buffer descriptors.
- `skippy-server::engine_transport` serves that contract using the existing
  `StageWireMessage`, ready handshake, activation codec, and reply codec.
- `skippy-server::llama_engine` proves the existing llama `RuntimeState` can
  implement the same dense contract, including F16/BF16/F32 residual conversion
  and checkpoint/restore/trim delegation, without changing the native ABI.
- `MlxStageEngine` auto-detects the materialized SafeTensors family. Dense
  Llama stages own per-session KV caches; the first frontier adapter executes
  one internal, stateless Nemotron-H Nano MoE layer. MLX objects remain on a
  dedicated worker thread.
- `mlx-stage` starts a stage process or drives a chain as a proof client.
- `StagePrepare` / `StageLoad` with `backend=mlx` and an immutable
  `hf-model://org/repo@<commit>` reference now derive or reuse a validated
  quantized stage and start the same engine through the normal host
  stage-control loop.

No process in the proof has access to the complete checkpoint. The tokenizer
and config files are small shared metadata; tensor data comes only from that
process's `model.safetensors`.

## Verified result

On Apple Silicon Metal, using two materialized 155.28 MiB partial files:

| Process | Layers | Tensor file available | RSS after the proof |
| --- | ---: | ---: | ---: |
| stage 0 | `0..15` | 155.28 MiB | 188,784 KiB |
| stage 1 | `15..30` | 155.28 MiB | 189,168 KiB |

The processes exchanged F16 residual activations and generated:

```text
[284, 260, 2240, 314, 1343, 327, 624, 8685]
```

That exactly matches the whole-model and in-process split reference for the
same prompt across prompt prefill and seven subsequent decode calls. Each stage
kept an independent per-layer KV cache, and `Stop` cleared the session in both
processes.

The host-managed proof also passed from a clean MLX stage cache. Both ranges
shared checkpoint identity
`303b5a31e5226edb03a48f6f77464736a91a404b1500f385ec43d0951ce81e87`,
but retained distinct stage cache keys:

| Layers | Planned HTTP payload | Complete source shard | Avoided | Requests |
| --- | ---: | ---: | ---: | ---: |
| `0..15` | 162,857,381 bytes | 269,060,552 bytes | 106,204,032 bytes | 3 |
| `15..30` | 162,858,533 bytes | 269,060,552 bytes | 106,202,880 bytes | 4 |

The test submitted Prepare, polled inventory, submitted Load, checked the
materialized status identity/path, generated the same eight reference tokens,
and submitted Stop through `spawn_stage_control_loop`. The runtime status does
not mislabel the derived slice as the full source model or claim a cache pin
that does not exist.

The next engine-level proof enabled tensor-at-a-time JIT weight quantization.
The pinned safemlx loader visits one dense tensor at a time, quantizes it, and
eagerly evaluates and synchronizes the packed weight/scales/biases before
visiting the next tensor. This bounds the lazy graph, but the TensorView and
stream-copy path can temporarily hold more than one physical source copy. With
affine 4-bit, group size 64, the whole 30-layer reference and the two
independently loaded 15-layer stages generated the same quantized-model tokens:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

The two-stage processes retained 349 MLX parameters each and had post-proof RSS
of 87,392 KiB and 87,952 KiB, versus roughly 189 MiB each at source precision.
This proves deterministic per-stage quantization and quantized stage execution;
it does not by itself prove peak RSS or remove the dense partial artifact.

The next proof removed that dense partial artifact. `mlx-stage derive` consumes
the sequential exact-range session from `model-hf`, quantizes and synchronizes
one matrix at a time, copies packed results into a bounded host-side output
shard, and deletes each dense source tensor before fetching the next. Pure-Rust
SafeTensors I/O avoids linking MLX's bundled GGUF symbols into the existing
Skippy/llama.cpp binary.

With 16 MiB output shards, the two SmolLM2 halves produced:

| Layers | Dense ranges fetched | Quantized artifact | Shards | Largest source temp | MLX peak active | Process max RSS | macOS peak footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0..15` | 162,825,984 B | 45,887,848 B | 3 | 56,623,208 B | 72,548,352 B | 140,853,248 B | 240,157,440 B |
| `15..30` | 162,827,136 B | 45,889,726 B | 3 | 56,623,208 B | 72,548,352 B | 140,722,176 B | 241,550,080 B |

Both derived directories loaded without a quantization request because their
config records the affine-4/group-64 encoding. They again produced exactly:

```text
[260, 2240, 314, 253, 1379, 282, 25801, 28]
```

No complete dense stage or source shard was written. The report separates the
checkpoint/plan/quantizer recipe hash from an output-content digest and records
every output shard hash. Repeating the one-layer derivation produced a
byte-identical weight shard; the whole directories intentionally differ because
reports include local paths and runtime memory evidence. The shard-size option
is a soft bundle target, so one packed tensor may exceed it. This is a bounded
`model_type=llama` artifact builder, not evidence that frontier expert-bank
transforms fit the same bound. Artifact byte counts and the measured
source-plus-output working-disk high-water mark exclude the report, lock files,
and filesystem allocation overhead.

`mlx-stage derive-cached` then proved the reusable cache seam. It maps the
strong recipe identity to a locked managed directory and validates schema,
recipe, aggregate artifact bytes, output-content digest, and every shard hash
before accepting a hit. On the same pinned layer-14 slice, the cold call made 9
tensor-payload range requests; the warm call returned the identical recipe and
content hashes with `cache_hit=true`, made 0 tensor-payload range requests, and
used 17,809,408 B max RSS. It still re-plans lightweight config/index/header
metadata to reconstruct the strong recipe key.

The host control path now consumes this cache directly. `StagePrepare` maps the
load request to a derivation recipe and builds or validates it on a blocking
worker; `StageLoad` validates the same entry and loads MLX from the derived
directory. It fails on a cache miss instead of downloading or quantizing tensor
payloads during Load. The load request carries an additive quantization profile:
`auto`, affine 4-bit, affine 8-bit, or MXFP4. An absent profile from an older
peer means `auto`; an unknown value fails closed. On the current Apple Metal
backend, `auto` selects affine 4-bit with group size 64. The chosen profile is
part of the recipe identity and is carried through inventory, preparation, and
running status. Inventory responses echo the requested profile, so one profile
cannot satisfy readiness for another, including across mixed-version peers.

The host's claimed checkpoint identity is verified from the lightweight
metadata plan before the first tensor payload request. Prepare cancellation is
also threaded into cache-lock waits and the sequential visitor. It is checked
before every payload request and before and after each quantization callback;
an HTTP transfer or MLX operation already in flight finishes cooperatively
before its temporary file is removed.

A clean host-control run built both halves without retaining a dense stage:

| Layers | Exact source tensor bytes | Derived artifact | Payload requests |
| --- | ---: | ---: | ---: |
| `0..15` | 162,825,984 B | 45,859,713 B | 136 |
| `15..30` | 162,827,136 B | 45,861,308 B | 137 |

It completed Prepare, Load, Start, generation, and Stop in 120.74 seconds and
produced the established affine-4 token reference. An immediate identical run
hit both validated entries, performed the same lifecycle in 8.87 seconds, and
used 258,162,688 B max RSS. `MESH_MLX_DERIVED_CACHE_DIR` can isolate or relocate
the host cache for testing and operations. Cache capacity and eviction still
need an owner; warm lookup also still probes lightweight upstream metadata to
reconstruct the strong recipe, then streams each cached shard once to verify
both its shard hash and the aggregate content digest.

The two partial files are the exact-range artifacts described in
`../../spikes/mlx-safetensors-stages/FINDINGS.md`. Tied input/output embeddings
are intentionally duplicated across the stages; that is why the sum of the two
files is larger than the full checkpoint even though neither process downloads
the full checkpoint.

The production `model-hf` planner now also understands the `nemotron_h`
architecture used by Nemotron 3 Nano, including its `backbone.layers.*` layout
and first/final boundary tensors. Against pinned
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16` layer `1`, it selected
2,594,936,576 bytes in 261 tensors from one 4,991,210,024-byte shard; the
largest individual tensor was 19,955,712 bytes. This is metadata/range-planning
evidence only for the general family layout. The derived builder and stage
engine now support exactly one internal Nemotron-H Nano MoE layer at a time.
Mamba, attention, first/final boundaries, and multi-layer hybrid stages remain
fail-closed until their state and boundary semantics are implemented.

Reproduce the metadata-only proof (it downloads the pinned config, index, and
one SafeTensors header, but no tensor payloads):

```bash
cargo test -p model-hf --lib \
  plans_real_nemotron_h_moe_layer_without_tensor_payloads -- \
  --ignored --nocapture
```

The bounded affine4 implementation has also been exercised against that exact
pinned layer. It streamed 2,594,936,576 BF16 bytes through 261 individual range
requests, quantized 258 matrices while retaining three dense tensors, and wrote
730,324,736 tensor bytes. Maximum process RSS was 822,165,504 bytes and the
largest ephemeral source tensor file was 19,955,848 bytes. The resulting
artifact strict-loaded into safemlx's real layer-1 `TransformerBlock` and
produced a finite `[1, 1, 2688]` output for a deterministic nonzero input.
The same artifact then loaded through `MlxStageEngine`; execution through the
shared F32 `StageActivation` contract matched direct block execution within
`atol=1e-4`, `rtol=1e-4` (across repeated validation runs, worst observed max
absolute difference `1.1920929e-7`, max relative difference `1.8225228e-5` for
reference values above `atol`). It
compared two session IDs, reset session 1, and independently compared its
repeated output too. Separate sparse executions were not bit-identical, so the
validator records both hashes and enforces the declared numerical tolerance.
Here, bounded memory means bounded by the final packed layer: the six routed
bank buffers total 718,405,632 bytes. It does not mean derivation stays at the
one-expert (~20 MB source tensor) footprint. The forward is an executable smoke
test, not a dense-versus-quantized numerical parity result.

```bash
just mlx-stage-build
just mlx-stage derive \
  --repo nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16 \
  --revision 97ab8012882a655dc38df4fee47422aca9caca07 \
  --layer-start 1 --layer-end 2 \
  --output /tmp/nemotron-nano-layer1-affine4 \
  --weight-quantization affine4
just mlx-stage validate-nemotron-h \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1
just mlx-stage validate-nemotron-h-stage \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1
just mlx-stage validate-nemotron-h-wire \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1 --wire-dtype f32
just mlx-stage validate-nemotron-h-wire \
  --model /tmp/nemotron-nano-layer1-affine4 --layer 1 --wire-dtype f16
```

The last two commands deliberately put the real layer-1 engine in an
unnecessary two-stage loopback chain. The downstream stage is a synthetic
capture/final engine, not another Nemotron layer. It asserts the forwarded
`PrefillFinal` kind, session, token, position, and `[1, 1, 2688]` residual; it
returns a sentinel prediction; and it records the session reset before the
upstream Stop/ACK completes. The F32 boundary matched direct block execution
with maximum absolute error `1.1920929e-7` under `atol=1e-4`, `rtol=1e-4`.
The F16 boundary had maximum absolute error `0.00062298775` and maximum
relative error `0.00048053052` under `atol=5e-4`, `rtol=1e-3`. Those thresholds
are empirical evidence for this layer and deterministic one-token input, not a
family certification. The input values are multiples of 1/32, so the F16 result
mostly exercises output-boundary rounding rather than difficult input
rounding.

This proves the real Skippy TCP framing, activation codec, sideband forwarding,
predicted reply propagation, and chained Stop/ACK around one real MLX frontier
layer. It does not prove a second real model stage, multi-token prefill, decode,
Nemotron recurrent state, host/QUIC orchestration, or end-to-end token logits.

## Reproduce

Build once:

```bash
just mlx-stage-build
```

Derive both quantized stage directories directly from immutable source ranges:

```bash
just mlx-stage derive \
  --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --layer-start 0 --layer-end 15 \
  --output /tmp/mlx-derived-smol-stage0 \
  --weight-quantization affine4 --shard-size-mib 16

just mlx-stage derive \
  --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --layer-start 15 --layer-end 30 \
  --output /tmp/mlx-derived-smol-stage1 \
  --weight-quantization affine4 --shard-size-mib 16
```

To use the identity-bound cache instead of an explicit output path, replace
`derive` with `derive-cached`, omit `--output`, and optionally pass
`--cache-root`. Repeating the command reports `cache_hit=true` and
`source_range_request_count=0`.

The derived directories are already quantized; do not pass
`--weight-quantization` when serving them.

Start the final stage:

```bash
just mlx-stage serve \
  --model /tmp/mlx-split-smol/stage1 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 1 --layer-start 15 --layer-end 30 \
  --bind 127.0.0.1:19091 --wire-dtype f16 --compute-dtype bf16
```

Add `--weight-quantization affine4` to both stage commands to reproduce the
JIT-quantized proof, then pass
`--expected 260,2240,314,253,1379,282,25801,28` to `mlx-stage prove`.

Start the first stage in another terminal:

```bash
just mlx-stage serve \
  --model /tmp/mlx-split-smol/stage0 \
  --model-id HuggingFaceTB/SmolLM2-135M-Instruct \
  --stage-index 0 --layer-start 0 --layer-end 15 \
  --bind 127.0.0.1:19090 --downstream 127.0.0.1:19091 \
  --wire-dtype f16 --compute-dtype bf16
```

Drive the chain:

```bash
just mlx-stage prove --connect 127.0.0.1:19090 --wire-dtype f16
```

## Deliberate limitations of this checkpoint

- `MlxStageEngine` supports dense Llama ranges and exactly one internal,
  stateless Nemotron-H Nano `E`/MoE layer. It rejects Nemotron Mamba, attention,
  dense-MLP, first/final, and multi-layer ranges. Inkling is not exposed through
  the partial-stage adapter.
- The derived builder handles ordinary rank-2 Llama weights and one Nano split
  expert bank. Inkling still needs its transformed rank-3 grouped-expert loader;
  unsupported families are not silently treated as Llama.
- The pinned safemlx Nemotron-H implementation matches the 52-layer Nano
  schema, not Nemotron 3 Ultra's 108-layer latent-MoE schema. Ultra range plans
  are storage-locality evidence, not executable-family support.
- Bounded Nemotron-H derivation and execution currently accept exactly one
  internal `E`/MoE layer. They do not expose a hybrid multi-layer stage or
  recurrent state on the wire.
- The Nemotron binary-wire validator uses a synthetic adjacent final stage and
  a one-token loopback request. Its three-layer synthetic topology exists only
  to exercise the transport harness; it is not a deployable 52-layer model
  topology.
- Greedy sampling only; sampling metadata is preserved in the contract and
  rejected explicitly when enabled.
- No KV page import/export, cache trim/checkpoint, MTP, speculative verify,
  multimodal projection, or transport batching yet.
- `engine_transport` is the reduced compatibility lane. The mature llama.cpp
  binary server remains unchanged and still owns telemetry, exact-prefix cache,
  batching, and OpenAI orchestration.
- Mesh topology planning does not yet produce MLX stage assignments. The host
  can consume explicit `backend=mlx` Prepare/Load requests, but automatic
  placement, capability advertisement, coordinator model planning, and an
  OpenAI stage-0 frontend remain. Explicit host requests now derive and reuse
  quantized artifacts, but cache eviction and an optional local
  request-to-recipe locator remain. The quantization field is an additive mesh
  protocol change; old peers omit it and therefore mean `auto`, while unknown
  values fail closed on new peers. Automatic placement must capability-gate
  explicit non-default profiles before mixed-version deployment. No Skippy ABI
  changed.
