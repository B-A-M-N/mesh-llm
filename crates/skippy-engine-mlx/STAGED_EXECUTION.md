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
- `MlxStageEngine` loads one materialized partial SafeTensors file, owns
  per-session KV caches on a dedicated MLX worker thread, and executes only its
  configured layer range.
- `mlx-stage` starts a stage process or drives a chain as a proof client.
- `StagePrepare` / `StageLoad` with `backend=mlx` and an immutable
  `hf-model://org/repo@<commit>` reference now materialize and start the same
  engine through the normal host stage-control loop.

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
it does not yet prove peak RSS, direct range-to-quantized-cache disk bounds, or
host/topology selection of the quantization profile.

The two partial files are the exact-range artifacts described in
`../../spikes/mlx-safetensors-stages/FINDINGS.md`. Tied input/output embeddings
are intentionally duplicated across the stages; that is why the sum of the two
files is larger than the full checkpoint even though neither process downloads
the full checkpoint.

## Reproduce

Build once:

```bash
just mlx-stage-build
```

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

- Dense Llama-family checkpoints only in `MlxStageEngine`. The pinned safemlx
  revision has whole-model Inkling and Nemotron-H implementations, but neither
  is exposed through this partial-stage adapter yet.
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
  OpenAI stage-0 frontend remain. Explicit host requests still use source
  precision; quantization selection currently exists only in the engine config
  and `mlx-stage` proof CLI. There is no mesh protocol or Skippy ABI break in
  the explicit consumer path.
