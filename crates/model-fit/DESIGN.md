# model-fit Design

`model-fit` is the local model suitability engine for Mesh LLM. Given a GGUF
model profile and a measured hardware profile, it ranks whether a model is a
good local fit, estimates memory use, estimates steady decode and prefill
performance, and explains the decision in machine-readable detail.

The crate is intentionally deterministic and metadata-first. It must not use a
model filename, catalog reputation, backend name, or validation-set identity as
a performance shortcut. The long-term objective is that a GGUF discovered in the
Hugging Face cache can be inspected, converted into a `ModelProfile`, scored
against a `HardwareProfile`, and reported with honest fit status and confidence.

## Objectives

- Decide whether a GGUF fits locally before serving it.
- Rank local GGUFs by memory fit, context fit, decode cost, prefill cost, and
  workload suitability.
- Estimate steady decode tokens per second from model metadata and measured
  hardware/probe facts alone.
- Estimate prefill throughput and first-token latency separately from decode
  throughput.
- Explain every major score and warning so humans and agents can diagnose why a
  model was selected, penalized, or rejected.
- Validate the estimator with repeatable benchmark runs that output JSON for
  later analysis.

## Non-Goals

- No split-placement scoring until the local fitter is trustworthy.
- No ML-based ranking.
- No backend-specific correction factors.
- No family/name-specific performance multipliers.
- No loosening confidence thresholds to make current validation rows pass.

## Inputs

`HardwareProfile` is produced from `mesh-llm gpus benchmark` plus local system
facts. The benchmark is the only supported source for performance-relevant
machine measurements because advertised memory bandwidth is not a reliable
predictor of local llama.cpp throughput.

Important hardware facts include:

- available host memory and accelerator memory
- accelerator kind and supported backend flags
- measured decode effective bandwidth
- measured decode fixed overhead
- measured dense decode graph probes
- measured scalar, prefill, and MoE-shaped matmul probes where available
- CPU cores and memory pressure facts used for fit and warnings

`ModelProfile` is derived from GGUF metadata. It includes architecture class,
layer count, hidden width, FFN width, attention heads, KV heads, context length,
rope metadata, tokenizer metadata, tensor byte totals, tensor-type byte
breakdown, quantization, and capability evidence.

## Memory Fit

Memory fit is a hard local gate. Runtime memory is estimated from resident model
bytes, KV cache bytes, and scratch/overhead. The model is rejected locally when
the estimate exceeds available memory after the configured safety margin.

KV cache is estimated from the shape llama.cpp needs at inference time:

```text
kv_cache_bytes ~= 2 * layer_count * context_tokens * kv_width * kv_bytes_per_value
```

The factor of two is key plus value. `kv_width` prefers explicit key/value and
KV-head metadata. If metadata is missing, the fitter falls back conservatively.

## Decode Model

Steady decode is usually memory-traffic-bound on local inference, but the bytes
that matter are not just the GGUF file size. llama.cpp submits a graph with
attention, feed-forward, normalization, KV, output/logits, and sampler work. The
fitter therefore estimates active decode cost from GGUF tensor groups and
source-shaped GGML probes rather than applying a single global bandwidth number.

For dense transformer GGUFs, the preferred evidence is a full-token decode graph
probe matching the model's source topology:

- layer count
- hidden width
- FFN width
- query heads
- KV heads
- head dimension
- context shape
- graph feature flags such as norm placement or attention norms
- block and output tensor types

When a full-token probe is available, it replaces the older separate estimates
for transformer block matmuls, KV/activation traffic, output matmul, logits
readback, and sampler overhead. This follows the llama.cpp execution boundary:
the full graph has already paid for those operation slots.

When GGUF metadata shows mixed tensor types inside a dense graph, the fitter
does not blindly add the whole residual tensor cost on top of the full-token
probe. The full graph already measured the same operation slots using the
dominant synthetic tensor type. If same-shape residual evidence exists, the
fitter charges only the replacement delta:

```text
max(actual_residual_type_ms - synthetic_stand_in_type_ms, 0)
```

That rule is source-grounded rather than result-fitted. It uses GGML block
formats only to convert resident bytes into equivalent element counts across
tensor encodings.

If exact source-shaped evidence is missing, the fitter can use shape-surrogate
probes, but those remain lower confidence. Surrogates are useful for ranking,
not for high-confidence ±10% tok/s claims.

## Prefill Model

Prefill is reported separately because it scales with prompt length and uses
different kernels than single-token decode. The validator should run multiple
Skippy workload profiles so prompt-heavy, tool-loop, chat, and long-context
scenarios can be compared independently. A model can be a strong steady-decode
fit and still be a poor long-prefill fit for a workload.

## Architecture Classes

The fitter should work across GGUF architecture classes, but confidence depends
on how much of the architecture is represented by metadata and probes.

- Dense transformers have the strongest current support.
- Sparse MoE models require active-expert accounting and MoE-shaped probes.
- Recurrent or linear-attention models require graph evidence that matches their
  nonstandard attention path.
- Embedding/reranking models are evaluated against embedding/reranking
  workloads rather than penalized globally.

Architecture class is used to choose a source-compatible cost path, not to
hardcode a family multiplier.

## Confidence

High confidence means the estimated steady decode tokens per second is expected
to land within ±10% of observed benchmark throughput for the same scenario.

The fitter should only claim high confidence when:

- the model fits locally with memory headroom
- core GGUF shape metadata is present
- tensor type byte mapping is understood
- measured hardware profile exists
- source-shaped probe evidence covers the dominant decode cost
- validation noise is low enough to make observed throughput meaningful

Medium or low confidence is not failure. It is an honest signal that the model
can still be ranked or served locally, but the tok/s estimate should not be used
as a tight promise.

## Validation

`model-fit-validate` is the repeatable validation harness. It takes model refs,
downloads or resolves the GGUFs, builds `ModelProfile`s, generates a
`HardwareProfile`, runs Skippy single-stage benchmarks for selected workload
scenarios, and writes JSON containing:

- hardware profile
- model profiles
- recommendations
- decode probe diagnostics
- runtime diagnostics
- benchmark observations
- interpretation of estimate versus observed throughput

Validation should cover stable dense quant ladders, small models, larger dense
models, MoE models, and unusual GGUF tensor layouts. Rows outside the supported
evidence envelope should be classified as low confidence rather than used to
tune hidden constants.

## Current Direction

The active work is tightening the estimator around llama.cpp/GGML execution
boundaries:

- prefer measured hardware and probe facts over advertised bandwidth
- expose graph/source diagnostics when estimates miss
- add missing quant probe coverage such as `Q5_K`
- account for mixed tensor residuals as replacement deltas when a full-token
  graph already covers the operation slots
- keep confidence strict instead of widening error bars

The next broadening step is better GGUF tensor-type coverage across Hugging Face
exports, followed by repeatable smoke validation on Metal and CUDA.
