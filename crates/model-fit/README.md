# model-fit

`model-fit` ranks GGUF model artifacts against a local hardware profile.

The crate is intentionally metadata-first and deterministic. It consumes:

- hardware facts and measured GPU bandwidth, decode overhead, scalar compute,
  dense prefill matmul throughput, and MoE-shaped prefill matmul throughput
  from `mesh-llm gpus benchmark`
- GGUF-derived model metadata such as tensor bytes, layer count, hidden width,
  KV dimensions, context length, architecture class, quantization, tokenizer
  metadata, and workload capability evidence
- workload preferences for chat, coding, tool use, summarization, embedding,
  reranking, and related local inference shapes

The selector estimates runtime memory, KV cache size, active decode bytes,
decode throughput, prefill throughput, first-token latency, workload fit, and
split candidacy. It does not use model filenames or catalog reputation as a
performance signal.

## Validation

The crate includes two validation manifests:

- `validation/smoke-models.txt` for self-hosted PR smoke testing
- `validation/deep-models.txt` for manual or nightly high-memory validation
- `validation/q8-moe-models.txt` for focused Q8, dense prefill, and MoE checks

Run a local validation pass:

```bash
just model-fit-release

target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/smoke-models.txt \
  --output-json "$HOME/tmp/model-fit-validation.json"

target/release/model-fit-check-validation \
  --scenario steady_decode \
  --min-models 5 \
  --min-accuracy-models 3 \
  --max-median-absolute-error 0.10 \
  --max-individual-error 0.10 \
  --max-noisy 2 \
  --require-graph-inventory-match \
  --allow-classified-individual-misses \
  "$HOME/tmp/model-fit-validation.json"

target/release/model-fit-check-validation \
  --scenario all \
  --report-only \
  --markdown-out "$HOME/tmp/model-fit-validation.md" \
  "$HOME/tmp/model-fit-validation.json"
```

The validation report is JSON so later agents can analyze hardware facts, model
profiles, recommendations, benchmark observations, the estimator input
contract, and scenario-level agreement.

The gate separates coverage from evidence: `--min-models` checks that the smoke
set ran, while `--min-accuracy-models` checks that enough rows survived the
validator's noise/probe/context exclusions to count as actual throughput
accuracy evidence.

For manual dense-depth investigations, add `--dense-probe-depth deep`. That
keeps regular smoke runs short while allowing an extra `l16` source-shaped dense
graph probe when a larger dense model looks depth-extrapolation limited.

Backend validation builds should compile the native benchmark backend that will
produce the `HardwareProfile`. Metal is built automatically on macOS. CUDA,
ROCm/HIP, and Intel backends are explicit feature selections:

```bash
# CUDA example after scripts/build-llama.sh produced the matching GGML archive.
just model-fit-release cuda .deps/llama-build/build-stage-abi-cuda-sm120

# ROCm/HIP and Intel use the same recipe shape.
just model-fit-release rocm .deps/llama-build/build-stage-abi-hip
just model-fit-release intel
```

The `llama_build_dir` argument is only needed when the GGML decode probe archive
is not in the platform default location.
