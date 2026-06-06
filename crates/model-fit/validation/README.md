# model-fit validation corpora

These manifests define repeatable GGUF sets for `model-fit-validate`.

The lists are stratified by estimator behavior rather than popularity:

- tiny dense models check fixed decode overhead and low-active-byte behavior
- small dense models check the transition into memory-bandwidth-bound decode
- 7B/8B and coder models check common local serving shapes
- quant pairs check Q4/Q5/Q6/Q8 slope changes without changing architecture
- MoE models check active expert bytes instead of total expert bytes
- embedding/reranker models check workload suitability in metadata reports

Use the smoke set for self-hosted PR validation:

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

Use the deep set for manual or nightly validation on high-memory runners:

```bash
target/release/model-fit-validate \
  --no-progress \
  --dense-probe-depth deep \
  --models-file crates/model-fit/validation/deep-models.txt \
  --output-json "$HOME/tmp/model-fit-validation-deep.json"
```

`--dense-probe-depth deep` adds a validation-only `l16` repeated dense graph
probe for Q4 dense models. Keep the default `standard` depth for short PR smoke
runs; use `deep` when investigating depth-extrapolation misses on larger dense
GGUFs.

`model-fit-validate --models-file` ignores blank lines and `#` comments.

Backend-specific runners must build `model-fit` with the native benchmark
backend that will generate `HardwareProfile`. Metal is automatic on macOS; CUDA,
ROCm/HIP, and Intel are explicit recipe arguments. Pass `llama_build_dir` when
the GGML decode-probe archives live outside the platform default build dir:

```bash
just model-fit-release cuda .deps/llama-build/build-stage-abi-cuda-sm120
```

Use the Q8/MoE focused set when changing tensor-type traffic or active-expert
dispatch modeling:

```bash
target/release/model-fit-validate \
  --no-progress \
  --models-file crates/model-fit/validation/q8-moe-models.txt \
  --output-json "$HOME/tmp/model-fit-validation-q8-moe.json"
```

Use the quant-ladder set when comparing Metal and CUDA behavior for the same
dense metadata shape across Q4/Q5/Q6/Q8 without running the broader corpus:

```bash
target/release/model-fit-validate \
  --no-progress \
  --scenario steady_decode \
  --models-file crates/model-fit/validation/quant-ladder-metal-cuda.txt \
  --output-json "$HOME/tmp/model-fit-validation-quant-ladder.json"
```

## ABI decode validation

The validator can run Skippy's single-stage benchmark and the Skippy decode ABI
probe for each GGUF. The ABI probe exercises llama.cpp's decode graph directly
and reports a denoised median over repeated observations; it is meant to check
whether the metadata-only fit estimate lands near the runtime's actual decode
path without feeding observed throughput back into scoring.

The smoke checker deliberately separates structural correctness from throughput
agreement:

- `--require-graph-inventory-match` fails when GGUF metadata no longer maps to
  the llama.cpp decode graph inventory. Dense transformer blocks must match the
  attention/FFN matmul families; sparse MoE blocks must match attention plus
  routed `GGML_OP_MUL_MAT_ID` expert families. A Q8 row may still report
  `metadata_inventory_matches_probe_depth_risk` when the tensor inventory
  matches but the selected timing probe is only one synthetic layer.
- `--allow-classified-individual-misses` does not make a miss disappear. It
  allows a row outside `--max-individual-error` only when the validator labels
  the residual as a concrete diagnostic class such as
  `metadata_estimate_miss`, `probe_not_representative`, or
  `runtime_path_mismatch`. The markdown report still prints the miss.
- `--min-accuracy-models` requires rows that actually participate in the
  throughput accuracy gate after noisy, request-unstable, probe-unstable, or
  otherwise diagnostic-only rows are excluded. Without this, a smoke run can
  cover the right models but prove zero `±10%` tok/s accuracy. The binary
  defaults to one accuracy-gated row for non-`--report-only` checks; the smoke
  command above requires three out of the five model rows.

This shape is intentional. The smoke should catch structural regressions and
unclassified surprises while still letting known, visible residuals stay in the
evidence table until there is a source-grounded fix.

CI gates only the `steady_decode` scenario. The validator still runs and
renders prefill, first-token, and KV-warm-reuse scenarios, but those rows are
report-only evidence until their timing models have the same backend coverage as
steady decode. This prevents a first-token or cache-reuse residual from hiding a
decode-regression signal, while keeping the extra measurements visible for
future source-grounded fixes.

When a steady-decode miss remains after graph inventory matches, treat it as a
timing-model investigation rather than a metadata classification bug. Recent
Metal and CUDA smoke runs show two useful shapes:

- CUDA sparse MoE can have clean tensor inventory and stable runtime samples,
  but still land slower than the metadata estimate because full single-stage
  serving carries runtime/session overhead not present in the selected sparse
  block ABI probe.
- Metal dense Qwen-shape rows can miss in opposite directions on the same
  machine while graph inventory stays exact. That points at backend timing and
  probe representativeness, not model-family names or GGUF tensor grouping.

The steady-decode estimator is source-grounded rather than model-name grounded:

- GGUF tensor profiles identify attention, FFN, output, and expert matmul bytes.
- Tensor type profiles distinguish Q8_0, K-quants, f16/f32, IQ, and unknown
  groups because llama.cpp dispatches different GGML matmul kernels for those
  tensor types.
- Layer count and logical matmul shape counts approximate the per-token graph
  shape around `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`.
- `mesh-llm gpus benchmark` supplies measured decode-shaped bandwidth and fixed
  backend submission overhead; model-fit consumes those hardware facts instead
  of assuming Metal, CUDA, or ROCm behavior.
- Dense prefill uses a separate roofline over GGUF matmul FLOPs, llama.cpp
  ubatch count, measured memory bandwidth, measured graph overhead, and the
  optional `prefill_matmul_tflops_fp16` hardware probe. CUDA, Metal, and ROCm
  backends can all report the same semantic probe through their native matrix
  APIs. Sparse MoE records a separate `prefill_moe_matmul_tflops_fp16` probe,
  but the scorer treats that as an upper bound because raw expert GEMM does not
  include router/id-map, weighting, and aggregation work around
  `GGML_OP_MUL_MAT_ID`. CPU profiles may carry the same optional facts when a
  CPU benchmark supplies them; model-fit does not invent CPU FP16 throughput for
  machines where that is not measured.
- First-token validation serializes predicted prefill, decode, and overhead
  components plus observed tokenize, prefill, decode, and unattributed request
  time. That keeps request/setup residuals visible instead of blaming them on
  decode or prefill.
- `Rejected` is the only non-local-fit status. It means the selected accelerator
  budget could not hold the estimated resident runtime allocation after safety
  margin, the model is not a standalone local GGUF, or the model conflicts with
  the requested workload. Validators skip local-single and ABI decode probes by
  default for rejected models because forcing full offload would only re-test a
  runtime shape the local fitter already rejected.
- On discrete-GPU hosts, transformer generation workloads do not silently fall
  back to CPU RAM to become `FitsLocal`. CPU budgets remain valid for CPU-only
  hosts and non-generative workloads such as embeddings/reranking, but chat,
  coding, tool-calling, summarization, and general generation must fit the
  selected accelerator or be reported as `Rejected`.
- `estimate_confidence=High` is reserved for local fits with measured hardware
  facts and llama-shaped decode kernel probes. Complete GGUF metadata alone is
  not enough to claim ±10% tok/s confidence; it can prove memory fit, but it
  does not prove the backend's quantized matmul path for that tensor/shape
  bucket.

TODO: Verify the ROCm/HIP dense and MoE prefill probes on a real AMD host. The
HIP backend is implemented with hipBLAS and should emit
`prefill_matmul_tflops_fp16` plus `prefill_moe_matmul_tflops_fp16`, but this
must be checked with `mesh-llm gpus benchmark --json` on AMD hardware before we
claim ROCm validation coverage. Treat the run as backend verification only; do
not tune model-fit constants from that single machine.

Representative two-machine broader validation after the ABI decode probe,
source-grounded graph-overhead estimator, prefill scenario split, and
single-model execution-budget selector:

| machine | backend | scenario | median abs error | notable result |
|---|---|---|---:|---|
| Mac Studio M1 Ultra | Metal | steady_decode | 9.8% | 16 benchmarked samples; tiny models now select measured Metal instead of optimistic CPU fallback, with Q5/Q6/Q8 coder and reranker misses still visible. |
| white.local | CUDA | steady_decode | 8.6% | 16 benchmarked samples; Qwen2.5-Coder Q4/Q5/Q6 matched, while Q8, OLMoE, and reranker behavior remain honest misses. |
| Mac Studio M1 Ultra | Metal | prefill | 24.6% | Broader prefill before the roofline split; focused Metal prefill below is now inside the target band. |
| white.local | CUDA | prefill | 108.1% | Broader prefill before the measured prefill-matmul hardware probe; focused CUDA dense prefill below is now inside the target band. |
| Mac Studio M1 Ultra | Metal | first_token | 19.2% | First-token latency includes tokenize, prefill, first decode, and request overhead; larger coder quants are still slower than fit. |
| white.local | CUDA | first_token | 58.9% | First-token prediction still needs prompt-shape and backend scheduling work. |
| Mac Studio M1 Ultra | Metal | kv_warm_reuse | 13.7% | KV reuse is close but still over-predicts Qwen2.5-Coder Q4/Q5 and bge reranker. |
| white.local | CUDA | kv_warm_reuse | 16.4% | CUDA KV reuse is stable but remains slower than fit on several dense and reranker samples. |

These numbers are validation evidence, not calibration inputs. Do not loosen
thresholds, add model-specific exceptions, or use observed throughput in the
metadata-only estimator to make this table pass. Treat misses as hypotheses to
test against source behavior, hardware facts, or broader held-out models.

Focused Q8/MoE follow-up after removing the Q8_0 stored-byte discount,
making measured MoE dispatch overhead use measured fixed submission cost, and
splitting prefill onto a dense matmul roofline when the hardware profile
contains `prefill_matmul_tflops_fp16`:

| machine | backend | scenario | samples | median abs error | notable result |
|---|---|---|---:|---:|---|
| Mac Studio M1 Ultra | Metal | steady_decode | 6 | 8.8% | Qwen2.5-Coder Q8 moved from slower-than-fit to a 1.06 observed/fit match; OLMoE remained close but noisy; bge reranker stayed slower-than-fit. |
| white.local | CUDA | steady_decode | 6 | 8.3% | Qwen2.5-Coder Q8 improved from 0.60 to 0.86 observed/fit, and OLMoE improved from 1.97 faster-than-fit to 0.90 match. |
| Mac Studio M1 Ultra | Metal | prefill | 5 | 3.7% | Dense roofline split matches Qwen2.5-Coder Q5/Q8 and OLMoE; Q4 is 15% faster-than-fit and Q6/Q8 had noisy samples in this run. |
| white.local | CUDA | prefill | 5 | 5.4% | Measured dense and MoE-shaped FP16 prefill probes keep Qwen2.5-Coder Q4/Q5/Q8 in band; Q6 is 14% slower-than-fit, and OLMoE remains noisy because the MoE probe is an upper bound, not a complete router/aggregation model. |
| Mac Studio M1 Ultra | Metal | kv_warm_reuse | 6 | 14.3% | Qwen2.5-Coder Q8 now matches KV reuse; Q4/Q5 and bge reranker remain slower-than-fit/noisy. |
| white.local | CUDA | kv_warm_reuse | 6 | 17.5% | Qwen2.5-Coder and bge reranker KV reuse remain slower than steady-decode fit. |

Focused CUDA first-token validation now reports component timing as well as the
aggregate miss. In the latest white.local Q8/MoE run, observed tokenize time was
about 7 ms and observed unattributed request time was near zero; the residual is
mostly the first decode after prefill, where the benchmark observed roughly
54-83 ms while the metadata-only steady-decode component predicted roughly
1-8 ms. Keep that as a visible residual until model-fit has a source-grounded
first-decode-after-prefill hardware fact.
