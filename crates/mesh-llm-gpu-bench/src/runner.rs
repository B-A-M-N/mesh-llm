use crate::BenchmarkOutput;
use anyhow::Result;
#[cfg(any(
    not(target_os = "macos"),
    not(feature = "cuda"),
    not(feature = "hip"),
    not(feature = "intel")
))]
use anyhow::anyhow;
use std::{hint::black_box, sync::mpsc, thread, time::Duration};

const SAMPLER_PROBE_PROMPT_TOKENS: usize = 4096;
const SAMPLER_PROBE_VOCAB_TOKENS: usize = 131_072;
const SAMPLER_PROBE_RUNS: usize = 9;
const SAMPLER_PROBE_TOP_K: usize = 40;
const SAMPLER_PROBE_TOP_P: f32 = 0.95;
const SAMPLER_PROBE_MIN_P: f32 = 0.05;
const SAMPLER_PROBE_TEMPERATURE: f32 = 0.8;
const RUNTIME_DECODE_OVERHEAD_TOKENS: usize = 4096;
const RUNTIME_DECODE_OVERHEAD_RUNS: usize = 7;

#[derive(Clone, Copy)]
struct SamplerProbe {
    history_us_per_token: f64,
    vocab_us_per_token: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub backend: BenchmarkBackend,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProbeDepth {
    HardwareOnly,
    #[default]
    Standard,
    Deep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkOptions {
    pub probe_depth: ProbeDepth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeBlockGraphProbeShape {
    pub expert_count: u32,
    pub experts_used: u32,
    pub expert_width: u32,
    pub hidden: u32,
    pub kv_width: u32,
    pub repeat_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseGraphProbeShape {
    pub hidden: u32,
    pub kv_width: u32,
    pub ffn: u32,
    pub repeat_layers: u32,
    pub graph_features: u32,
    pub norm_head_width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseSampledTokenProbeShape {
    pub hidden: u32,
    pub kv_width: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub repeat_layers: u32,
    pub graph_features: u32,
    pub norm_head_width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFullTokenProbeShape {
    pub hidden: u32,
    pub kv_width: u32,
    pub ffn: u32,
    pub vocab: u32,
    pub repeat_layers: u32,
    pub graph_features: u32,
    pub norm_head_width: u32,
    pub head_dim: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub context_tokens: u32,
    pub active_context_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionRuntimeProbeShape {
    pub head_dim: u32,
    pub query_heads: u32,
    pub kv_heads: u32,
    pub context_tokens: u32,
    pub repeat_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogitsReadbackProbeShape {
    pub vocab: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearAttentionGraphProbeShape {
    pub hidden: u32,
    pub qkv_width: u32,
    pub gate_width: u32,
    pub state_width: u32,
    pub output_input_width: u32,
    pub ffn: u32,
    pub recurrent_layers: u32,
    pub full_attention_layers: u32,
    pub kv_width: u32,
    pub graph_features: u32,
    pub norm_head_width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputProjectionProbeShape {
    pub hidden: u32,
    pub vocab: u32,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            probe_depth: ProbeDepth::Standard,
        }
    }
}

pub fn runner_for(
    os: &str,
    gpu_count: u8,
    gpu_name: Option<&str>,
    is_soc: bool,
) -> Option<BenchmarkRunner> {
    if gpu_count == 0 {
        tracing::debug!("no GPUs detected; skipping benchmark");
        return None;
    }

    let gpu_upper = gpu_name.unwrap_or("").to_uppercase();

    if os == "macos" && is_soc {
        return Some(BenchmarkRunner {
            backend: BenchmarkBackend::Metal,
        });
    }

    if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA")
            || gpu_upper.contains("ORIN")
            || gpu_upper.contains("NVGPU")
            || gpu_upper.contains("TEGRA")
        {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }

        if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Hip,
            });
        }

        if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!(
                "Intel GPU benchmark is not supported in standard mesh-llm builds; skipping"
            );
            return None;
        }

        if os == "linux" && is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA; attempting");
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }
    }

    tracing::warn!("could not identify benchmark runner for GPU platform: {gpu_name:?}");
    None
}

pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            let error_message = serde_json::from_slice::<serde_json::Value>(stdout)
                .ok()
                .and_then(|val| {
                    val.get("error")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                });
            if let Some(msg) = error_message {
                tracing::warn!("benchmark reported error: {msg}");
                return None;
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}

pub fn run_benchmark(runner: BenchmarkRunner, timeout: Duration) -> Result<Vec<BenchmarkOutput>> {
    run_benchmark_with_options(runner, timeout, BenchmarkOptions::default())
}

pub fn run_benchmark_with_options(
    runner: BenchmarkRunner,
    _timeout: Duration,
    options: BenchmarkOptions,
) -> Result<Vec<BenchmarkOutput>> {
    let mut outputs = match runner.backend {
        BenchmarkBackend::Metal => run_metal_benchmark(),
        BenchmarkBackend::Cuda => run_cuda_benchmark(),
        BenchmarkBackend::Hip => run_hip_benchmark(),
        BenchmarkBackend::Intel => run_intel_benchmark(),
    }?;
    attach_ggml_decode_probes(runner.backend, options.probe_depth, &mut outputs);
    attach_sampler_probe(&mut outputs);
    attach_decode_runtime_overhead_probe(&mut outputs);
    Ok(outputs)
}

pub fn run_model_moe_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    repeat_layers: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_moe_graph_probe_impl(
        backend,
        tensor_type,
        expert_count,
        experts_used,
        expert_width,
        hidden,
        repeat_layers,
    )
}

pub fn run_model_moe_block_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_moe_block_graph_probe_impl(backend, tensor_type, shape)
}

pub fn run_model_moe_block_decode_submission_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
    context_tokens: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_moe_block_decode_submission_probe_impl(backend, tensor_type, shape, context_tokens)
}

pub fn run_model_dense_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_graph_probe_impl(backend, tensor_type, shape)
}

pub fn run_model_dense_sampled_token_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseSampledTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_sampled_token_probe_impl(backend, tensor_type, shape)
}

pub fn run_model_dense_full_token_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_full_token_probe_impl(backend, block_tensor_type, output_tensor_type, shape)
}

pub fn run_model_dense_full_token_handoff_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_full_token_handoff_probe_impl(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

pub fn run_model_dense_decode_submission_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_decode_submission_probe_impl(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

pub fn run_model_dense_source_sampled_token_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_dense_source_sampled_token_probe_impl(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

pub fn run_model_attention_runtime_probe(
    backend: BenchmarkBackend,
    shape: AttentionRuntimeProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_attention_runtime_probe_impl(backend, shape)
}

pub fn run_model_logits_readback_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_logits_readback_probe_impl(backend, shape)
}

pub fn run_model_logits_sync_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_logits_sync_probe_impl(backend, shape)
}

pub fn run_model_logits_output_handoff_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_logits_output_handoff_probe_impl(backend, shape)
}

pub fn run_model_linear_attention_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: LinearAttentionGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_linear_attention_graph_probe_impl(backend, tensor_type, shape)
}

pub fn run_model_output_projection_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: OutputProjectionProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    run_model_output_projection_probe_impl(backend, tensor_type, shape)
}

fn attach_ggml_decode_probes(
    backend: BenchmarkBackend,
    probe_depth: ProbeDepth,
    outputs: &mut [BenchmarkOutput],
) {
    if probe_depth == ProbeDepth::HardwareOnly {
        return;
    }

    let probes = run_ggml_decode_probes(backend, probe_depth);
    if probes.is_empty() {
        return;
    }
    for output in outputs {
        output.decode_kernel_probes.extend(probes.clone());
    }
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_ggml_decode_probes(
    backend: BenchmarkBackend,
    probe_depth: ProbeDepth,
) -> Vec<crate::DecodeKernelProbe> {
    match crate::ggml_probe::run(backend, probe_depth) {
        Ok(probes) => probes,
        Err(error) => {
            tracing::warn!("GGML decode kernel probe failed: {error:#}");
            Vec::new()
        }
    }
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_ggml_decode_probes(
    _backend: BenchmarkBackend,
    _probe_depth: ProbeDepth,
) -> Vec<crate::DecodeKernelProbe> {
    Vec::new()
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_moe_graph_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    repeat_layers: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_moe_graph_probe(
        backend,
        tensor_type,
        expert_count,
        experts_used,
        expert_width,
        hidden,
        repeat_layers,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_moe_block_graph_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_moe_block_graph_probe(backend, tensor_type, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_moe_block_decode_submission_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
    context_tokens: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_moe_block_decode_submission_probe(
        backend,
        tensor_type,
        shape,
        context_tokens,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_graph_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_graph_probe(backend, tensor_type, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_sampled_token_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseSampledTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_sampled_token_probe(backend, tensor_type, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_full_token_probe_impl(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_full_token_probe(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_full_token_handoff_probe_impl(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_full_token_handoff_probe(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_decode_submission_probe_impl(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_decode_submission_probe(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_dense_source_sampled_token_probe_impl(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_dense_source_sampled_token_probe(
        backend,
        block_tensor_type,
        output_tensor_type,
        shape,
    )
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_attention_runtime_probe_impl(
    backend: BenchmarkBackend,
    shape: AttentionRuntimeProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_attention_runtime_probe(backend, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_logits_readback_probe_impl(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_logits_readback_probe(backend, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_logits_sync_probe_impl(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_logits_sync_probe(backend, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_logits_output_handoff_probe_impl(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_logits_output_handoff_probe(backend, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_linear_attention_graph_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: LinearAttentionGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_linear_attention_graph_probe(backend, tensor_type, shape)
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn run_model_output_projection_probe_impl(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: OutputProjectionProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    crate::ggml_probe::run_output_projection_probe(backend, tensor_type, shape)
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_moe_graph_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _expert_count: u32,
    _experts_used: u32,
    _expert_width: u32,
    _hidden: u32,
    _repeat_layers: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_moe_block_graph_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: MoeBlockGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_moe_block_decode_submission_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: MoeBlockGraphProbeShape,
    _context_tokens: u32,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_graph_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: DenseGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_sampled_token_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: DenseSampledTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_full_token_probe_impl(
    _backend: BenchmarkBackend,
    _block_tensor_type: &str,
    _output_tensor_type: &str,
    _shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_full_token_handoff_probe_impl(
    _backend: BenchmarkBackend,
    _block_tensor_type: &str,
    _output_tensor_type: &str,
    _shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_decode_submission_probe_impl(
    _backend: BenchmarkBackend,
    _block_tensor_type: &str,
    _output_tensor_type: &str,
    _shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_dense_source_sampled_token_probe_impl(
    _backend: BenchmarkBackend,
    _block_tensor_type: &str,
    _output_tensor_type: &str,
    _shape: DenseFullTokenProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_attention_runtime_probe_impl(
    _backend: BenchmarkBackend,
    _shape: AttentionRuntimeProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_logits_readback_probe_impl(
    _backend: BenchmarkBackend,
    _shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_logits_sync_probe_impl(
    _backend: BenchmarkBackend,
    _shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_logits_output_handoff_probe_impl(
    _backend: BenchmarkBackend,
    _shape: LogitsReadbackProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_linear_attention_graph_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: LinearAttentionGraphProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn run_model_output_projection_probe_impl(
    _backend: BenchmarkBackend,
    _tensor_type: &str,
    _shape: OutputProjectionProbeShape,
) -> Result<Vec<crate::DecodeKernelProbe>> {
    Ok(Vec::new())
}

fn attach_sampler_probe(outputs: &mut [BenchmarkOutput]) {
    let probe = measure_sampler_probe();
    for output in outputs {
        output.sampler_history_us_per_token = Some(probe.history_us_per_token);
        output.sampler_vocab_us_per_token = Some(probe.vocab_us_per_token);
    }
}

fn attach_decode_runtime_overhead_probe(outputs: &mut [BenchmarkOutput]) {
    let overhead_ms = measure_decode_runtime_overhead_ms();
    for output in outputs {
        output.decode_runtime_overhead_ms = Some(overhead_ms);
    }
}

fn measure_decode_runtime_overhead_ms() -> f64 {
    let mut samples = Vec::with_capacity(RUNTIME_DECODE_OVERHEAD_RUNS);
    for _ in 0..RUNTIME_DECODE_OVERHEAD_RUNS {
        samples.push(measure_decode_runtime_overhead_once_ms());
    }
    samples.sort_by(|left, right| left.total_cmp(right));
    samples[RUNTIME_DECODE_OVERHEAD_RUNS / 2]
}

fn measure_decode_runtime_overhead_once_ms() -> f64 {
    // This probe is deliberately host/runtime-shaped, not backend-shaped. The
    // native benchmark already measures empty GPU dispatch overhead; local
    // serving also pays CPU-side per-token control work around decode: a token
    // request crosses a runtime boundary, updates session-visible state, and
    // hands a result back to the caller before the next sampled token can be
    // accepted. A two-channel handoff with tiny state updates is a portable
    // lower-bound for that control path. It is not calibrated from any GGUF
    // benchmark result and it is intentionally reported separately so residual
    // model/runtime misses stay visible.
    let (token_tx, token_rx) = mpsc::sync_channel::<u64>(1);
    let (ack_tx, ack_rx) = mpsc::sync_channel::<u64>(1);
    let worker = thread::spawn(move || {
        let mut state = 0u64;
        while let Ok(token) = token_rx.recv() {
            state = state
                .wrapping_mul(1_099_511_628_211)
                .wrapping_add(token)
                .rotate_left(7);
            if ack_tx.send(state).is_err() {
                break;
            }
        }
        state
    });

    let started = std::time::Instant::now();
    let mut observed = 0u64;
    for token in 0..RUNTIME_DECODE_OVERHEAD_TOKENS as u64 {
        token_tx
            .send(token)
            .expect("runtime overhead worker should receive token");
        observed ^= ack_rx
            .recv()
            .expect("runtime overhead worker should return token");
    }
    drop(token_tx);
    let worker_state = worker.join().unwrap_or_default();
    black_box((observed, worker_state));
    started.elapsed().as_secs_f64() * 1000.0 / RUNTIME_DECODE_OVERHEAD_TOKENS as f64
}

fn measure_sampler_probe() -> SamplerProbe {
    if let Some(probe) = measure_native_source_sampler_probe() {
        return probe;
    }

    let mut history_samples = Vec::with_capacity(SAMPLER_PROBE_RUNS);
    let mut vocab_samples = Vec::with_capacity(SAMPLER_PROBE_RUNS);
    for _ in 0..SAMPLER_PROBE_RUNS {
        history_samples.push(measure_sampler_history_us_per_token());
        vocab_samples.push(measure_sampler_vocab_us_per_token());
    }
    history_samples.sort_by(|left, right| left.total_cmp(right));
    vocab_samples.sort_by(|left, right| left.total_cmp(right));
    // The sampler probe is deterministic CPU work: build/sort/filter a fixed
    // synthetic candidate set and accept a fixed token-history shape. Unlike
    // the GPU bandwidth probes, repeated sampler samples are not estimating a
    // device throughput distribution; they are trying to isolate a small
    // single-thread CPU cost that can be badly inflated by OS scheduling,
    // thermal transitions, or another process preempting the benchmark thread.
    //
    // Those disturbances only add time. If we take the median, a short burst of
    // host noise can become a bogus `us_per_vocab_entry` fact, and model-fit
    // then multiplies that bad measurement by GGUF vocab size. That is how a
    // noisy sampler probe turns into a multi-ms/token decode prediction miss
    // even when the source-shaped GGML graph probes and real Skippy decode
    // agree. Use the fastest finite positive repeat as the least-contaminated
    // lower-bound measurement, the same principle microbenchmarks use when the
    // measured work is deterministic and external interference is additive.
    SamplerProbe {
        history_us_per_token: fastest_positive_sample(&history_samples),
        vocab_us_per_token: fastest_positive_sample(&vocab_samples),
    }
}

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
fn measure_native_source_sampler_probe() -> Option<SamplerProbe> {
    // Prefer the native C++ sampler probe when it is linked because it follows
    // the source-visible Skippy/llama.cpp sampled decode path more closely
    // than the Rust fallback below:
    //
    //   * Skippy builds a fresh full-vocab `std::vector<llama_token_data>` from
    //     `llama_get_logits_ith()`.
    //   * llama.cpp's default top-k sampler uses `std::partial_sort` over that
    //     candidate array for k <= 128.
    //   * The chain then applies top-p, min-p, temperature, and distribution
    //     sampling with source-shaped CPU data structures.
    //
    // Tiny models expose this term brutally: once transformer decode is only a
    // few milliseconds, a too-cheap sampler probe can double the predicted
    // tok/s. This is not calibration from any fitted model's observed speed;
    // it is a machine-local measurement of a source-shaped runtime primitive
    // that model-fit later scales by GGUF vocabulary size.
    match crate::ggml_probe::run_sampler_probe(
        SAMPLER_PROBE_VOCAB_TOKENS,
        SAMPLER_PROBE_PROMPT_TOKENS,
    ) {
        Ok(probe)
            if probe.history_us_per_token.is_finite()
                && probe.vocab_us_per_token.is_finite()
                && probe.history_us_per_token > 0.0
                && probe.vocab_us_per_token > 0.0 =>
        {
            Some(SamplerProbe {
                history_us_per_token: probe.history_us_per_token,
                vocab_us_per_token: probe.vocab_us_per_token,
            })
        }
        Ok(probe) => {
            tracing::warn!(
                history_us_per_token = probe.history_us_per_token,
                vocab_us_per_token = probe.vocab_us_per_token,
                "GGML source-shaped sampler probe returned non-positive output; using Rust fallback"
            );
            None
        }
        Err(error) => {
            tracing::warn!("GGML source-shaped sampler probe failed: {error:#}");
            None
        }
    }
}

#[cfg(not(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe)))]
fn measure_native_source_sampler_probe() -> Option<SamplerProbe> {
    None
}

fn fastest_positive_sample(samples: &[f64]) -> f64 {
    samples
        .iter()
        .copied()
        .find(|sample| sample.is_finite() && *sample > 0.0)
        .unwrap_or(0.0)
}

fn measure_sampler_history_us_per_token() -> f64 {
    let tokens = (0..SAMPLER_PROBE_PROMPT_TOKENS)
        .map(|index| ((index * 1_103 + 17) % SAMPLER_PROBE_VOCAB_TOKENS) as u32)
        .collect::<Vec<_>>();
    let mut recent_counts = vec![0u16; 65_536];
    let started = std::time::Instant::now();
    let mut state = 0u64;
    for token in &tokens {
        let slot = (*token as usize) & (recent_counts.len() - 1);
        recent_counts[slot] = recent_counts[slot].wrapping_add(1);
        state = state
            .wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(*token))
            .wrapping_add(u64::from(recent_counts[slot]));
        black_box(state);
    }
    started.elapsed().as_secs_f64() * 1_000_000.0 / tokens.len() as f64
}

fn measure_sampler_vocab_us_per_token() -> f64 {
    let started = std::time::Instant::now();
    let mut candidates = sampler_probe_candidates();
    apply_sampler_probe_top_k(&mut candidates, SAMPLER_PROBE_TOP_K);
    apply_sampler_probe_top_p(&mut candidates, SAMPLER_PROBE_TOP_P);
    apply_sampler_probe_min_p(&mut candidates, SAMPLER_PROBE_MIN_P);
    apply_sampler_probe_temperature(&mut candidates, SAMPLER_PROBE_TEMPERATURE);
    let selected = sampler_probe_select(&candidates);
    black_box((selected, candidates.len()));
    started.elapsed().as_secs_f64() * 1_000_000.0 / SAMPLER_PROBE_VOCAB_TOKENS as f64
}

#[derive(Clone, Copy)]
struct SamplerProbeCandidate {
    id: u32,
    logit: f32,
    p: f32,
}

fn sampler_probe_candidates() -> Vec<SamplerProbeCandidate> {
    (0..SAMPLER_PROBE_VOCAB_TOKENS as u32)
        .map(|id| {
            let logit = ((id.wrapping_mul(1_664_525).wrapping_add(1_013_904_223) & 0xffff) as f32)
                / 65_536.0;
            SamplerProbeCandidate { id, logit, p: 0.0 }
        })
        .collect()
}

fn apply_sampler_probe_top_k(candidates: &mut Vec<SamplerProbeCandidate>, k: usize) {
    // Skippy's chat sampler follows the llama.cpp sampler chain. With current
    // server defaults, sampled decode first applies top-k before top-p/min-p
    // and temperature. The old benchmark used a greedy max scan, which missed
    // the source-visible candidate ranking work that dominates very small
    // models. This probe stays model-independent: it measures host/runtime
    // work per vocab entry and model-fit scales it by GGUF vocab size.
    if k == 0 || candidates.len() <= k {
        return;
    }
    let target = k - 1;
    candidates.select_nth_unstable_by(target, compare_sampler_probe_candidate_desc);
    candidates.truncate(k);
    candidates.sort_unstable_by(compare_sampler_probe_candidate_desc);
}

fn apply_sampler_probe_top_p(candidates: &mut Vec<SamplerProbeCandidate>, top_p: f32) {
    if !(0.0..1.0).contains(&top_p) || candidates.is_empty() {
        return;
    }
    sampler_probe_softmax(candidates);
    let mut cumulative = 0.0f32;
    let mut keep = candidates.len();
    for (index, candidate) in candidates.iter().enumerate() {
        cumulative += candidate.p;
        if cumulative >= top_p {
            keep = index + 1;
            break;
        }
    }
    candidates.truncate(keep.max(1));
}

fn apply_sampler_probe_min_p(candidates: &mut Vec<SamplerProbeCandidate>, min_p: f32) {
    if min_p <= 0.0 || candidates.is_empty() {
        return;
    }
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let min_logit = max_logit + min_p.ln();
    candidates.retain(|candidate| candidate.logit >= min_logit);
    if candidates.is_empty() {
        candidates.push(SamplerProbeCandidate {
            id: 0,
            logit: max_logit,
            p: 1.0,
        });
    }
}

fn apply_sampler_probe_temperature(candidates: &mut [SamplerProbeCandidate], temperature: f32) {
    if temperature <= 0.0 {
        return;
    }
    for candidate in candidates {
        candidate.logit /= temperature;
    }
}

fn sampler_probe_select(candidates: &[SamplerProbeCandidate]) -> Option<(u32, f32, f32)> {
    candidates
        .iter()
        .max_by(|left, right| left.logit.total_cmp(&right.logit))
        .map(|candidate| (candidate.id, candidate.logit, candidate.p))
}

fn sampler_probe_softmax(candidates: &mut [SamplerProbeCandidate]) {
    let max_logit = candidates
        .iter()
        .map(|candidate| candidate.logit)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0f32;
    for candidate in candidates.iter_mut() {
        candidate.p = (candidate.logit - max_logit).exp();
        total += candidate.p;
    }
    if total <= 0.0 {
        return;
    }
    for candidate in candidates {
        candidate.p /= total;
    }
}

fn compare_sampler_probe_candidate_desc(
    left: &SamplerProbeCandidate,
    right: &SamplerProbeCandidate,
) -> std::cmp::Ordering {
    right.logit.total_cmp(&left.logit)
}

#[cfg(target_os = "macos")]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::metal::run()
}

#[cfg(not(target_os = "macos"))]
fn run_metal_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Metal benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "cuda")]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::cuda::run()
}

#[cfg(not(feature = "cuda"))]
fn run_cuda_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "CUDA benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "hip")]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::hip::run()
}

#[cfg(not(feature = "hip"))]
fn run_hip_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "HIP benchmark backend was not compiled into this mesh-llm binary"
    ))
}

#[cfg(feature = "intel")]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    crate::intel::run()
}

#[cfg(not(feature = "intel"))]
fn run_intel_benchmark() -> Result<Vec<BenchmarkOutput>> {
    Err(anyhow!(
        "Intel benchmark backend was not compiled into this mesh-llm binary"
    ))
}
