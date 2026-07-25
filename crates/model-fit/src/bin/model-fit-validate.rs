use anyhow::{Context, Result, bail};
use mesh_llm_gpu_bench::DecodeKernelProbe;
use mesh_llm_system::hardware::HardwareSurvey;
use model_artifact::{ModelFormat, ResolvedModelArtifact, resolve_model_artifact_ref};
use model_fit::{
    AcceleratorKind, BackendKind, CpuProfile, FitStatus, GpuBenchmarkAcceleratorFacts,
    GpuBenchmarkHardwareInput, GpuBenchmarkOutput, HardwareProfile, MemoryProfile, ModelProfile,
    ModelRecommendation, SelectionConfig, TensorTypeBytes, WorkloadProfile,
    hardware_profile_from_gpu_benchmark, profile_gguf_path, score_model,
    score_model_for_context_tokens, throughput_sample_stats,
};
use model_hf::{HfModelRepository, ModelDownloadProgress, ModelDownloadProgressEvent};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering as AtomicOrdering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const DEFAULT_CTX_SIZE: u32 = 8192;
const DEFAULT_WARMUP_TOKENS: usize = 16;
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_REMEASURE_REPEATS: usize = 3;
const DEFAULT_REMEASURE_RAW_SPREAD: f64 = 0.25;
const DEFAULT_REMEASURE_ORDERED_DROP: f64 = 0.20;
const DEFAULT_REMEASURE_PAUSE: Duration = Duration::from_secs(3);
const DEFAULT_CONFIRM_REPEATS: usize = 3;
const DEFAULT_CONFIRM_DELTA: f64 = 0.20;
const DEFAULT_TOLERANCE: f64 = 0.10;
const DEFAULT_MAX_SPREAD: f64 = 0.10;
const DEFAULT_ABI_DECODE_REPEATS: usize = 3;
const DEFAULT_ABI_DECODE_MEASURED_TOKENS: usize = 128;
const FIRST_TOKEN_MAX_NEW_TOKENS: usize = 1;
const KV_WARM_REUSE_MAX_NEW_TOKENS: usize = 16;

#[derive(Clone, Debug)]
struct Args {
    output_json: PathBuf,
    skippy_bench_bin: PathBuf,
    skippy_server_bin: PathBuf,
    metrics_server_bin: PathBuf,
    gpu_benchmark_json: Option<PathBuf>,
    model_files: Vec<PathBuf>,
    benchmark_scenarios: Vec<String>,
    base_port: u16,
    benchmark_all: bool,
    fit_only: bool,
    dense_probe_depth: DenseProbeDepth,
    show_progress: bool,
    skip_context_aligned_abi: bool,
    allow_debug_validation: bool,
    models: Vec<ModelInput>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DenseProbeDepth {
    Standard,
    Deep,
}

#[derive(Clone, Debug)]
enum ModelInput {
    Ref(String),
    Local(LocalModelInput),
}

#[derive(Clone, Debug)]
struct LocalModelInput {
    model_ref: String,
    gguf_path: PathBuf,
}

#[derive(Debug, Serialize)]
struct ValidationReport {
    schema_version: u32,
    generated_at_unix_secs: u64,
    command: Vec<String>,
    validation_build: ValidationBuild,
    fit_input_contract: FitInputContract,
    hardware_profile: HardwareProfile,
    gpu_benchmark_outputs: Vec<GpuBenchmarkOutput>,
    gpu_benchmark_json: Value,
    selection_config: SelectionConfig,
    validation_config: ValidationConfig,
    models: Vec<ModelValidationReport>,
    summary: ValidationSummary,
}

#[derive(Debug, Serialize)]
struct ValidationBuild {
    package_version: &'static str,
    profile: &'static str,
    debug_assertions: bool,
    allow_debug_validation: bool,
    benchmark_binary_warnings: Vec<String>,
    note: &'static str,
}

#[derive(Debug, Serialize)]
struct FitInputContract {
    hardware_fields_consumed: Vec<&'static str>,
    model_fields_consumed: Vec<&'static str>,
    validation_backend: &'static str,
    validation_note: &'static str,
}

#[derive(Debug, Serialize)]
struct ValidationConfig {
    ctx_size: u32,
    warmup_tokens: usize,
    max_new_tokens: usize,
    repeats: usize,
    tolerance: f64,
    max_spread: f64,
    remeasure_repeats: usize,
    remeasure_raw_spread: f64,
    remeasure_ordered_drop: f64,
    confirm_repeats: usize,
    confirm_delta: f64,
    abi_decode_repeats: usize,
    abi_decode_measured_tokens: usize,
    benchmark_all: bool,
    fit_only: bool,
    dense_probe_depth: &'static str,
    show_progress: bool,
    skip_context_aligned_abi: bool,
    prompt: String,
    primary_workload: String,
    scored_workloads: Vec<String>,
    benchmark_scenarios: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct ModelValidationReport {
    input_ref: String,
    resolved_ref: Option<String>,
    artifact: Option<ResolvedModelArtifact>,
    downloaded_paths: Vec<PathBuf>,
    primary_gguf_path: Option<PathBuf>,
    model_profile: Option<ModelProfile>,
    recommendation: Option<ModelRecommendation>,
    fit_interpretation: Option<FitInterpretation>,
    runtime_diagnostic: Option<RuntimeDiagnostic>,
    recommendations: Vec<WorkloadRecommendation>,
    abi_decode_probe: Option<AbiDecodeProbeSummary>,
    context_aligned_abi_decode_probe: Option<AbiDecodeProbeSummary>,
    decode_probe_diagnostic: Option<DecodeProbeDiagnostic>,
    graph_inventory_diagnostic: Option<GraphInventoryDiagnostic>,
    operation_bucket_diagnostic: Option<OperationBucketDiagnostic>,
    model_specific_decode_kernel_probes: Vec<DecodeKernelProbe>,
    model_specific_probe_errors: Vec<String>,
    benchmarks: Vec<BenchmarkScenarioSummary>,
    benchmark: BenchmarkSummary,
    errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FitInterpretation {
    local_accelerated_fit: bool,
    single_node_validation_allowed: bool,
    summary: String,
    details: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct RuntimeDiagnostic {
    validation_shape: &'static str,
    selected_backend: String,
    selected_accelerator: Option<String>,
    layer_start: u32,
    layer_end: Option<u32>,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: &'static str,
    cache_type_v: &'static str,
    flash_attn_type: &'static str,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    load_mode: &'static str,
    filter_tensors_on_load: bool,
    include_embeddings: bool,
    include_output: bool,
    steady_decode_command: Option<Vec<String>>,
    notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbiDecodeProbeKind {
    Standard,
    ContextAligned,
}

#[derive(Clone, Debug, Serialize)]
struct AbiDecodeProbeSummary {
    attempted: bool,
    skip_reason: Option<String>,
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    non_eval_overhead_pct: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    logits_ready_ms: Option<f64>,
    logits_scan_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
    command: Vec<String>,
    observations: Vec<AbiDecodeProbeObservation>,
    sample_count: usize,
    raw_sample_count: usize,
    min_tokens_per_second: Option<f64>,
    max_tokens_per_second: Option<f64>,
    spread_pct: Option<f64>,
    raw_spread_pct: Option<f64>,
    denoised_outlier_count: usize,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct AbiGraphInventoryBucket {
    family: Option<String>,
    ggml_op: Option<i64>,
    ggml_type: Option<u64>,
    node_count: Option<u64>,
    element_count: Option<u64>,
    output_bytes: Option<u64>,
    src0_bytes: Option<u64>,
    src1_bytes: Option<u64>,
    ne: Vec<i64>,
}

#[derive(Clone, Debug, Serialize)]
struct AbiDecodeProbeObservation {
    repeat: usize,
    command: Vec<String>,
    status_code: Option<i32>,
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    logits_ready_ms: Option<f64>,
    logits_scan_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
    stderr_tail: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct DecodeProbeDiagnostic {
    predicted_tokens_per_second: Option<f64>,
    scenario_predicted_tokens_per_second: Option<f64>,
    abi_tokens_per_second: Option<f64>,
    observed_tokens_per_second: Option<f64>,
    observed_over_fit: Option<f64>,
    observed_over_scenario_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    abi_over_scenario_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    scenario_prediction_source: Option<&'static str>,
    observed_vs_fit: String,
    observed_vs_scenario_fit: String,
    abi_vs_fit: String,
    abi_vs_scenario_fit: String,
    observed_vs_abi: String,
    predicted_decode_submission_ms_per_token: Option<f64>,
    abi_decode_call_ms_per_token: Option<f64>,
    decode_submission_residual_ms_per_token: Option<f64>,
    decode_submission_residual_share_of_predicted: Option<f64>,
    predicted_sampler_sync_ms_per_token: Option<f64>,
    abi_sampling_ms_per_token: Option<f64>,
    abi_logits_ready_ms_per_token: Option<f64>,
    abi_logits_scan_ms_per_token: Option<f64>,
    abi_sampling_over_selected_fit: Option<f64>,
    sampler_sync_residual_ms_per_token: Option<f64>,
    sampler_sync_residual_share_of_predicted: Option<f64>,
    selected_full_token_handoff_probe: bool,
    selected_full_token_source_sampled_probe: bool,
    selected_fit_probe_count: usize,
    selected_fit_probe_max_spread_pct: Option<f64>,
    classification: String,
    notes: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct DecodeProbeClassificationInput {
    predicted: Option<f64>,
    abi: Option<f64>,
    observed: Option<f64>,
    missing_representative_model_probe: bool,
    abi_probe_noisy: bool,
    request_decode_noisy: bool,
    request_spread_pct: Option<f64>,
    observed_over_fit: Option<f64>,
    observed_over_scenario_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    abi_over_scenario_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    abi_sampling_over_selected_fit: Option<f64>,
    selected_full_token_handoff_probe: bool,
    selected_full_token_source_sampled_probe: bool,
    decode_submission_residual_share_of_predicted: Option<f64>,
    sampler_sync_residual_share_of_predicted: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct DecodeProbeNotesInput<'a> {
    observed_over_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    decode_submission_residual_ms_per_token: Option<f64>,
    abi_decode_call_ms_per_token: Option<f64>,
    sampler_sync_residual_ms_per_token: Option<f64>,
    abi_sampling_over_selected_fit: Option<f64>,
    abi_logits_ready_ms_per_token: Option<f64>,
    abi_logits_scan_ms_per_token: Option<f64>,
    request_spread_pct: Option<f64>,
    selected_fit_probe_max_spread_pct: Option<f64>,
    selected_full_token_source_sampled_probe: bool,
    classification: &'a str,
}

#[derive(Clone, Debug, Serialize)]
struct GraphInventoryDiagnostic {
    available: bool,
    status: String,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    selected_transformer_probe: Option<String>,
    selected_transformer_probe_layers: Option<u32>,
    selected_probe_context_tokens: Option<u32>,
    abi_graph_context_tokens: Option<u32>,
    selected_probe_node_count: Option<u64>,
    selected_probe_nodes_over_abi: Option<f64>,
    selected_probe_inventory_bucket_mismatch_count: Option<u64>,
    selected_probe_inventory_abs_node_delta: Option<u64>,
    selected_probe_inventory_abs_src0_delta_bytes: Option<u64>,
    selected_probe_inventory_abs_src1_delta_bytes: Option<u64>,
    selected_probe_inventory_abs_output_delta_bytes: Option<u64>,
    metadata_transformer_matmul_nodes: u64,
    graph_transformer_matmul_nodes: u64,
    metadata_transformer_weight_bytes: u64,
    graph_transformer_weight_src0_bytes: u64,
    graph_unclassified_matmul_src0_bytes: u64,
    graph_transformer_src0_over_metadata: Option<f64>,
    graph_transformer_plus_unclassified_src0_over_metadata: Option<f64>,
    estimated_transformer_block_ms: Option<f64>,
    abi_ms_per_token: Option<f64>,
    estimated_transformer_over_abi: Option<f64>,
    comparisons: Vec<GraphInventoryComparison>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GraphInventoryComparison {
    name: &'static str,
    metadata_weight_bytes: u64,
    metadata_node_count: u64,
    graph_weight_src0_bytes: u64,
    graph_node_count: u64,
    src0_over_metadata: Option<f64>,
    node_count_delta: i64,
}

#[derive(Clone, Copy, Debug, Default)]
struct SelectedProbeInventoryDelta {
    bucket_mismatch_count: u64,
    abs_node_delta: u64,
    abs_src0_delta_bytes: u64,
    abs_src1_delta_bytes: u64,
    abs_output_delta_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct GraphInventoryNotesInput {
    metadata_transformer_weight_bytes: u64,
    graph_transformer_weight_src0_bytes: u64,
    graph_unclassified_matmul_src0_bytes: u64,
    metadata_transformer_matmul_nodes: u64,
    graph_transformer_matmul_nodes: u64,
    selected_probe_layers: Option<u32>,
    estimated_transformer_over_abi: Option<f64>,
    selected_probe_nodes_over_abi: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct OperationBucketDiagnostic {
    available: bool,
    estimated_selected_ms_per_token: Option<f64>,
    abi_ms_per_token: Option<f64>,
    estimated_over_abi: Option<f64>,
    buckets: Vec<OperationBucketRow>,
    raw_graph_families: Vec<GraphOperationFamilyRow>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct OperationBucketRow {
    bucket: &'static str,
    source: String,
    graph_families: Vec<&'static str>,
    estimated_ms: Option<f64>,
    estimated_traffic_bytes: u64,
    metadata_weight_bytes: u64,
    graph_node_count: u64,
    graph_src0_bytes: u64,
    graph_src1_bytes: u64,
    graph_output_bytes: u64,
    graph_src0_over_metadata: Option<f64>,
    estimated_share_of_selected_ms: Option<f64>,
    notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct GraphOperationFamilyRow {
    family: String,
    node_count: u64,
    src0_bytes: u64,
    src1_bytes: u64,
    output_bytes: u64,
    element_count: u64,
}

#[derive(Clone, Copy, Debug)]
struct OperationBucketSpec {
    bucket: &'static str,
    graph_families: &'static [&'static str],
    cost_group: &'static str,
    metadata_weight_bytes: u64,
    note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct WorkloadRecommendation {
    workload: String,
    recommendation: ModelRecommendation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchmarkScenarioKind {
    // Sustained one-token-at-a-time generation. This is the scenario model-fit
    // is currently best at predicting because it is closest to llama.cpp's
    // memory-bandwidth-bound decode loop.
    SteadyDecode,
    // Sustained decode after a prompt shaped to the primary recommendation's
    // context. The short `steady_decode` scenario is useful for isolating the
    // decode loop with cheap validation runs, but it cannot prove a primary
    // chat/workload estimate that charges thousands of active KV tokens. This
    // scenario deliberately creates a long synthetic prompt and then uses the
    // tokenizer count reported by Skippy for comparison, so the estimator still
    // sees only metadata + measured hardware and the benchmark only supplies
    // the observed sequence shape being validated.
    PrimaryContextSteadyDecode,
    // Prompt ingestion only, measured as prompt tokens divided by Skippy's
    // `prefill_elapsed_ms`. This is intentionally separate from first-token
    // latency so a miss can be attributed to prefill matmul throughput rather
    // than request setup or the first decode step after prefill.
    Prefill,
    // End-to-end request latency for a long prompt and one generated token.
    // Lower is better, so verdict labels are inverted after the generic ratio
    // check. This scenario is a user-visible latency target, not a pure decode
    // or pure prefill micro-benchmark.
    FirstToken,
    // Short repeated generation with session reuse. This gives us a small
    // signal for agent/tool loops where the same prefix remains resident.
    KvWarmReuse,
}

impl BenchmarkScenarioKind {
    fn is_steady_decode(self) -> bool {
        matches!(
            self,
            BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::PrimaryContextSteadyDecode
        )
    }

    fn uses_decode_tps_prediction(self) -> bool {
        matches!(
            self,
            BenchmarkScenarioKind::SteadyDecode
                | BenchmarkScenarioKind::PrimaryContextSteadyDecode
                | BenchmarkScenarioKind::KvWarmReuse
        )
    }
}

#[derive(Clone, Debug)]
struct BenchmarkScenarioSpec {
    kind: BenchmarkScenarioKind,
    name: &'static str,
    fit_metric: &'static str,
    prompt: String,
    ctx_size: u32,
    max_new_tokens: usize,
    warmup_tokens: usize,
    request_count: usize,
    reuse_session: bool,
}

#[derive(Clone, Copy, Debug)]
struct BenchmarkExpected {
    predicted: Option<f64>,
    range: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
struct PromptLengthCandidate {
    sequence: usize,
    word_count: u32,
    prompt: String,
    prompt_tokens: u32,
    requested_tokens: u32,
    fits_context: bool,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkScenarioSummary {
    scenario: String,
    fit_metric: String,
    predicted: Option<f64>,
    predicted_range: Option<(f64, f64)>,
    prediction_source: &'static str,
    primary_predicted: Option<f64>,
    primary_predicted_range: Option<(f64, f64)>,
    primary_observed_over_fit: Option<f64>,
    prediction_context_tokens: Option<u32>,
    prediction_decode_cost_breakdown: Option<model_fit::DecodeCostBreakdown>,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    first_token_breakdown: Option<FirstTokenBreakdown>,
    verdict: String,
    benchmark: BenchmarkSummary,
}

#[derive(Clone, Debug, Serialize)]
struct FirstTokenBreakdown {
    prompt_token_count: Option<u64>,
    tokenizer_vocab_size: Option<u32>,
    chat_template_available: bool,
    predicted_prefill_ms: Option<f64>,
    predicted_decode_ms: Option<f64>,
    predicted_overhead_ms: Option<f64>,
    predicted_sampler_ms: Option<f64>,
    predicted_sampled_decode_ms: Option<f64>,
    observed_tokenize_ms: Option<f64>,
    observed_prefill_ms: Option<f64>,
    observed_decode_ms: Option<f64>,
    observed_sampled_decode_residual_ms: Option<f64>,
    observed_sampled_decode_residual_us_per_prompt_token: Option<f64>,
    observed_unattributed_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct BenchmarkSummary {
    attempted: bool,
    skip_reason: Option<String>,
    observations: Vec<BenchmarkObservation>,
    successful_repeats: usize,
    sample_count: usize,
    raw_sample_count: usize,
    // Historical field name: for throughput scenarios this is tokens/sec; for
    // first-token latency it stores milliseconds so the same denoising and
    // spread machinery can be reused. The scenario wrapper exposes the actual
    // metric name through `fit_metric`, and Markdown rendering labels the value
    // generically as predicted/observed.
    median_tokens_per_sec: Option<f64>,
    min_tokens_per_sec: Option<f64>,
    max_tokens_per_sec: Option<f64>,
    spread_pct: Option<f64>,
    raw_median_tokens_per_sec: Option<f64>,
    raw_min_tokens_per_sec: Option<f64>,
    raw_max_tokens_per_sec: Option<f64>,
    raw_spread_pct: Option<f64>,
    request_sample_count: usize,
    request_median_tokens_per_sec: Option<f64>,
    request_min_tokens_per_sec: Option<f64>,
    request_max_tokens_per_sec: Option<f64>,
    request_spread_pct: Option<f64>,
    denoised_outlier_count: usize,
    remeasured: bool,
    remeasure_reason: Option<String>,
    initial_observations: Vec<BenchmarkObservation>,
    rejected_remeasure_observations: Vec<BenchmarkObservation>,
    observed_over_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    verdict: String,
    errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkObservation {
    repeat: usize,
    run_id: String,
    command: Vec<String>,
    status_code: Option<i32>,
    wall_seconds: f64,
    prompt_token_count: Option<u64>,
    generated_tokens_per_sec: Option<f64>,
    generated_token_count: Option<u64>,
    text_request_elapsed_ms: Option<f64>,
    request_count: Option<u64>,
    reuse_session: Option<bool>,
    request_results: Vec<BenchmarkRequestObservation>,
    stdout_json_path: Option<PathBuf>,
    report_json_path: PathBuf,
    stderr_tail: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct BenchmarkRequestObservation {
    request_id: Option<String>,
    session_id: Option<String>,
    elapsed_ms: Option<f64>,
    tokenize_elapsed_ms: Option<f64>,
    prefill_elapsed_ms: Option<f64>,
    decode_elapsed_ms: Option<f64>,
    prompt_token_count: Option<u64>,
    generated_token_count: Option<u64>,
    generated_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
}

#[derive(Debug, Default, Serialize)]
struct ValidationSummary {
    model_count: usize,
    benchmarked_count: usize,
    matched_count: usize,
    slower_than_fit_count: usize,
    faster_than_fit_count: usize,
    metadata_estimate_miss_count: usize,
    runtime_path_mismatch_count: usize,
    probe_mismatch_count: usize,
    noisy_count: usize,
    skipped_count: usize,
    error_count: usize,
    runtime_error_count: usize,
    median_observed_over_fit: Option<f64>,
    mean_observed_over_fit: Option<f64>,
    median_absolute_percent_error: Option<f64>,
    within_tolerance_count: usize,
    scenario_summaries: Vec<ScenarioValidationSummary>,
}

#[derive(Debug, Default, Serialize)]
struct ScenarioValidationSummary {
    scenario: String,
    sample_count: usize,
    matched_count: usize,
    slower_than_fit_count: usize,
    faster_than_fit_count: usize,
    metadata_estimate_miss_count: usize,
    runtime_path_mismatch_count: usize,
    probe_mismatch_count: usize,
    noisy_count: usize,
    skipped_count: usize,
    error_count: usize,
    runtime_error_count: usize,
    within_tolerance_count: usize,
    median_observed_over_fit: Option<f64>,
    mean_observed_over_fit: Option<f64>,
    median_absolute_percent_error: Option<f64>,
}

struct PreparedModel {
    input_ref: String,
    resolved_ref: Option<String>,
    artifact: Option<ResolvedModelArtifact>,
    downloaded_paths: Vec<PathBuf>,
    primary_gguf_path: PathBuf,
    profile: ModelProfile,
}

struct LoadedHardware {
    profile: HardwareProfile,
    benchmark_outputs: Vec<GpuBenchmarkOutput>,
    raw_json: Value,
}

struct LocalGpuBenchmark {
    outputs: Vec<GpuBenchmarkOutput>,
    backend: BackendKind,
}

#[tokio::main]
async fn main() -> Result<()> {
    let raw_args = std::env::args().collect::<Vec<_>>();
    if raw_args.get(1).is_some_and(|arg| arg == "__native-probe") {
        return run_native_probe_child(&raw_args[2..]);
    }

    let args = Args::parse()?;
    enforce_validation_build(&args)?;
    let hardware = load_hardware_profile(&args)?;
    let selection_config = selection_config(&primary_workload_profile());
    let repository = HfModelRepository::from_env().context("create Hugging Face repository")?;

    let mut models = Vec::new();
    for (index, input) in args.models.iter().enumerate() {
        let report = validate_model(&args, &repository, &hardware.profile, input, index).await;
        models.push(report);
        let partial_report = build_validation_report(&args, &hardware, &selection_config, &models);
        write_json_report(&args.output_json, &partial_report)?;
    }

    let report = build_validation_report(&args, &hardware, &selection_config, &models);
    write_json_report(&args.output_json, &report)?;
    print_markdown_table(&report.models);
    eprintln!("wrote {}", args.output_json.display());
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum NativeProbeKind {
    FullToken,
    FullTokenHandoff,
    DecodeSubmission,
    SourceSampledToken,
}

impl NativeProbeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::FullToken => "dense-full-token",
            Self::FullTokenHandoff => "dense-full-token-handoff",
            Self::DecodeSubmission => "dense-decode-submission",
            Self::SourceSampledToken => "dense-source-sampled-token",
        }
    }

    fn parse(value: &str) -> Result<Self> {
        match value {
            "dense-full-token" => Ok(Self::FullToken),
            "dense-full-token-handoff" => Ok(Self::FullTokenHandoff),
            "dense-decode-submission" => Ok(Self::DecodeSubmission),
            "dense-source-sampled-token" => Ok(Self::SourceSampledToken),
            other => bail!("unknown native probe kind {other}"),
        }
    }
}

fn run_native_probe_child(args: &[String]) -> Result<()> {
    let request = NativeProbeRequest::parse(args)?;
    let probes = request.run()?;
    serde_json::to_writer(std::io::stdout(), &probes).context("write native probe JSON")?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct NativeProbeRequest<'a> {
    kind: NativeProbeKind,
    backend: mesh_llm_gpu_bench::BenchmarkBackend,
    block_tensor_type: &'a str,
    output_tensor_type: &'a str,
    shape: mesh_llm_gpu_bench::DenseFullTokenProbeShape,
}

impl<'a> NativeProbeRequest<'a> {
    fn parse(args: &'a [String]) -> Result<Self> {
        if args.len() != 16 {
            bail!(
                "native probe expects 16 args: kind backend block_tensor_type output_tensor_type hidden kv_width ffn vocab repeat_layers graph_features norm_head_width head_dim query_heads kv_heads context_tokens active_context_tokens"
            );
        }
        let kind = NativeProbeKind::parse(&args[0])?;
        let backend = parse_native_probe_backend(&args[1])?;
        Ok(Self {
            kind,
            backend,
            block_tensor_type: &args[2],
            output_tensor_type: &args[3],
            shape: mesh_llm_gpu_bench::DenseFullTokenProbeShape {
                hidden: parse_native_probe_u32(&args[4], "hidden")?,
                kv_width: parse_native_probe_u32(&args[5], "kv_width")?,
                ffn: parse_native_probe_u32(&args[6], "ffn")?,
                vocab: parse_native_probe_u32(&args[7], "vocab")?,
                repeat_layers: parse_native_probe_u32(&args[8], "repeat_layers")?,
                graph_features: parse_native_probe_u32(&args[9], "graph_features")?,
                norm_head_width: parse_native_probe_u32(&args[10], "norm_head_width")?,
                head_dim: parse_native_probe_u32(&args[11], "head_dim")?,
                query_heads: parse_native_probe_u32(&args[12], "query_heads")?,
                kv_heads: parse_native_probe_u32(&args[13], "kv_heads")?,
                context_tokens: parse_native_probe_u32(&args[14], "context_tokens")?,
                active_context_tokens: parse_native_probe_u32(&args[15], "active_context_tokens")?,
            },
        })
    }

    fn run(self) -> Result<Vec<DecodeKernelProbe>> {
        match self.kind {
            NativeProbeKind::FullToken => mesh_llm_gpu_bench::run_model_dense_full_token_probe(
                self.backend,
                self.block_tensor_type,
                self.output_tensor_type,
                self.shape,
            ),
            NativeProbeKind::FullTokenHandoff => {
                mesh_llm_gpu_bench::run_model_dense_full_token_handoff_probe(
                    self.backend,
                    self.block_tensor_type,
                    self.output_tensor_type,
                    self.shape,
                )
            }
            NativeProbeKind::DecodeSubmission => {
                mesh_llm_gpu_bench::run_model_dense_decode_submission_probe(
                    self.backend,
                    self.block_tensor_type,
                    self.output_tensor_type,
                    self.shape,
                )
            }
            NativeProbeKind::SourceSampledToken => {
                mesh_llm_gpu_bench::run_model_dense_source_sampled_token_probe(
                    self.backend,
                    self.block_tensor_type,
                    self.output_tensor_type,
                    self.shape,
                )
            }
        }
    }
}

fn parse_native_probe_backend(value: &str) -> Result<mesh_llm_gpu_bench::BenchmarkBackend> {
    match value {
        "metal" => Ok(mesh_llm_gpu_bench::BenchmarkBackend::Metal),
        "cuda" => Ok(mesh_llm_gpu_bench::BenchmarkBackend::Cuda),
        "hip" => Ok(mesh_llm_gpu_bench::BenchmarkBackend::Hip),
        other => bail!("unsupported native probe backend {other}"),
    }
}

fn native_probe_backend_arg(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> &'static str {
    match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => "metal",
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => "cuda",
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => "hip",
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => "intel",
    }
}

fn parse_native_probe_u32(value: &str, name: &str) -> Result<u32> {
    value
        .parse::<u32>()
        .with_context(|| format!("parse native probe {name}={value}"))
}

fn run_native_probe_isolated(
    kind: NativeProbeKind,
    backend: mesh_llm_gpu_bench::BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: mesh_llm_gpu_bench::DenseFullTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    // Some GGML backends can accept a graph through `supports_op()` and still
    // abort the process when a source-shaped synthetic probe reaches a concrete
    // kernel. CUDA Flash Attention has produced that failure mode for a valid
    // small-model GQA shape. The fitter must not turn a backend abort into a
    // validator abort: a crashing probe means "no safe probe evidence for this
    // shape", not "the model is bad" and not "fall back to observed tok/s".
    //
    // Keep isolation at the probe boundary instead of hardcoding backend,
    // architecture, or model-family exceptions. The child receives only the
    // metadata-derived shape and measured backend kind, writes probe JSON on
    // success, and may die without taking the parent validation report with it.
    let output = Command::new(std::env::current_exe().context("locate current validator binary")?)
        .arg("__native-probe")
        .arg(kind.as_str())
        .arg(native_probe_backend_arg(backend))
        .arg(block_tensor_type)
        .arg(output_tensor_type)
        .arg(shape.hidden.to_string())
        .arg(shape.kv_width.to_string())
        .arg(shape.ffn.to_string())
        .arg(shape.vocab.to_string())
        .arg(shape.repeat_layers.to_string())
        .arg(shape.graph_features.to_string())
        .arg(shape.norm_head_width.to_string())
        .arg(shape.head_dim.to_string())
        .arg(shape.query_heads.to_string())
        .arg(shape.kv_heads.to_string())
        .arg(shape.context_tokens.to_string())
        .arg(shape.active_context_tokens.to_string())
        .output()
        .with_context(|| format!("run isolated native probe {}", kind.as_str()))?;
    if !output.status.success() {
        match parse_isolated_native_probe_stdout(kind, &output.stdout) {
            Ok(probes) if !probes.is_empty() => return Ok(probes),
            Ok(_) | Err(_) => {}
        }
        let stderr = diagnostic_prefix(&String::from_utf8_lossy(&output.stderr), 2048);
        bail!(
            "isolated native probe {} exited with status {}; stderr prefix: {}",
            kind.as_str(),
            output.status,
            stderr.trim()
        );
    }
    parse_isolated_native_probe_stdout(kind, &output.stdout)
}

fn parse_isolated_native_probe_stdout(
    kind: NativeProbeKind,
    stdout: &[u8],
) -> Result<Vec<DecodeKernelProbe>> {
    // Some backends, notably Metal with residency-set lifetime checks, can
    // abort during process teardown after the probe has already completed and
    // written valid JSON. The isolated child exists specifically so backend
    // cleanup failures do not take down validation. Accept complete JSON from
    // stdout, but do not invent probe evidence when stdout is empty, partial, or
    // malformed.
    serde_json::from_slice(stdout).with_context(|| {
        let stdout = String::from_utf8_lossy(stdout)
            .chars()
            .take(512)
            .collect::<String>();
        format!(
            "parse isolated native probe {} JSON; stdout prefix: {stdout}",
            kind.as_str()
        )
    })
}

fn diagnostic_prefix(text: &str, max_chars: usize) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_chars)
        .collect()
}

fn build_validation_report(
    args: &Args,
    hardware: &LoadedHardware,
    selection_config: &SelectionConfig,
    models: &[ModelValidationReport],
) -> ValidationReport {
    let summary = summarize(args, models, DEFAULT_TOLERANCE);
    ValidationReport {
        schema_version: 1,
        generated_at_unix_secs: unix_timestamp_secs(),
        command: std::env::args().collect(),
        validation_build: validation_build(args),
        fit_input_contract: fit_input_contract(),
        hardware_profile: hardware.profile.clone(),
        gpu_benchmark_outputs: hardware.benchmark_outputs.clone(),
        gpu_benchmark_json: hardware.raw_json.clone(),
        selection_config: selection_config.clone(),
        validation_config: ValidationConfig::from_args(args),
        models: models.to_vec(),
        summary,
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = std::env::args().skip(1);
        let mut parsed = Self {
            output_json: default_output_json_path(),
            skippy_bench_bin: default_binary_path("skippy-bench"),
            skippy_server_bin: default_binary_path("skippy-server"),
            metrics_server_bin: default_binary_path("metrics-server"),
            gpu_benchmark_json: None,
            model_files: Vec::new(),
            benchmark_scenarios: Vec::new(),
            base_port: 18400,
            benchmark_all: false,
            fit_only: false,
            dense_probe_depth: DenseProbeDepth::Standard,
            show_progress: true,
            skip_context_aligned_abi: false,
            allow_debug_validation: false,
            models: Vec::new(),
        };

        while let Some(arg) = values.next() {
            parsed.parse_arg(arg, &mut values)?;
        }
        parsed.load_model_files()?;
        parsed.validate_benchmark_scenarios()?;

        if parsed.models.is_empty() {
            bail!("provide at least one model ref");
        }
        Ok(parsed)
    }

    fn parse_arg(&mut self, arg: String, values: &mut impl Iterator<Item = String>) -> Result<()> {
        match arg.as_str() {
            "--output-json" => {
                self.output_json = PathBuf::from(next_value(values, "--output-json")?)
            }
            "--skippy-bench-bin" => {
                self.skippy_bench_bin = PathBuf::from(next_value(values, "--skippy-bench-bin")?);
            }
            "--skippy-server-bin" => {
                self.skippy_server_bin = PathBuf::from(next_value(values, "--skippy-server-bin")?);
            }
            "--metrics-server-bin" => {
                self.metrics_server_bin =
                    PathBuf::from(next_value(values, "--metrics-server-bin")?);
            }
            "--gpu-benchmark-json" => {
                self.gpu_benchmark_json =
                    Some(PathBuf::from(next_value(values, "--gpu-benchmark-json")?));
            }
            "--models-file" => {
                self.model_files
                    .push(PathBuf::from(next_value(values, "--models-file")?));
            }
            "--scenario" => self
                .benchmark_scenarios
                .push(next_value(values, "--scenario")?),
            "--scenarios" => {
                self.benchmark_scenarios.extend(
                    next_value(values, "--scenarios")?
                        .split(',')
                        .map(str::trim)
                        .filter(|value| !value.is_empty())
                        .map(str::to_string),
                );
            }
            "--base-port" => self.base_port = parse_next(values, "--base-port")?,
            "--benchmark-all" => self.benchmark_all = true,
            "--fit-only" => self.fit_only = true,
            "--dense-probe-depth" => {
                self.dense_probe_depth =
                    DenseProbeDepth::parse(&next_value(values, "--dense-probe-depth")?)?;
            }
            "--no-progress" => self.show_progress = false,
            "--skip-context-aligned-abi" => self.skip_context_aligned_abi = true,
            "--allow-debug-validation" => self.allow_debug_validation = true,
            "--model-ref" => self
                .models
                .push(ModelInput::Ref(next_value(values, "--model-ref")?)),
            "--model" => {
                self.models
                    .push(ModelInput::Local(parse_local_model(&next_value(
                        values, "--model",
                    )?)?));
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other if other.starts_with('-') => bail!("unknown argument {other}"),
            model_ref => self.models.push(ModelInput::Ref(model_ref.to_string())),
        }
        Ok(())
    }

    fn load_model_files(&mut self) -> Result<()> {
        for path in &self.model_files {
            let contents = fs::read_to_string(path)
                .with_context(|| format!("read model manifest {}", path.display()))?;
            for (line_index, line) in contents.lines().enumerate() {
                let Some(model_ref) = parse_model_manifest_line(line) else {
                    continue;
                };
                if model_ref.contains('=') {
                    bail!(
                        "invalid model manifest entry {}:{}: key/value metadata is not supported yet",
                        path.display(),
                        line_index + 1
                    );
                }
                self.models.push(ModelInput::Ref(model_ref.to_string()));
            }
        }
        Ok(())
    }

    fn validate_benchmark_scenarios(&self) -> Result<()> {
        if self.benchmark_scenarios.is_empty()
            || self
                .benchmark_scenarios
                .iter()
                .any(|scenario| scenario == "all")
        {
            return Ok(());
        }
        let valid = benchmark_scenarios()
            .into_iter()
            .map(|scenario| scenario.name)
            .collect::<Vec<_>>();
        for requested in &self.benchmark_scenarios {
            if !valid.contains(&requested.as_str()) {
                bail!(
                    "unknown benchmark scenario {requested}; valid scenarios: {}",
                    valid.join(", ")
                );
            }
        }
        Ok(())
    }
}

fn enforce_validation_build(args: &Args) -> Result<()> {
    if cfg!(debug_assertions) && !args.allow_debug_validation {
        bail!(
            "model-fit-validate must be run from a release build for performance validation; \
             debug builds change Rust-side hardware/probe timings and can produce false tok/s \
             evidence. Rebuild with `just model-fit-release ...` or pass \
             --allow-debug-validation for local development only."
        );
    }
    for warning in benchmark_binary_warnings(args) {
        if !args.allow_debug_validation {
            bail!(
                "{warning}. Use release benchmark binaries or pass --allow-debug-validation for local development only."
            );
        }
        eprintln!("warning: {warning}; continuing because --allow-debug-validation was set");
    }
    if cfg!(debug_assertions) {
        eprintln!(
            "warning: debug validation enabled by override; do not use this report as tok/s evidence"
        );
    }
    Ok(())
}

fn validation_build(args: &Args) -> ValidationBuild {
    ValidationBuild {
        package_version: env!("CARGO_PKG_VERSION"),
        profile: build_profile_label(),
        debug_assertions: cfg!(debug_assertions),
        allow_debug_validation: args.allow_debug_validation,
        benchmark_binary_warnings: benchmark_binary_warnings(args),
        note: "model-fit validation uses runtime probe timings; release builds are required for tok/s evidence because debug Rust code changes measured overheads.",
    }
}

fn build_profile_label() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn benchmark_binary_warnings(args: &Args) -> Vec<String> {
    if args.fit_only {
        return Vec::new();
    }
    [
        ("skippy-bench", &args.skippy_bench_bin),
        ("skippy-server", &args.skippy_server_bin),
        ("metrics-server", &args.metrics_server_bin),
    ]
    .into_iter()
    .filter(|(_, path)| path_looks_like_debug_target(path))
    .map(|(name, path)| {
        format!(
            "{name} path {} appears to be a target/debug binary, which is not valid tok/s evidence",
            path.display()
        )
    })
    .collect()
}

fn path_looks_like_debug_target(path: &Path) -> bool {
    let mut saw_target = false;
    for component in path
        .components()
        .filter_map(|component| component.as_os_str().to_str())
    {
        if saw_target && component == "debug" {
            return true;
        }
        saw_target = component == "target";
    }
    false
}

impl DenseProbeDepth {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "standard" => Ok(Self::Standard),
            "deep" => Ok(Self::Deep),
            other => bail!("unknown dense probe depth {other}; valid values: standard, deep"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Deep => "deep",
        }
    }
}

fn default_output_json_path() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("tmp").join("model-fit-validation.json"))
        .unwrap_or_else(|| PathBuf::from("model-fit-validation.json"))
}

fn parse_model_manifest_line(line: &str) -> Option<&str> {
    let without_comment = line.split_once('#').map_or(line, |(value, _)| value);
    let trimmed = without_comment.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn default_binary_path(name: &str) -> PathBuf {
    let release = PathBuf::from(format!("target/release/{name}"));
    if release.exists() {
        release
    } else {
        PathBuf::from(format!("target/debug/{name}"))
    }
}

impl ValidationConfig {
    fn from_args(args: &Args) -> Self {
        let benchmark_scenarios = selected_benchmark_scenarios(args)
            .into_iter()
            .map(|scenario| scenario.name.to_string())
            .collect();
        Self {
            ctx_size: DEFAULT_CTX_SIZE,
            warmup_tokens: DEFAULT_WARMUP_TOKENS,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            repeats: DEFAULT_REPEATS,
            tolerance: DEFAULT_TOLERANCE,
            max_spread: DEFAULT_MAX_SPREAD,
            remeasure_repeats: DEFAULT_REMEASURE_REPEATS,
            remeasure_raw_spread: DEFAULT_REMEASURE_RAW_SPREAD,
            remeasure_ordered_drop: DEFAULT_REMEASURE_ORDERED_DROP,
            confirm_repeats: DEFAULT_CONFIRM_REPEATS,
            confirm_delta: DEFAULT_CONFIRM_DELTA,
            abi_decode_repeats: DEFAULT_ABI_DECODE_REPEATS,
            abi_decode_measured_tokens: DEFAULT_ABI_DECODE_MEASURED_TOKENS,
            benchmark_all: args.benchmark_all,
            fit_only: args.fit_only,
            dense_probe_depth: args.dense_probe_depth.as_str(),
            show_progress: args.show_progress,
            skip_context_aligned_abi: args.skip_context_aligned_abi,
            prompt: validation_prompt().into(),
            primary_workload: primary_workload_label().into(),
            scored_workloads: workload_profiles()
                .iter()
                .map(|(label, _)| (*label).to_string())
                .collect(),
            benchmark_scenarios,
        }
    }
}

async fn validate_model(
    args: &Args,
    repository: &HfModelRepository,
    hardware: &HardwareProfile,
    input: &ModelInput,
    model_index: usize,
) -> ModelValidationReport {
    let label = input_label(input);
    heartbeat(
        Some(model_index),
        &label,
        "model_start",
        "starting validation",
    );
    match prepare_model(args, repository, input, model_index).await {
        Ok(prepared) => {
            heartbeat(
                Some(model_index),
                &prepared.input_ref,
                "model_prepared",
                "metadata profile is ready",
            );
            let report = validate_prepared_model(args, hardware, prepared, model_index);
            heartbeat(
                Some(model_index),
                &report.input_ref,
                "model_done",
                &format!("steady_decode_verdict={}", report.benchmark.verdict),
            );
            report
        }
        Err(err) => {
            let error = format!("{err:#}");
            heartbeat(Some(model_index), &label, "model_error", &error);
            error_report(label, error)
        }
    }
}

fn validate_prepared_model(
    args: &Args,
    hardware: &HardwareProfile,
    prepared: PreparedModel,
    model_index: usize,
) -> ModelValidationReport {
    let model_specific_probes =
        model_specific_decode_kernel_probes(args, hardware, &prepared.profile, model_index);
    let mut hardware_for_model = hardware.clone();
    for accelerator in &mut hardware_for_model.accelerators {
        accelerator
            .decode_kernel_probes
            .extend(model_specific_probes.probes.clone());
    }
    heartbeat(
        Some(model_index),
        &prepared.input_ref,
        "score_start",
        "scoring workload recommendations",
    );
    let recommendations = score_workloads(&hardware_for_model, &prepared.profile);
    let recommendation = recommendations
        .iter()
        .find(|entry| entry.workload == primary_workload_label())
        .map(|entry| entry.recommendation.clone())
        .unwrap_or_else(|| {
            recommendations
                .first()
                .expect("workload list is not empty")
                .recommendation
                .clone()
        });
    heartbeat(
        Some(model_index),
        &prepared.input_ref,
        "score_done",
        &format!(
            "fit_status={:?} selected_backend={:?} selected_accelerator={} decode_tps={}",
            recommendation.fit_status,
            recommendation.selected_backend,
            recommendation
                .selected_accelerator
                .as_deref()
                .unwrap_or("-"),
            display_opt(
                recommendation
                    .estimated_decode_tokens_per_sec
                    .map(f64::from)
            )
        ),
    );
    let mut benchmarks = if args.fit_only {
        Vec::new()
    } else {
        benchmark_model(
            args,
            &hardware_for_model,
            &prepared,
            &recommendation,
            model_index,
        )
    };
    let abi_decode_probe = if args.fit_only {
        None
    } else {
        Some(run_abi_decode_probe_for_recommendation(
            args,
            &prepared,
            &recommendation,
            model_index,
        ))
    };
    apply_observed_over_abi(&mut benchmarks, abi_decode_probe.as_ref());
    let fit_interpretation = Some(fit_interpretation(&recommendation));
    let runtime_diagnostic = Some(runtime_diagnostic(
        &prepared.profile,
        &recommendation,
        &benchmarks,
    ));
    let steady_benchmark = steady_benchmark_for_decode_diagnostic(&benchmarks);
    let benchmark = steady_benchmark
        .map(|benchmark| benchmark.benchmark.clone())
        .unwrap_or_else(|| BenchmarkSummary {
            verdict: "skipped".into(),
            skip_reason: Some("steady_decode scenario was not produced".into()),
            ..BenchmarkSummary::default()
        });
    let decode_probe_diagnostic = decode_probe_diagnostic(
        &recommendation,
        abi_decode_probe.as_ref(),
        steady_benchmark,
        &model_specific_probes.errors,
    );
    let context_aligned_abi_decode_probe = context_aligned_abi_decode_probe(
        args,
        &prepared,
        &recommendation,
        model_index,
        abi_decode_probe.as_ref(),
        decode_probe_diagnostic.as_ref(),
    );
    let graph_inventory_diagnostic = graph_inventory_diagnostic(
        &prepared.profile,
        &recommendation,
        &model_specific_probes.probes,
        context_aligned_abi_decode_probe
            .as_ref()
            .or(abi_decode_probe.as_ref()),
        if context_aligned_abi_decode_probe.is_some() {
            AbiDecodeProbeKind::ContextAligned
        } else {
            AbiDecodeProbeKind::Standard
        },
    );
    let operation_bucket_diagnostic = operation_bucket_diagnostic(
        &prepared.profile,
        &recommendation,
        abi_decode_probe.as_ref(),
    );
    ModelValidationReport {
        input_ref: prepared.input_ref,
        resolved_ref: prepared.resolved_ref,
        artifact: prepared.artifact,
        downloaded_paths: prepared.downloaded_paths,
        primary_gguf_path: Some(prepared.primary_gguf_path),
        model_profile: Some(prepared.profile),
        recommendation: Some(recommendation),
        fit_interpretation,
        runtime_diagnostic,
        recommendations,
        abi_decode_probe,
        context_aligned_abi_decode_probe,
        decode_probe_diagnostic,
        graph_inventory_diagnostic,
        operation_bucket_diagnostic,
        model_specific_decode_kernel_probes: model_specific_probes.probes,
        model_specific_probe_errors: model_specific_probes.errors,
        benchmarks,
        benchmark,
        errors: Vec::new(),
    }
}

#[derive(Clone, Debug, Default)]
struct ModelSpecificDecodeProbes {
    probes: Vec<DecodeKernelProbe>,
    errors: Vec<String>,
}

fn model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let mut collected = if has_recurrent_attention_profile(profile) {
        linear_attention_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
    } else {
        match profile.architecture_class {
            model_fit::ModelArchitectureClass::SparseMoeTransformer => {
                moe_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
            }
            model_fit::ModelArchitectureClass::DenseTransformer => {
                dense_model_specific_decode_kernel_probes(args, hardware, profile, model_index)
            }
            _ => ModelSpecificDecodeProbes::default(),
        }
    };
    if matches!(
        profile.architecture_class,
        model_fit::ModelArchitectureClass::DenseTransformer
            | model_fit::ModelArchitectureClass::SparseMoeTransformer
    ) {
        append_attention_runtime_probes(args, hardware, profile, model_index, &mut collected);
    }
    append_model_output_projection_probes(args, hardware, profile, model_index, &mut collected);
    append_model_logits_readback_probes(args, hardware, profile, model_index, &mut collected);
    collected
}

fn linear_attention_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = linear_attention_graph_probe_plans(profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped linear-attention graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    }
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            heartbeat(
                Some(model_index),
                &model_label,
                "model_linear_attention_probe_start",
                &format!(
                    "backend={:?} tensor_type={} hidden={} qkv={} gate={} state={} out={} ffn={} recurrent_layers={} full_attention_layers={} kv_width={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    plan.hidden,
                    plan.qkv_width,
                    plan.gate_width,
                    plan.state_width,
                    plan.output_input_width,
                    plan.ffn,
                    plan.recurrent_layers,
                    plan.full_attention_layers,
                    plan.kv_width,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped linear attention graph {} {} r{} f{}",
                    model_label,
                    plan.tensor_type,
                    plan.recurrent_layers,
                    plan.full_attention_layers
                ),
            );
            match mesh_llm_gpu_bench::run_model_linear_attention_graph_probe(
                backend,
                plan.tensor_type,
                mesh_llm_gpu_bench::LinearAttentionGraphProbeShape {
                    hidden: plan.hidden,
                    qkv_width: plan.qkv_width,
                    gate_width: plan.gate_width,
                    state_width: plan.state_width,
                    output_input_width: plan.output_input_width,
                    ffn: plan.ffn,
                    recurrent_layers: plan.recurrent_layers,
                    full_attention_layers: plan.full_attention_layers,
                    kv_width: plan.kv_width,
                    graph_features: plan.graph_features,
                    norm_head_width: plan.norm_head_width,
                },
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_linear_attention_probe_done",
                        &format!("tensor_type={} probes={}", plan.tensor_type, probes.len()),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!("tensor_type={}: {error:#}", plan.tensor_type);
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_linear_attention_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
        }
    }
    collected
}

fn dense_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = dense_graph_probe_plans(args, profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped dense graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    }
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            for &repeat_layers in &plan.repeat_layers {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_dense_probe_start",
                    &format!(
                        "backend={:?} tensor_type={} hidden={} kv_width={} ffn={} layers={} graph_features={} norm_head_width={}",
                        accelerator.backend,
                        plan.tensor_type,
                        plan.hidden,
                        plan.kv_width,
                        plan.ffn,
                        repeat_layers,
                        plan.graph_features,
                        plan.norm_head_width,
                    ),
                );
                let _status = TerminalStatus::start(
                    args.show_progress,
                    format!(
                        "Probing model-shaped dense graph {} {} l{} {}x{} f{}",
                        model_label,
                        plan.tensor_type,
                        repeat_layers,
                        plan.ffn,
                        plan.hidden,
                        plan.graph_features
                    ),
                );
                match mesh_llm_gpu_bench::run_model_dense_graph_probe(
                    backend,
                    plan.tensor_type,
                    mesh_llm_gpu_bench::DenseGraphProbeShape {
                        hidden: plan.hidden,
                        kv_width: plan.kv_width,
                        ffn: plan.ffn,
                        repeat_layers,
                        graph_features: plan.graph_features,
                        norm_head_width: plan.norm_head_width,
                    },
                ) {
                    Ok(probes) => {
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_dense_probe_done",
                            &format!(
                                "tensor_type={} layers={} probes={}",
                                plan.tensor_type,
                                repeat_layers,
                                probes.len()
                            ),
                        );
                        collected.probes.extend(probes);
                    }
                    Err(error) => {
                        let message = format!(
                            "tensor_type={} layers={repeat_layers}: {error:#}",
                            plan.tensor_type
                        );
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_dense_probe_error",
                            &message,
                        );
                        collected.errors.push(message);
                    }
                }
            }
            // Do not run the dense sampled-token graph probe in default
            // validation. That synthetic graph tried to combine transformer
            // decode, output projection, logits handoff, and sampling into one
            // backend-specific GGML graph. In practice it is both expensive and
            // weaker evidence than the source-shaped pieces we score with:
            // dense/full-token graph probes for model compute, explicit logits
            // handoff probes, and the hardware-profile sampler probe. Metal can
            // also report invalid-resource failures for large sampled-token
            // shapes while still returning a result, which makes it poor smoke
            // evidence. Keep the low-level probe callable for manual debugging,
            // but do not spend every validation run on a non-scoring signal.
            let Some(model_layers) = profile.layer_count.filter(|count| *count > 0) else {
                continue;
            };
            let Some(vocab) = profile.tokenizer.vocab_size.filter(|vocab| *vocab > 0) else {
                continue;
            };
            let Some(output_tensor_type) = output_projection_probe_tensor_type(profile) else {
                continue;
            };
            let Some(query_heads) = profile.attention_heads.filter(|heads| *heads > 0) else {
                continue;
            };
            let kv_heads = profile.kv_heads.unwrap_or(query_heads).max(1);
            let head_dim = dense_probe_norm_head_width(profile);
            // Full-token probes include context-sensitive runtime work: Flash
            // Attention over the active KV length and llama.cpp-shaped K/V
            // cache writes through GGML_OP_SET_ROWS. Do not confuse that active
            // attention length with the runtime's KV allocation capacity:
            // llama.cpp's `llama_kv_cache::get_n_kv()` pads the used sequence
            // length so the graph can be reused, but it does not make every
            // decode step attend over the entire `--ctx-size` allocation.
            //
            // Probe the same decode context that the scorer will charge for
            // this workload. Memory-fit logic separately sizes the full KV
            // allocation; decode tok/s should follow the occupied prompt/past
            // length, which is what Skippy steady-decode validation measures.
            let context_tokens = decode_context_tokens_for_validation(
                &selection_config(&primary_workload_profile()),
                profile,
            );
            let active_context_tokens = active_decode_context_tokens_for_validation(
                &selection_config(&primary_workload_profile()),
                profile,
                context_tokens,
            );
            if head_dim == 0
                || plan.kv_width != head_dim.saturating_mul(kv_heads)
                || profile
                    .context_length
                    .is_some_and(|native| context_tokens > native)
            {
                continue;
            }
            heartbeat(
                Some(model_index),
                &model_label,
                "model_dense_full_token_probe_start",
                &format!(
                    "backend={:?} block_tensor_type={} output_tensor_type={} hidden={} kv_width={} ffn={} vocab={} layers={} ctx={} nkv={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    output_tensor_type,
                    plan.hidden,
                    plan.kv_width,
                    plan.ffn,
                    vocab,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped full token {} {}->{} l{} ctx{} nkv{} vocab{}",
                    model_label,
                    plan.tensor_type,
                    output_tensor_type,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    vocab,
                ),
            );
            let full_token_shape = mesh_llm_gpu_bench::DenseFullTokenProbeShape {
                hidden: plan.hidden,
                kv_width: plan.kv_width,
                ffn: plan.ffn,
                vocab,
                repeat_layers: model_layers,
                graph_features: plan.graph_features,
                norm_head_width: plan.norm_head_width,
                head_dim,
                query_heads,
                kv_heads,
                context_tokens,
                active_context_tokens,
            };
            match run_native_probe_isolated(
                NativeProbeKind::FullToken,
                backend,
                plan.tensor_type,
                output_tensor_type,
                full_token_shape,
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_full_token_probe_done",
                        &format!(
                            "block_tensor_type={} output_tensor_type={} layers={} ctx={} vocab={} probes={}",
                            plan.tensor_type,
                            output_tensor_type,
                            model_layers,
                            context_tokens,
                            vocab,
                            probes.len()
                        ),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "full-token block_tensor_type={} output_tensor_type={} layers={model_layers} ctx={context_tokens} vocab={vocab}: {error:#}",
                        plan.tensor_type, output_tensor_type
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_full_token_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
            heartbeat(
                Some(model_index),
                &model_label,
                "model_dense_full_token_handoff_probe_start",
                &format!(
                    "backend={:?} block_tensor_type={} output_tensor_type={} hidden={} kv_width={} ffn={} vocab={} layers={} ctx={} nkv={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    output_tensor_type,
                    plan.hidden,
                    plan.kv_width,
                    plan.ffn,
                    vocab,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped full token handoff {} {}->{} l{} ctx{} nkv{} vocab{}",
                    model_label,
                    plan.tensor_type,
                    output_tensor_type,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    vocab,
                ),
            );
            match run_native_probe_isolated(
                NativeProbeKind::FullTokenHandoff,
                backend,
                plan.tensor_type,
                output_tensor_type,
                full_token_shape,
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_full_token_handoff_probe_done",
                        &format!(
                            "block_tensor_type={} output_tensor_type={} layers={} ctx={} vocab={} probes={}",
                            plan.tensor_type,
                            output_tensor_type,
                            model_layers,
                            context_tokens,
                            vocab,
                            probes.len()
                        ),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "full-token handoff block_tensor_type={} output_tensor_type={} layers={model_layers} ctx={context_tokens} vocab={vocab}: {error:#}",
                        plan.tensor_type, output_tensor_type
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_full_token_handoff_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
            heartbeat(
                Some(model_index),
                &model_label,
                "model_dense_decode_submission_probe_start",
                &format!(
                    "backend={:?} block_tensor_type={} output_tensor_type={} hidden={} kv_width={} ffn={} vocab={} layers={} ctx={} nkv={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    output_tensor_type,
                    plan.hidden,
                    plan.kv_width,
                    plan.ffn,
                    vocab,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped decode submission {} {}->{} l{} ctx{} nkv{} vocab{}",
                    model_label,
                    plan.tensor_type,
                    output_tensor_type,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    vocab,
                ),
            );
            match run_native_probe_isolated(
                NativeProbeKind::DecodeSubmission,
                backend,
                plan.tensor_type,
                output_tensor_type,
                full_token_shape,
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_decode_submission_probe_done",
                        &format!(
                            "block_tensor_type={} output_tensor_type={} layers={} ctx={} vocab={} probes={}",
                            plan.tensor_type,
                            output_tensor_type,
                            model_layers,
                            context_tokens,
                            vocab,
                            probes.len()
                        ),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "decode submission block_tensor_type={} output_tensor_type={} layers={model_layers} ctx={context_tokens} vocab={vocab}: {error:#}",
                        plan.tensor_type, output_tensor_type
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_decode_submission_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
            heartbeat(
                Some(model_index),
                &model_label,
                "model_dense_source_sampled_token_probe_start",
                &format!(
                    "backend={:?} block_tensor_type={} output_tensor_type={} hidden={} kv_width={} ffn={} vocab={} layers={} ctx={} nkv={} graph_features={} norm_head_width={}",
                    accelerator.backend,
                    plan.tensor_type,
                    output_tensor_type,
                    plan.hidden,
                    plan.kv_width,
                    plan.ffn,
                    vocab,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    plan.graph_features,
                    plan.norm_head_width,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped source sampled token {} {}->{} l{} ctx{} nkv{} vocab{}",
                    model_label,
                    plan.tensor_type,
                    output_tensor_type,
                    model_layers,
                    context_tokens,
                    active_context_tokens,
                    vocab,
                ),
            );
            match run_native_probe_isolated(
                NativeProbeKind::SourceSampledToken,
                backend,
                plan.tensor_type,
                output_tensor_type,
                full_token_shape,
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_source_sampled_token_probe_done",
                        &format!(
                            "block_tensor_type={} output_tensor_type={} layers={} ctx={} vocab={} probes={}",
                            plan.tensor_type,
                            output_tensor_type,
                            model_layers,
                            context_tokens,
                            vocab,
                            probes.len()
                        ),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "source sampled-token block_tensor_type={} output_tensor_type={} layers={model_layers} ctx={context_tokens} vocab={vocab}: {error:#}",
                        plan.tensor_type, output_tensor_type
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_dense_source_sampled_token_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
        }
    }
    collected
}

fn moe_model_specific_decode_kernel_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
) -> ModelSpecificDecodeProbes {
    let plans = moe_graph_probe_plans(profile);
    if plans.is_empty() {
        return ModelSpecificDecodeProbes {
            probes: Vec::new(),
            errors: vec![
                "could not derive model-shaped MoE graph probe dimensions from GGUF metadata"
                    .into(),
            ],
        };
    };
    let mut collected = ModelSpecificDecodeProbes::default();
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            let model_layers = profile.layer_count.unwrap_or(1).max(1);
            let context_tokens = decode_context_tokens_for_validation(
                &selection_config(&primary_workload_profile()),
                profile,
            );
            for &repeat_layers in plan.repeat_layers {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_moe_probe_start",
                    &format!(
                        "backend={:?} tensor_type={} experts={} used={} expert_width={} hidden={} kv_width={} layers={}",
                        accelerator.backend,
                        plan.tensor_type,
                        plan.expert_count,
                        plan.experts_used,
                        plan.expert_width,
                        plan.hidden,
                        plan.kv_width,
                        repeat_layers
                    ),
                );
                let _status = TerminalStatus::start(
                    args.show_progress,
                    format!(
                        "Probing model-shaped MoE block graph {} {} l{} {}x{} kv{}",
                        model_label,
                        plan.tensor_type,
                        repeat_layers,
                        plan.expert_width,
                        plan.hidden,
                        plan.kv_width
                    ),
                );
                match mesh_llm_gpu_bench::run_model_moe_block_graph_probe(
                    backend,
                    plan.tensor_type,
                    mesh_llm_gpu_bench::MoeBlockGraphProbeShape {
                        expert_count: plan.expert_count,
                        experts_used: plan.experts_used,
                        expert_width: plan.expert_width,
                        hidden: plan.hidden,
                        kv_width: plan.kv_width,
                        repeat_layers,
                    },
                ) {
                    Ok(probes) => {
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_moe_probe_done",
                            &format!(
                                "tensor_type={} layers={} probes={}",
                                plan.tensor_type,
                                repeat_layers,
                                probes.len()
                            ),
                        );
                        collected.probes.extend(probes);
                    }
                    Err(error) => {
                        let message = format!(
                            "tensor_type={} layers={repeat_layers}: {error:#}",
                            plan.tensor_type
                        );
                        heartbeat(
                            Some(model_index),
                            &model_label,
                            "model_moe_probe_error",
                            &message,
                        );
                        collected.errors.push(message);
                    }
                }
            }
            heartbeat(
                Some(model_index),
                &model_label,
                "model_moe_submission_probe_start",
                &format!(
                    "backend={:?} tensor_type={} experts={} used={} expert_width={} hidden={} kv_width={} layers={} ctx={}",
                    accelerator.backend,
                    plan.tensor_type,
                    plan.expert_count,
                    plan.experts_used,
                    plan.expert_width,
                    plan.hidden,
                    plan.kv_width,
                    model_layers,
                    context_tokens
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped MoE submission {} {} l{} ctx{} {}x{} kv{}",
                    model_label,
                    plan.tensor_type,
                    model_layers,
                    context_tokens,
                    plan.expert_width,
                    plan.hidden,
                    plan.kv_width
                ),
            );
            match mesh_llm_gpu_bench::run_model_moe_block_decode_submission_probe(
                backend,
                plan.tensor_type,
                mesh_llm_gpu_bench::MoeBlockGraphProbeShape {
                    expert_count: plan.expert_count,
                    experts_used: plan.experts_used,
                    expert_width: plan.expert_width,
                    hidden: plan.hidden,
                    kv_width: plan.kv_width,
                    repeat_layers: model_layers,
                },
                context_tokens,
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_moe_submission_probe_done",
                        &format!(
                            "tensor_type={} layers={} ctx={} probes={}",
                            plan.tensor_type,
                            model_layers,
                            context_tokens,
                            probes.len()
                        ),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "submission tensor_type={} layers={model_layers} ctx={context_tokens}: {error:#}",
                        plan.tensor_type
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_moe_submission_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
        }
    }
    collected
}

fn model_probe_label(profile: &ModelProfile) -> String {
    profile
        .source
        .metadata_name
        .clone()
        .unwrap_or_else(|| profile.source.id.clone())
}

fn append_attention_runtime_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
    collected: &mut ModelSpecificDecodeProbes,
) {
    let plans = attention_runtime_probe_plans(args, profile);
    if plans.is_empty() {
        return;
    }
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        for plan in &plans {
            heartbeat(
                Some(model_index),
                &model_label,
                "model_attention_runtime_probe_start",
                &format!(
                    "backend={:?} head_dim={} query_heads={} kv_heads={} ctx={} layers={}",
                    accelerator.backend,
                    plan.head_dim,
                    plan.query_heads,
                    plan.kv_heads,
                    plan.context_tokens,
                    plan.repeat_layers,
                ),
            );
            let _status = TerminalStatus::start(
                args.show_progress,
                format!(
                    "Probing model-shaped attention runtime {} h{} qh{} kvh{} ctx{} l{}",
                    model_label,
                    plan.head_dim,
                    plan.query_heads,
                    plan.kv_heads,
                    plan.context_tokens,
                    plan.repeat_layers
                ),
            );
            match mesh_llm_gpu_bench::run_model_attention_runtime_probe(
                backend,
                mesh_llm_gpu_bench::AttentionRuntimeProbeShape {
                    head_dim: plan.head_dim,
                    query_heads: plan.query_heads,
                    kv_heads: plan.kv_heads,
                    context_tokens: plan.context_tokens,
                    repeat_layers: plan.repeat_layers,
                },
            ) {
                Ok(probes) => {
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_attention_runtime_probe_done",
                        &format!("ctx={} probes={}", plan.context_tokens, probes.len()),
                    );
                    collected.probes.extend(probes);
                }
                Err(error) => {
                    let message = format!(
                        "attention runtime ctx={} layers={}: {error:#}",
                        plan.context_tokens, plan.repeat_layers
                    );
                    heartbeat(
                        Some(model_index),
                        &model_label,
                        "model_attention_runtime_probe_error",
                        &message,
                    );
                    collected.errors.push(message);
                }
            }
        }
    }
}

fn append_model_output_projection_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
    collected: &mut ModelSpecificDecodeProbes,
) {
    let Some(plan) = output_projection_probe_plan(profile) else {
        return;
    };
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        heartbeat(
            Some(model_index),
            &model_label,
            "model_output_projection_probe_start",
            &format!(
                "backend={:?} tensor_type={} vocab={} hidden={}",
                accelerator.backend, plan.tensor_type, plan.vocab, plan.hidden
            ),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Probing model-shaped output projection {} {} {}x{}",
                model_label, plan.tensor_type, plan.vocab, plan.hidden
            ),
        );
        match mesh_llm_gpu_bench::run_model_output_projection_probe(
            backend,
            plan.tensor_type,
            mesh_llm_gpu_bench::OutputProjectionProbeShape {
                hidden: plan.hidden,
                vocab: plan.vocab,
            },
        ) {
            Ok(probes) => {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_output_projection_probe_done",
                    &format!("tensor_type={} probes={}", plan.tensor_type, probes.len()),
                );
                collected.probes.extend(probes);
            }
            Err(error) => {
                let message = format!("output tensor_type={}: {error:#}", plan.tensor_type);
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_output_projection_probe_error",
                    &message,
                );
                collected.errors.push(message);
            }
        }
    }
}

fn append_model_logits_readback_probes(
    args: &Args,
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    model_index: usize,
    collected: &mut ModelSpecificDecodeProbes,
) {
    let Some(vocab) = profile.tokenizer.vocab_size.filter(|vocab| *vocab > 0) else {
        return;
    };
    let model_label = model_probe_label(profile);
    for accelerator in &hardware.accelerators {
        let Some(backend) = gpu_bench_backend(accelerator.backend) else {
            continue;
        };
        heartbeat(
            Some(model_index),
            &model_label,
            "model_logits_readback_probe_start",
            &format!("backend={:?} vocab={vocab}", accelerator.backend),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Probing model-shaped logits readback {} vocab{}",
                model_label, vocab
            ),
        );
        match mesh_llm_gpu_bench::run_model_logits_readback_probe(
            backend,
            mesh_llm_gpu_bench::LogitsReadbackProbeShape { vocab },
        ) {
            Ok(probes) => {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_readback_probe_done",
                    &format!("vocab={vocab} probes={}", probes.len()),
                );
                collected.probes.extend(probes);
            }
            Err(error) => {
                let message = format!("logits readback vocab={vocab}: {error:#}");
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_readback_probe_error",
                    &message,
                );
                collected.errors.push(message);
            }
        }
        heartbeat(
            Some(model_index),
            &model_label,
            "model_logits_sync_probe_start",
            &format!("backend={:?} vocab={vocab}", accelerator.backend),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Probing model-shaped logits sync {} vocab{}",
                model_label, vocab
            ),
        );
        match mesh_llm_gpu_bench::run_model_logits_sync_probe(
            backend,
            mesh_llm_gpu_bench::LogitsReadbackProbeShape { vocab },
        ) {
            Ok(probes) => {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_sync_probe_done",
                    &format!("vocab={vocab} probes={}", probes.len()),
                );
                collected.probes.extend(probes);
            }
            Err(error) => {
                let message = format!("logits sync vocab={vocab}: {error:#}");
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_sync_probe_error",
                    &message,
                );
                collected.errors.push(message);
            }
        }
        heartbeat(
            Some(model_index),
            &model_label,
            "model_logits_output_handoff_probe_start",
            &format!("backend={:?} vocab={vocab}", accelerator.backend),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Probing model-shaped logits output handoff {} vocab{}",
                model_label, vocab
            ),
        );
        match mesh_llm_gpu_bench::run_model_logits_output_handoff_probe(
            backend,
            mesh_llm_gpu_bench::LogitsReadbackProbeShape { vocab },
        ) {
            Ok(probes) => {
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_output_handoff_probe_done",
                    &format!("vocab={vocab} probes={}", probes.len()),
                );
                collected.probes.extend(probes);
            }
            Err(error) => {
                let message = format!("logits output handoff vocab={vocab}: {error:#}");
                heartbeat(
                    Some(model_index),
                    &model_label,
                    "model_logits_output_handoff_probe_error",
                    &message,
                );
                collected.errors.push(message);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct MoeGraphProbePlan {
    tensor_type: &'static str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    kv_width: u32,
    repeat_layers: &'static [u32],
}

#[derive(Clone, Debug)]
struct DenseGraphProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    kv_width: u32,
    ffn: u32,
    graph_features: u32,
    norm_head_width: u32,
    repeat_layers: Vec<u32>,
}

#[derive(Clone, Copy, Debug)]
struct AttentionRuntimeProbePlan {
    head_dim: u32,
    query_heads: u32,
    kv_heads: u32,
    context_tokens: u32,
    repeat_layers: u32,
}

#[derive(Clone, Debug)]
struct LinearAttentionGraphProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    qkv_width: u32,
    gate_width: u32,
    state_width: u32,
    output_input_width: u32,
    ffn: u32,
    recurrent_layers: u32,
    full_attention_layers: u32,
    kv_width: u32,
    graph_features: u32,
    norm_head_width: u32,
}

#[derive(Clone, Copy, Debug)]
struct OutputProjectionProbePlan {
    tensor_type: &'static str,
    hidden: u32,
    vocab: u32,
}

fn output_projection_probe_plan(profile: &ModelProfile) -> Option<OutputProjectionProbePlan> {
    let bytes = output_projection_probe_bytes(profile);
    if bytes == 0 {
        return None;
    }
    let hidden = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            u32::try_from(profile.tensor_matmul.output.shape.max_input_width)
                .ok()
                .filter(|width| *width > 0)
        })?;
    let vocab = profile
        .tokenizer
        .vocab_size
        .filter(|vocab| *vocab > 0)
        .or_else(|| {
            u32::try_from(profile.tensor_matmul.output.shape.max_output_width)
                .ok()
                .filter(|width| *width > 0)
        })?;
    let tensor_type = output_projection_probe_tensor_type(profile)
        .or_else(|| dense_probe_tensor_type_from_quant(profile.quantization.as_deref()))?;
    Some(OutputProjectionProbePlan {
        tensor_type,
        hidden,
        vocab,
    })
}

fn output_projection_probe_bytes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return profile
            .tensor_matmul
            .output
            .bytes
            .max(profile.tensor_group_bytes.output_bytes);
    }
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::DenseTransformer
        | model_fit::ModelArchitectureClass::SparseMoeTransformer
        | model_fit::ModelArchitectureClass::Unknown => profile.tensor_group_bytes.embedding_bytes,
        _ => 0,
    }
}

fn output_projection_probe_tensor_type(profile: &ModelProfile) -> Option<&'static str> {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return dominant_supported_tensor_type(profile.tensor_matmul.output.type_bytes);
    }
    dominant_supported_tensor_type(profile.tensor_group_bytes.embedding_type_bytes)
}

fn dominant_supported_tensor_type(bytes: TensorTypeBytes) -> Option<&'static str> {
    let mut candidates = [
        ("f16", bytes.f16_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
    ];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    candidates
        .into_iter()
        .find_map(|(kind, bytes)| (bytes > 0).then_some(kind))
}

fn linear_attention_graph_probe_plans(
    profile: &ModelProfile,
) -> Vec<LinearAttentionGraphProbePlan> {
    if !has_recurrent_attention_profile(profile) {
        return Vec::new();
    }
    let recurrent = &profile.recurrent_attention;
    let Some(hidden) = profile.hidden_size.filter(|hidden| *hidden > 0) else {
        return Vec::new();
    };
    let Some(ffn) = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        u32::try_from(
            profile
                .tensor_matmul
                .feed_forward
                .shape
                .weighted_avg_output_width
                .max(profile.tensor_matmul.feed_forward.shape.max_output_width),
        )
        .ok()
        .filter(|width| *width > 0)
    }) else {
        return Vec::new();
    };
    let Some(qkv_width) = recurrent_projection_output_width(&recurrent.qkv_projection) else {
        return Vec::new();
    };
    let Some(gate_width) = recurrent_projection_output_width(&recurrent.gate_projection) else {
        return Vec::new();
    };
    let Some(state_width) = recurrent_projection_output_width(&recurrent.beta_projection).max(
        recurrent_projection_output_width(&recurrent.alpha_projection),
    ) else {
        return Vec::new();
    };
    let Some(output_input_width) = recurrent_projection_input_width(&recurrent.output_projection)
    else {
        return Vec::new();
    };
    if output_input_width > qkv_width {
        return Vec::new();
    }
    let recurrent_layers = recurrent.recurrent_layer_count.max(1);
    let full_attention_layers = profile
        .layer_count
        .unwrap_or(recurrent_layers)
        .saturating_sub(recurrent_layers);
    let mut tensor_types = dense_probe_tensor_types(profile);
    tensor_types.dedup();
    tensor_types
        .into_iter()
        .map(|tensor_type| LinearAttentionGraphProbePlan {
            tensor_type,
            hidden,
            qkv_width,
            gate_width,
            state_width,
            output_input_width,
            ffn,
            recurrent_layers,
            full_attention_layers,
            kv_width: dense_probe_kv_width(profile, hidden),
            graph_features: dense_probe_graph_features(profile),
            norm_head_width: dense_probe_norm_head_width(profile),
        })
        .collect()
}

fn has_recurrent_attention_profile(profile: &ModelProfile) -> bool {
    let recurrent = &profile.recurrent_attention;
    recurrent.recurrent_layer_count > 0
        && recurrent.qkv_projection.shape.tensor_count > 0
        && recurrent.gate_projection.shape.tensor_count > 0
        && recurrent.output_projection.shape.tensor_count > 0
}

fn recurrent_projection_input_width(group: &model_fit::TensorMatmulGroupProfile) -> Option<u32> {
    u32::try_from(group.shape.max_input_width)
        .ok()
        .filter(|width| *width > 0)
}

fn recurrent_projection_output_width(group: &model_fit::TensorMatmulGroupProfile) -> Option<u32> {
    u32::try_from(group.shape.max_output_width)
        .ok()
        .filter(|width| *width > 0)
}

fn dense_graph_probe_plans(args: &Args, profile: &ModelProfile) -> Vec<DenseGraphProbePlan> {
    let hidden = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = profile.tensor_matmul.attention.shape;
            u32::try_from(shape.max_input_width.max(shape.max_output_width))
                .ok()
                .filter(|width| *width > 0)
        });
    let ffn = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        let shape = profile.tensor_matmul.feed_forward.shape;
        u32::try_from(
            shape
                .weighted_avg_output_width
                .max(shape.max_output_width)
                .max(shape.max_input_width),
        )
        .ok()
        .filter(|width| *width > 0)
    });
    let (Some(hidden), Some(ffn)) = (hidden, ffn) else {
        return Vec::new();
    };
    let mut tensor_types = dense_probe_tensor_types(profile);
    tensor_types.dedup();
    tensor_types
        .into_iter()
        .map(|tensor_type| DenseGraphProbePlan {
            tensor_type,
            hidden,
            kv_width: dense_probe_kv_width(profile, hidden),
            ffn,
            graph_features: dense_probe_graph_features(profile),
            norm_head_width: dense_probe_norm_head_width(profile),
            repeat_layers: dense_probe_repeat_layers(args, profile, tensor_type),
        })
        .collect()
}

fn attention_runtime_probe_plans(
    args: &Args,
    profile: &ModelProfile,
) -> Vec<AttentionRuntimeProbePlan> {
    let Some(layer_count) = profile.layer_count.filter(|layers| *layers > 0) else {
        return Vec::new();
    };
    let Some(query_heads) = profile.attention_heads.filter(|heads| *heads > 0) else {
        return Vec::new();
    };
    let kv_heads = profile.kv_heads.unwrap_or(query_heads).max(1);
    if query_heads % kv_heads != 0 {
        return Vec::new();
    }
    let head_dim = dense_probe_norm_head_width(profile);
    if head_dim == 0 {
        return Vec::new();
    }
    attention_runtime_context_ladder(args)
        .into_iter()
        .filter(|context_tokens| {
            profile
                .context_length
                .is_none_or(|native| *context_tokens <= native)
        })
        .map(|context_tokens| AttentionRuntimeProbePlan {
            head_dim,
            query_heads,
            kv_heads,
            context_tokens,
            repeat_layers: layer_count,
        })
        .collect()
}

fn attention_runtime_context_ladder(args: &Args) -> Vec<u32> {
    // llama.cpp decode attention cost is not just a dimensionless KV byte
    // count. The graph built in llama.cpp's one-token decode path calls
    // `ggml_flash_attn_ext` against the current KV length, and backend kernels
    // can change behavior materially between short contexts and the several
    // thousand token prompts used by agent/chat workloads. The validator's
    // primary-context scenario intentionally benchmarks decode after a
    // realistic prompt, so the hardware evidence we feed into model-fit must
    // include a synthetic attention runtime probe at that same context scale.
    //
    // This is still metadata-only evidence: the probe shape is derived from
    // GGUF fields such as layer count, head count, KV-head count, and head
    // width. It does not load model weights, it does not look at observed
    // model tok/s, and it does not branch on backend name. The smaller points
    // remain useful for short-context scenarios and for detecting nonlinear
    // context behavior; the default primary point prevents the scorer from
    // extrapolating a 512-token synthetic attention measurement to a ~4K
    // validation prompt.
    let mut contexts = vec![128, 512, DEFAULT_CTX_SIZE];
    if args.dense_probe_depth == DenseProbeDepth::Deep {
        contexts.push(2048);
    }
    contexts.sort_unstable();
    contexts.dedup();
    contexts
}

fn dense_probe_norm_head_width(profile: &ModelProfile) -> u32 {
    profile
        .key_length
        .filter(|width| *width > 0)
        .or_else(|| {
            let hidden = profile.hidden_size?;
            let heads = profile.attention_heads.filter(|heads| *heads > 0)?;
            (hidden % heads == 0).then_some(hidden / heads)
        })
        .unwrap_or_default()
}

fn dense_probe_graph_features(profile: &ModelProfile) -> u32 {
    let mut features = 0;
    if profile.dense_graph_features.attention_q_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_Q_NORM;
    }
    if profile.dense_graph_features.attention_k_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_K_NORM;
    }
    if profile.dense_graph_features.attention_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_ATTENTION_POST_NORM;
    }
    if profile.dense_graph_features.feed_forward_post_norm {
        features |= mesh_llm_gpu_bench::GRAPH_FEATURE_FFN_POST_NORM;
    }
    features
}

fn dense_probe_repeat_layers(args: &Args, profile: &ModelProfile, tensor_type: &str) -> Vec<u32> {
    if !supports_dense_depth_probe_tensor_type(tensor_type) {
        return vec![1];
    }

    let mut layers = match args.dense_probe_depth {
        DenseProbeDepth::Standard => vec![1, 4, 8],
        DenseProbeDepth::Deep => vec![1, 4, 8, 16],
    };

    if let Some(model_layers) = profile.layer_count.filter(|count| *count > 0) {
        if args.dense_probe_depth == DenseProbeDepth::Standard {
            add_standard_dense_depth_probes(&mut layers, model_layers);
        }
        if args.dense_probe_depth == DenseProbeDepth::Deep {
            // Deep validation is allowed to spend extra time building a
            // source-shaped synthetic graph whose layer count comes directly
            // from GGUF metadata. This is not an observed-throughput feedback
            // path: it does not load or run the real model weights, and it
            // does not consume the ABI/full-model tok/s result. Its purpose is
            // to falsify extrapolation from shallower graph probes to the
            // actual model depth.
            layers.push(model_layers);
        }
    }

    layers.sort_unstable();
    layers.dedup();
    layers
}

fn supports_dense_depth_probe_tensor_type(tensor_type: &str) -> bool {
    tensor_type.eq_ignore_ascii_case("q4_k")
        || tensor_type.eq_ignore_ascii_case("q5_k")
        || tensor_type.eq_ignore_ascii_case("q6_k")
        || tensor_type.eq_ignore_ascii_case("q8_0")
}

fn add_standard_dense_depth_probes(layers: &mut Vec<u32>, model_layers: u32) {
    // llama.cpp decode does not run one isolated matmul per layer. It submits a
    // whole one-token graph containing repeated attention/FFN matmuls, KV
    // reads/writes, normalization, residuals, output projection, backend graph
    // optimization, and command scheduling. Metal in particular can amortize
    // source-shaped graphs very differently between l8, l16, and full model
    // depth, while CUDA often stays closer to linear scaling. That difference
    // is a measured property of the backend graph, not a backend name rule.
    //
    // The default validator therefore collects enough synthetic graph depth to
    // falsify the old "scale l8 linearly to the whole model" assumption for
    // medium Q4_K dense models. These probes are still metadata-only: the
    // synthetic graph shape comes from GGUF fields such as layer count, hidden
    // width, KV width, FFN width, tensor type, and norm/head features. We do
    // not load the real model weights and we never feed observed tok/s back
    // into scoring.
    //
    // The rule applies to every dense graph tensor type that the estimator can
    // consume as a repeated llama block. Keeping Q6_K/Q8_0 shallow was useful
    // while the validator was still distinguishing tensor traffic from graph
    // depth, but it made the default evidence weaker exactly where llama.cpp's
    // source graph says repeated-block scheduling matters. Unsupported small
    // widths are rejected by the native GGML probe before graph construction,
    // so collecting depth rows here does not force invalid synthetic layouts.
    if model_layers >= 16 {
        layers.push(16);
    }

    if model_layers <= 32 {
        layers.push(model_layers);
    }
}

fn dense_probe_kv_width(profile: &ModelProfile, hidden: u32) -> u32 {
    let key_width = dense_probe_kv_vector_width(profile, profile.key_length, hidden);
    let value_width = dense_probe_kv_vector_width(profile, profile.value_length, hidden);
    key_width.max(value_width).max(1)
}

fn dense_probe_kv_vector_width(
    profile: &ModelProfile,
    vector_length: Option<u32>,
    hidden: u32,
) -> u32 {
    match (profile.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => kv_heads.saturating_mul(length).max(1),
        _ => hidden,
    }
}

fn dense_probe_tensor_types(profile: &ModelProfile) -> Vec<&'static str> {
    let bytes = add_tensor_type_bytes(
        profile.tensor_matmul.attention.type_bytes,
        profile.tensor_matmul.feed_forward.type_bytes,
    );
    let mut candidates = [
        ("f16", bytes.f16_bytes),
        ("q4_k", bytes.q4_k_bytes),
        ("q5_k", bytes.q5_k_bytes),
        ("q6_k", bytes.q6_k_bytes),
        ("q8_0", bytes.q8_0_bytes),
    ];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    let mut tensor_types = candidates
        .into_iter()
        .filter_map(|(kind, bytes)| (bytes > 0).then_some(kind))
        .collect::<Vec<_>>();
    if let (true, Some(tensor_type)) = (
        tensor_types.is_empty(),
        dense_probe_tensor_type_from_quant(profile.quantization.as_deref()),
    ) {
        tensor_types.push(tensor_type);
    }
    tensor_types
}

fn dense_probe_tensor_type_from_quant(quantization: Option<&str>) -> Option<&'static str> {
    let quantization = quantization?.to_ascii_lowercase();
    if quantization.contains("q4_k") {
        Some("q4_k")
    } else if quantization.contains("q5_k") {
        Some("q5_k")
    } else if quantization.contains("q6_k") {
        Some("q6_k")
    } else if quantization.contains("q8_0") || quantization.contains("q8") {
        Some("q8_0")
    } else if quantization.contains("f16") {
        Some("f16")
    } else {
        None
    }
}

fn add_tensor_type_bytes(left: TensorTypeBytes, right: TensorTypeBytes) -> TensorTypeBytes {
    TensorTypeBytes {
        f32_bytes: left.f32_bytes.saturating_add(right.f32_bytes),
        f16_bytes: left.f16_bytes.saturating_add(right.f16_bytes),
        bf16_bytes: left.bf16_bytes.saturating_add(right.bf16_bytes),
        q4_0_bytes: left.q4_0_bytes.saturating_add(right.q4_0_bytes),
        q4_k_bytes: left.q4_k_bytes.saturating_add(right.q4_k_bytes),
        q5_k_bytes: left.q5_k_bytes.saturating_add(right.q5_k_bytes),
        q6_k_bytes: left.q6_k_bytes.saturating_add(right.q6_k_bytes),
        q8_0_bytes: left.q8_0_bytes.saturating_add(right.q8_0_bytes),
        iq_bytes: left.iq_bytes.saturating_add(right.iq_bytes),
        other_quantized_bytes: left
            .other_quantized_bytes
            .saturating_add(right.other_quantized_bytes),
        unknown_bytes: left.unknown_bytes.saturating_add(right.unknown_bytes),
    }
}

fn moe_graph_probe_plans(profile: &ModelProfile) -> Vec<MoeGraphProbePlan> {
    let Some(expert_count) = profile.expert_count.filter(|count| *count > 0) else {
        return Vec::new();
    };
    let experts_used = profile
        .expert_used_count
        .filter(|used| *used > 0)
        .unwrap_or(expert_count)
        .min(expert_count);
    let Some(hidden) = profile
        .hidden_size
        .filter(|hidden| *hidden > 0)
        .or_else(|| {
            let shape = profile.tensor_matmul.expert_feed_forward.shape;
            u32::try_from(shape.max_input_width.max(shape.max_output_width))
                .ok()
                .filter(|width| *width > 0)
        })
    else {
        return Vec::new();
    };
    let Some(expert_width) = profile.ffn_size.filter(|ffn| *ffn > 0).or_else(|| {
        let shape = profile.tensor_matmul.expert_feed_forward.shape;
        u32::try_from(shape.min_input_width.min(shape.min_output_width))
            .ok()
            .filter(|width| *width > 0)
    }) else {
        return Vec::new();
    };
    let kv_width = model_attention_kv_width(profile).min(u64::from(hidden));
    let Ok(kv_width) = u32::try_from(kv_width.max(1)) else {
        return Vec::new();
    };
    moe_probe_tensor_types(profile)
        .into_iter()
        .map(|tensor_type| MoeGraphProbePlan {
            tensor_type,
            expert_count,
            experts_used,
            expert_width,
            hidden,
            kv_width,
            repeat_layers: &[1, 4, 8],
        })
        .collect()
}

fn model_attention_kv_width(profile: &ModelProfile) -> u64 {
    let key_width = model_kv_width(profile, profile.key_length);
    let value_width = model_kv_width(profile, profile.value_length);
    key_width.max(value_width).max(1)
}

fn model_kv_width(profile: &ModelProfile, vector_length: Option<u32>) -> u64 {
    match (profile.kv_heads, vector_length) {
        (Some(kv_heads), Some(length)) => u64::from(kv_heads).saturating_mul(u64::from(length)),
        _ => u64::from(profile.hidden_size.unwrap_or(1)),
    }
}

fn moe_probe_tensor_types(profile: &ModelProfile) -> Vec<&'static str> {
    let bytes = profile.tensor_matmul.expert_feed_forward.type_bytes;
    let mut candidates = [("q4_k", bytes.q4_k_bytes), ("q6_k", bytes.q6_k_bytes)];
    candidates.sort_by(|(_, left), (_, right)| right.cmp(left));
    candidates
        .into_iter()
        .filter_map(|(kind, bytes)| (bytes > 0).then_some(kind))
        .collect()
}

fn gpu_bench_backend(backend: BackendKind) -> Option<mesh_llm_gpu_bench::BenchmarkBackend> {
    match backend {
        BackendKind::Metal => Some(mesh_llm_gpu_bench::BenchmarkBackend::Metal),
        BackendKind::Cuda => Some(mesh_llm_gpu_bench::BenchmarkBackend::Cuda),
        BackendKind::Rocm => Some(mesh_llm_gpu_bench::BenchmarkBackend::Hip),
        _ => None,
    }
}

fn fit_interpretation(recommendation: &ModelRecommendation) -> FitInterpretation {
    let local_accelerated_fit = matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    ) && recommendation.selected_backend != BackendKind::Cpu;
    let single_node_validation_allowed = matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    );
    let summary = match recommendation.fit_status {
        FitStatus::FitsLocal => "fits local selected backend".into(),
        FitStatus::FitsWithWarning => "fits local selected backend with warnings".into(),
        FitStatus::Rejected => "does not fit local selected backend".into(),
    };
    let mut details = Vec::new();
    match recommendation.fit_status {
        FitStatus::Rejected => {
            details.push(
                "validation is skipped by default because no local serving shape was selected"
                    .into(),
            );
        }
        _ => {
            details.push(format!(
                "selected backend {:?} is the local validation target",
                recommendation.selected_backend
            ));
        }
    }
    FitInterpretation {
        local_accelerated_fit,
        single_node_validation_allowed,
        summary,
        details,
    }
}

fn runtime_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    benchmarks: &[BenchmarkScenarioSummary],
) -> RuntimeDiagnostic {
    // This diagnostic records the Skippy single-stage shape used for
    // validation. It is intentionally separate from the fit estimate. When a
    // model misses, the first question is whether the observed runtime used a
    // different launch shape than the metadata estimator assumed: partial layer
    // loading, explicit CPU fallback, lower KV precision, flash-attention
    // override, or a batch/ubatch override. Capturing those knobs makes
    // anomalies reproducible without feeding benchmark results back into
    // model-fit scoring.
    let steady_decode_command = benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == "steady_decode")
        .and_then(|benchmark| benchmark.benchmark.observations.first())
        .map(|observation| observation.command.clone());
    RuntimeDiagnostic {
        validation_shape: "skippy-bench local-single full-model runtime-slice",
        selected_backend: format!("{:?}", recommendation.selected_backend),
        selected_accelerator: recommendation.selected_accelerator.clone(),
        layer_start: 0,
        layer_end: profile.layer_count,
        ctx_size: DEFAULT_CTX_SIZE,
        n_gpu_layers: -1,
        cache_type_k: "f16",
        cache_type_v: "f16",
        flash_attn_type: "auto",
        n_batch: None,
        n_ubatch: None,
        load_mode: "runtime-slice",
        filter_tensors_on_load: false,
        include_embeddings: true,
        include_output: true,
        steady_decode_command,
        notes: vec![
            "n_gpu_layers=-1 asks llama.cpp/Skippy to offload as much as the selected backend can support.".into(),
            "No validator-level n_batch or n_ubatch override is passed; defaults come from the native runtime.".into(),
            "Metal/CUDA kernel selection is not yet exposed in this report; use native GGML/llama logging or ABI hooks for that next layer.".into(),
        ],
    }
}

async fn prepare_model(
    args: &Args,
    repository: &HfModelRepository,
    input: &ModelInput,
    model_index: usize,
) -> Result<PreparedModel> {
    match input {
        ModelInput::Ref(model_ref) => {
            prepare_model_ref(args, repository, model_ref, model_index).await
        }
        ModelInput::Local(local) => prepare_local_model(args, local, model_index),
    }
}

async fn prepare_model_ref(
    args: &Args,
    repository: &HfModelRepository,
    model_ref: &str,
    model_index: usize,
) -> Result<PreparedModel> {
    heartbeat(
        Some(model_index),
        model_ref,
        "resolve_start",
        "resolving model artifact",
    );
    let artifact = {
        let _status = TerminalStatus::start(args.show_progress, format!("Resolving {model_ref}"));
        resolve_model_artifact_ref(model_ref, repository)
            .await
            .with_context(|| format!("resolve model ref {model_ref}"))?
    };
    heartbeat(
        Some(model_index),
        model_ref,
        "resolve_done",
        &format!("canonical_ref={}", artifact.canonical_ref),
    );
    if artifact.format != ModelFormat::Gguf {
        bail!(
            "{model_ref} resolved to {:?}, expected GGUF",
            artifact.format
        );
    }
    heartbeat(
        Some(model_index),
        model_ref,
        "download_start",
        "ensuring GGUF artifact is available",
    );
    let progress = download_progress(args, model_ref);
    let downloaded_paths = repository
        .download_artifact_files_with_progress(&artifact, progress)
        .await
        .with_context(|| format!("download model ref {model_ref}"))?;
    heartbeat(
        Some(model_index),
        model_ref,
        "download_done",
        &format!("files={}", downloaded_paths.len()),
    );
    let primary_gguf_path = primary_download_path(&artifact, &downloaded_paths)?;
    heartbeat(
        Some(model_index),
        model_ref,
        "profile_start",
        &format!("path={}", primary_gguf_path.display()),
    );
    let mut profile = {
        let _status = TerminalStatus::start(args.show_progress, format!("Profiling {model_ref}"));
        profile_gguf_path(&primary_gguf_path)?
    };
    heartbeat(
        Some(model_index),
        model_ref,
        "profile_done",
        &profile_summary(&profile),
    );
    profile.source.id = model_ref.to_string();
    profile.source.path = Some(primary_gguf_path.clone());

    Ok(PreparedModel {
        input_ref: model_ref.to_string(),
        resolved_ref: Some(artifact.canonical_ref.clone()),
        artifact: Some(artifact),
        downloaded_paths,
        primary_gguf_path,
        profile,
    })
}

fn prepare_local_model(
    args: &Args,
    local: &LocalModelInput,
    model_index: usize,
) -> Result<PreparedModel> {
    heartbeat(
        Some(model_index),
        &local.model_ref,
        "profile_start",
        &format!("path={}", local.gguf_path.display()),
    );
    let mut profile = {
        let _status =
            TerminalStatus::start(args.show_progress, format!("Profiling {}", local.model_ref));
        profile_gguf_path(&local.gguf_path)?
    };
    heartbeat(
        Some(model_index),
        &local.model_ref,
        "profile_done",
        &profile_summary(&profile),
    );
    profile.source.id = local.model_ref.clone();
    profile.source.path = Some(local.gguf_path.clone());
    Ok(PreparedModel {
        input_ref: local.model_ref.clone(),
        resolved_ref: None,
        artifact: None,
        downloaded_paths: vec![local.gguf_path.clone()],
        primary_gguf_path: local.gguf_path.clone(),
        profile,
    })
}

fn run_abi_decode_probe_for_recommendation(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
) -> AbiDecodeProbeSummary {
    if let Some(reason) = abi_decode_probe_skip_reason(args, recommendation) {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_probe_skip",
            &reason,
        );
        return skipped_abi_decode_probe(reason);
    }
    run_abi_decode_probe(
        args,
        model,
        model_index,
        abi_decode_measured_tokens(recommendation),
        DEFAULT_WARMUP_TOKENS,
        "abi_probe",
    )
}

fn context_aligned_abi_decode_probe(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
    standard_probe: Option<&AbiDecodeProbeSummary>,
    decode_diagnostic: Option<&DecodeProbeDiagnostic>,
) -> Option<AbiDecodeProbeSummary> {
    if args.skip_context_aligned_abi {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_context_probe_skip",
            "skipping context-aligned ABI replay because --skip-context-aligned-abi was set",
        );
        return None;
    }
    if !context_aligned_abi_probe_needed(model, recommendation, standard_probe, decode_diagnostic) {
        return None;
    }
    let standard_probe = standard_probe?;
    let selected_context = selected_graph_probe_context_tokens(&model.profile, recommendation)?;
    let abi_context = abi_graph_context_tokens(standard_probe)?;
    let measured_tokens = abi_decode_measured_tokens(recommendation);
    let prompt_tokens = first_abi_prompt_tokens(standard_probe)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or_default();
    let warmup_tokens =
        context_aligned_abi_warmup_tokens(selected_context, prompt_tokens, measured_tokens)?;
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "abi_context_probe_plan",
        &format!(
            "selected_context={selected_context} abi_context={abi_context} prompt_tokens={prompt_tokens} warmup_tokens={warmup_tokens} measured_tokens={measured_tokens}"
        ),
    );
    Some(run_abi_decode_probe(
        args,
        model,
        model_index,
        measured_tokens,
        warmup_tokens,
        "abi_context_probe",
    ))
}

fn context_aligned_abi_probe_needed(
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    standard_probe: Option<&AbiDecodeProbeSummary>,
    decode_diagnostic: Option<&DecodeProbeDiagnostic>,
) -> bool {
    let Some(standard_probe) = standard_probe else {
        return false;
    };
    if standard_probe.error.is_some() || !standard_probe.attempted {
        return false;
    }
    if decode_diagnostic.is_none_or(|diagnostic| diagnostic.classification == "match") {
        return false;
    }
    let selected_context = selected_graph_probe_context_tokens(&model.profile, recommendation);
    let abi_context = abi_graph_context_tokens(standard_probe);
    !graph_contexts_match(selected_context, abi_context)
}

fn context_aligned_abi_warmup_tokens(
    selected_context: u32,
    prompt_tokens: usize,
    measured_tokens: usize,
) -> Option<usize> {
    let selected_context = usize::try_from(selected_context).ok()?;
    // `skippy_session_benchmark_decode()` builds the decode graph after the
    // prompt is loaded, after warmup decode calls, and during the measured
    // decode window. The graph inventory reports the active `n_kv` view for
    // that late measured decode graph, not just the requested context size.
    // To compare source-shaped synthetic probes to the ABI graph at the same
    // active-cache depth, put most of the distance to the selected synthetic
    // `_nkv` into warmup tokens and keep the same metadata-sized measured
    // window. This is diagnostic-only; the fitter still sees only metadata and
    // hardware benchmark facts.
    let occupied_before_warmup = prompt_tokens.saturating_add(measured_tokens);
    let warmup_tokens = selected_context.saturating_sub(occupied_before_warmup);
    (warmup_tokens > DEFAULT_WARMUP_TOKENS).then_some(warmup_tokens)
}

fn abi_decode_probe_skip_reason(
    args: &Args,
    recommendation: &ModelRecommendation,
) -> Option<String> {
    if !args.benchmark_all
        && !matches!(
            recommendation.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    {
        return Some(format!(
            "fit status is {:?}; use --benchmark-all to force single-stage ABI decode probe",
            recommendation.fit_status
        ));
    }
    if !args.benchmark_all && recommendation.selected_backend == BackendKind::Cpu {
        return Some(
            "fit selected CPU backend; use --benchmark-all to force the single-stage ABI decode probe"
                .into(),
        );
    }
    None
}

fn skipped_abi_decode_probe(reason: String) -> AbiDecodeProbeSummary {
    AbiDecodeProbeSummary {
        attempted: false,
        skip_reason: Some(reason),
        tokens_per_second: None,
        elapsed_ms: None,
        llama_eval_tokens_per_second: None,
        llama_eval_ms: None,
        non_eval_overhead_ms: None,
        non_eval_overhead_pct: None,
        decode_call_tokens_per_second: None,
        decode_call_ms: None,
        sampling_tokens_per_second: None,
        sampling_ms: None,
        logits_ready_ms: None,
        logits_scan_ms: None,
        llama_eval_count: None,
        llama_graph_reuse_count: None,
        graph_node_count: None,
        graph_inventory_bucket_overflow_count: None,
        graph_inventory: Vec::new(),
        measured_tokens: None,
        prompt_token_count: None,
        command: Vec::new(),
        observations: Vec::new(),
        sample_count: 0,
        raw_sample_count: 0,
        min_tokens_per_second: None,
        max_tokens_per_second: None,
        spread_pct: None,
        raw_spread_pct: None,
        denoised_outlier_count: 0,
        error: None,
    }
}

fn run_abi_decode_probe(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    measured_tokens: usize,
    warmup_tokens: usize,
    phase: &'static str,
) -> AbiDecodeProbeSummary {
    let mut summary = AbiDecodeProbeSummary {
        attempted: true,
        skip_reason: None,
        tokens_per_second: None,
        elapsed_ms: None,
        llama_eval_tokens_per_second: None,
        llama_eval_ms: None,
        non_eval_overhead_ms: None,
        non_eval_overhead_pct: None,
        decode_call_tokens_per_second: None,
        decode_call_ms: None,
        sampling_tokens_per_second: None,
        sampling_ms: None,
        logits_ready_ms: None,
        logits_scan_ms: None,
        llama_eval_count: None,
        llama_graph_reuse_count: None,
        graph_node_count: None,
        graph_inventory_bucket_overflow_count: None,
        graph_inventory: Vec::new(),
        measured_tokens: None,
        prompt_token_count: None,
        command: Vec::new(),
        observations: Vec::new(),
        sample_count: 0,
        raw_sample_count: 0,
        min_tokens_per_second: None,
        max_tokens_per_second: None,
        spread_pct: None,
        raw_spread_pct: None,
        denoised_outlier_count: 0,
        error: None,
    };
    let Some(layer_count) = model.profile.layer_count else {
        summary.error = Some("model metadata did not include layer count".into());
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "abi_probe_skip",
            "model metadata did not include layer count",
        );
        return summary;
    };
    let command_args = vec![
        "abi-decode-probe".to_string(),
        "--model-path".to_string(),
        model.primary_gguf_path.display().to_string(),
        "--ctx-size".to_string(),
        DEFAULT_CTX_SIZE.to_string(),
        "--n-gpu-layers=-1".to_string(),
        "--layer-end".to_string(),
        layer_count.to_string(),
        "--prompt".to_string(),
        validation_prompt().to_string(),
        "--warmup-tokens".to_string(),
        warmup_tokens.to_string(),
        "--measured-tokens".to_string(),
        measured_tokens.to_string(),
    ];
    summary.command = command_display(&args.skippy_bench_bin, &command_args);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        &format!("{phase}_start"),
        &format!(
            "repeats={} warmup_tokens={} measured_tokens={}",
            DEFAULT_ABI_DECODE_REPEATS, warmup_tokens, measured_tokens
        ),
    );
    for repeat in 0..DEFAULT_ABI_DECODE_REPEATS {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            &format!("{phase}_repeat_start"),
            &format!("repeat={}", repeat + 1),
        );
        summary.observations.push(run_abi_decode_probe_once(
            args,
            &model.input_ref,
            model_index,
            &command_args,
            repeat,
        ));
        if let Some(observation) = summary.observations.last() {
            heartbeat(
                Some(model_index),
                &model.input_ref,
                &format!("{phase}_repeat_done"),
                &abi_probe_observation_detail(observation),
            );
            if fatal_abi_probe_observation(observation) {
                heartbeat(
                    Some(model_index),
                    &model.input_ref,
                    &format!("{phase}_repeats_abort"),
                    "aborting ABI decode probe repeats after runtime startup failure",
                );
                break;
            }
        }
    }
    let summary = finalize_abi_decode_probe_summary(summary);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        &format!("{phase}_done"),
        &format!(
            "tok_s={} sample_count={} error={}",
            display_opt(summary.tokens_per_second),
            summary.sample_count,
            summary.error.as_deref().unwrap_or("-")
        ),
    );
    summary
}

fn abi_decode_measured_tokens(recommendation: &ModelRecommendation) -> usize {
    // The ABI decode probe is a validation diagnostic, not a scorer input. A
    // fixed 128-token window is enough for large local models where each token
    // is dominated by the repeated llama.cpp decode graph, but it is too short
    // for tiny models: sampler/session/runtime scheduling becomes a visible
    // fraction of elapsed time and the probe can classify as noisy even when
    // the graph inventory is correct. Reuse the steady-decode active-byte rule
    // so the measurement window grows only from metadata-derived model shape,
    // not from observed benchmark results, backend names, or model families.
    recommendation
        .estimated_active_decode_bytes_per_token
        .map(steady_decode_tokens_for_active_bytes)
        .unwrap_or(DEFAULT_ABI_DECODE_MEASURED_TOKENS)
        .max(DEFAULT_ABI_DECODE_MEASURED_TOKENS)
}

fn run_abi_decode_probe_once(
    args: &Args,
    model_ref: &str,
    model_index: usize,
    command_args: &[String],
    repeat: usize,
) -> AbiDecodeProbeObservation {
    let command = command_display(&args.skippy_bench_bin, command_args);
    let output = Command::new(&args.skippy_bench_bin)
        .args(command_args)
        .output();
    match output {
        Ok(output) if output.status.success() => {
            match parse_abi_decode_probe_json(&output.stdout) {
                Ok(parsed) => AbiDecodeProbeObservation {
                    repeat,
                    command,
                    status_code: output.status.code(),
                    tokens_per_second: parsed.tokens_per_second,
                    elapsed_ms: parsed.elapsed_ms,
                    llama_eval_tokens_per_second: parsed.llama_eval_tokens_per_second,
                    llama_eval_ms: parsed.llama_eval_ms,
                    non_eval_overhead_ms: parsed.non_eval_overhead_ms,
                    decode_call_tokens_per_second: parsed.decode_call_tokens_per_second,
                    decode_call_ms: parsed.decode_call_ms,
                    sampling_tokens_per_second: parsed.sampling_tokens_per_second,
                    sampling_ms: parsed.sampling_ms,
                    logits_ready_ms: parsed.logits_ready_ms,
                    logits_scan_ms: parsed.logits_scan_ms,
                    llama_eval_count: parsed.llama_eval_count,
                    llama_graph_reuse_count: parsed.llama_graph_reuse_count,
                    graph_node_count: parsed.graph_node_count,
                    graph_inventory_bucket_overflow_count: parsed
                        .graph_inventory_bucket_overflow_count,
                    graph_inventory: parsed.graph_inventory,
                    measured_tokens: parsed.measured_tokens,
                    prompt_token_count: parsed.prompt_token_count,
                    stderr_tail: stderr_tail(&output.stderr),
                    error: None,
                },
                Err(err) => AbiDecodeProbeObservation {
                    repeat,
                    command,
                    status_code: output.status.code(),
                    tokens_per_second: None,
                    elapsed_ms: None,
                    llama_eval_tokens_per_second: None,
                    llama_eval_ms: None,
                    non_eval_overhead_ms: None,
                    decode_call_tokens_per_second: None,
                    decode_call_ms: None,
                    sampling_tokens_per_second: None,
                    sampling_ms: None,
                    logits_ready_ms: None,
                    logits_scan_ms: None,
                    llama_eval_count: None,
                    llama_graph_reuse_count: None,
                    graph_node_count: None,
                    graph_inventory_bucket_overflow_count: None,
                    graph_inventory: Vec::new(),
                    measured_tokens: None,
                    prompt_token_count: None,
                    stderr_tail: stderr_tail(&output.stderr),
                    error: Some(err),
                },
            }
        }
        Ok(output) => AbiDecodeProbeObservation {
            repeat,
            command,
            status_code: output.status.code(),
            tokens_per_second: None,
            elapsed_ms: None,
            llama_eval_tokens_per_second: None,
            llama_eval_ms: None,
            non_eval_overhead_ms: None,
            decode_call_tokens_per_second: None,
            decode_call_ms: None,
            sampling_tokens_per_second: None,
            sampling_ms: None,
            logits_ready_ms: None,
            logits_scan_ms: None,
            llama_eval_count: None,
            llama_graph_reuse_count: None,
            graph_node_count: None,
            graph_inventory_bucket_overflow_count: None,
            graph_inventory: Vec::new(),
            measured_tokens: None,
            prompt_token_count: None,
            stderr_tail: stderr_tail(&output.stderr),
            error: Some(format!(
                "abi decode probe exited with status {}",
                output.status.code().unwrap_or(-1)
            )),
        },
        Err(err) => {
            heartbeat(
                Some(model_index),
                model_ref,
                "abi_probe_start_error",
                &format!("repeat={} error={err}", repeat + 1),
            );
            AbiDecodeProbeObservation {
                repeat,
                command,
                status_code: None,
                tokens_per_second: None,
                elapsed_ms: None,
                llama_eval_tokens_per_second: None,
                llama_eval_ms: None,
                non_eval_overhead_ms: None,
                decode_call_tokens_per_second: None,
                decode_call_ms: None,
                sampling_tokens_per_second: None,
                sampling_ms: None,
                logits_ready_ms: None,
                logits_scan_ms: None,
                llama_eval_count: None,
                llama_graph_reuse_count: None,
                graph_node_count: None,
                graph_inventory_bucket_overflow_count: None,
                graph_inventory: Vec::new(),
                measured_tokens: None,
                prompt_token_count: None,
                stderr_tail: None,
                error: Some(format!("failed to start abi decode probe: {err}")),
            }
        }
    }
}

fn finalize_abi_decode_probe_summary(mut summary: AbiDecodeProbeSummary) -> AbiDecodeProbeSummary {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.tokens_per_second)
        .collect::<Vec<_>>();
    summary.raw_sample_count = samples.len();
    if samples.is_empty() {
        summary.error = Some("all abi decode probe repeats failed".into());
        return summary;
    }

    let stats = throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD);
    let median = stats.clean_median.expect("non-empty ABI sample stats");
    summary.tokens_per_second = Some(median);
    summary.sample_count = stats.clean_sample_count;
    summary.min_tokens_per_second = stats.clean_min;
    summary.max_tokens_per_second = stats.clean_max;
    summary.spread_pct = stats.clean_spread.map(|spread| spread * 100.0);
    summary.raw_spread_pct = stats.raw_spread.map(|spread| spread * 100.0);
    summary.denoised_outlier_count = stats.outlier_count;
    summary.measured_tokens = first_abi_measured_tokens(&summary);
    summary.prompt_token_count = first_abi_prompt_tokens(&summary);
    summary.elapsed_ms = summary
        .measured_tokens
        .map(|tokens| tokens as f64 * 1000.0 / median);
    summary.llama_eval_tokens_per_second = median_abi_llama_eval_tokens_per_second(&summary);
    summary.decode_call_tokens_per_second = median_abi_decode_call_tokens_per_second(&summary);
    summary.sampling_tokens_per_second = median_abi_sampling_tokens_per_second(&summary);
    summary.llama_eval_ms = match (
        summary.measured_tokens,
        summary.llama_eval_tokens_per_second,
    ) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => None,
    };
    summary.decode_call_ms = match (
        summary.measured_tokens,
        summary.decode_call_tokens_per_second,
    ) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => first_abi_decode_call_ms(&summary),
    };
    summary.sampling_ms = match (summary.measured_tokens, summary.sampling_tokens_per_second) {
        (Some(tokens), Some(tps)) if tps > 0.0 => Some(tokens as f64 * 1000.0 / tps),
        _ => first_abi_sampling_ms(&summary),
    };
    summary.logits_ready_ms = median_abi_logits_ready_ms(&summary);
    summary.logits_scan_ms = median_abi_logits_scan_ms(&summary);
    summary.non_eval_overhead_ms = match (
        summary.elapsed_ms,
        summary.decode_call_ms,
        summary.sampling_ms,
    ) {
        (Some(elapsed_ms), Some(decode_ms), Some(sampling_ms)) => {
            Some((elapsed_ms - decode_ms - sampling_ms).max(0.0))
        }
        _ => match (summary.elapsed_ms, summary.llama_eval_ms) {
            (Some(elapsed_ms), Some(llama_eval_ms)) if llama_eval_ms > 0.0 => {
                Some((elapsed_ms - llama_eval_ms).max(0.0))
            }
            _ => first_abi_non_eval_overhead_ms(&summary),
        },
    };
    summary.non_eval_overhead_pct = match (summary.non_eval_overhead_ms, summary.elapsed_ms) {
        (Some(overhead_ms), Some(elapsed_ms)) if elapsed_ms > 0.0 => {
            Some(overhead_ms / elapsed_ms * 100.0)
        }
        _ => None,
    };
    summary.llama_eval_count = first_abi_llama_eval_count(&summary);
    summary.llama_graph_reuse_count = first_abi_llama_graph_reuse_count(&summary);
    summary.graph_node_count = first_abi_graph_node_count(&summary);
    summary.graph_inventory_bucket_overflow_count =
        first_abi_graph_inventory_bucket_overflow_count(&summary);
    summary.graph_inventory = first_abi_graph_inventory(&summary).unwrap_or_default();
    summary.error = abi_decode_probe_error(&summary);
    summary
}

#[derive(Clone, Debug)]
struct ParsedAbiDecodeProbe {
    tokens_per_second: Option<f64>,
    elapsed_ms: Option<f64>,
    llama_eval_tokens_per_second: Option<f64>,
    llama_eval_ms: Option<f64>,
    non_eval_overhead_ms: Option<f64>,
    decode_call_tokens_per_second: Option<f64>,
    decode_call_ms: Option<f64>,
    sampling_tokens_per_second: Option<f64>,
    sampling_ms: Option<f64>,
    logits_ready_ms: Option<f64>,
    logits_scan_ms: Option<f64>,
    llama_eval_count: Option<u64>,
    llama_graph_reuse_count: Option<i64>,
    graph_node_count: Option<u64>,
    graph_inventory_bucket_overflow_count: Option<u64>,
    graph_inventory: Vec<AbiGraphInventoryBucket>,
    measured_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
}

fn parse_abi_decode_probe_json(stdout: &[u8]) -> Result<ParsedAbiDecodeProbe, String> {
    let value = serde_json::from_slice::<Value>(stdout)
        .map_err(|err| format!("parse abi decode probe JSON: {err}"))?;
    let tokens_per_second = value.get("tokens_per_second").and_then(Value::as_f64);
    if tokens_per_second.is_none() {
        return Err("abi decode probe omitted tokens_per_second".into());
    }
    Ok(ParsedAbiDecodeProbe {
        tokens_per_second,
        elapsed_ms: value.get("elapsed_ms").and_then(Value::as_f64),
        llama_eval_tokens_per_second: value
            .get("llama_eval_tokens_per_second")
            .and_then(Value::as_f64),
        llama_eval_ms: value.get("llama_eval_ms").and_then(Value::as_f64),
        non_eval_overhead_ms: value.get("non_eval_overhead_ms").and_then(Value::as_f64),
        decode_call_tokens_per_second: value
            .get("decode_call_tokens_per_second")
            .and_then(Value::as_f64),
        decode_call_ms: value.get("decode_call_ms").and_then(Value::as_f64),
        sampling_tokens_per_second: value
            .get("sampling_tokens_per_second")
            .and_then(Value::as_f64),
        sampling_ms: value.get("sampling_ms").and_then(Value::as_f64),
        logits_ready_ms: value.get("logits_ready_ms").and_then(Value::as_f64),
        logits_scan_ms: value.get("logits_scan_ms").and_then(Value::as_f64),
        llama_eval_count: value.get("llama_eval_count").and_then(Value::as_u64),
        llama_graph_reuse_count: value.get("llama_graph_reuse_count").and_then(Value::as_i64),
        graph_node_count: value.get("graph_node_count").and_then(Value::as_u64),
        graph_inventory_bucket_overflow_count: value
            .get("graph_inventory_bucket_overflow_count")
            .and_then(Value::as_u64),
        graph_inventory: parse_abi_graph_inventory(&value),
        measured_tokens: value.get("measured_tokens").and_then(Value::as_u64),
        prompt_token_count: value.get("prompt_token_count").and_then(Value::as_u64),
    })
}

fn parse_abi_graph_inventory(value: &Value) -> Vec<AbiGraphInventoryBucket> {
    value
        .get("graph_inventory")
        .and_then(Value::as_array)
        .map(|buckets| {
            buckets
                .iter()
                .map(parse_abi_graph_inventory_bucket)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_abi_graph_inventory_bucket(value: &Value) -> AbiGraphInventoryBucket {
    AbiGraphInventoryBucket {
        family: value
            .get("family")
            .and_then(Value::as_str)
            .map(str::to_string),
        ggml_op: value.get("ggml_op").and_then(Value::as_i64),
        ggml_type: value.get("ggml_type").and_then(Value::as_u64),
        node_count: value.get("node_count").and_then(Value::as_u64),
        element_count: value.get("element_count").and_then(Value::as_u64),
        output_bytes: value.get("output_bytes").and_then(Value::as_u64),
        src0_bytes: value.get("src0_bytes").and_then(Value::as_u64),
        src1_bytes: value.get("src1_bytes").and_then(Value::as_u64),
        ne: value
            .get("ne")
            .and_then(Value::as_array)
            .map(|values| values.iter().filter_map(Value::as_i64).collect())
            .unwrap_or_default(),
    }
}

fn first_abi_measured_tokens(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.measured_tokens)
}

fn first_abi_prompt_tokens(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.prompt_token_count)
}

fn median_abi_llama_eval_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.llama_eval_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn median_abi_decode_call_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.decode_call_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn median_abi_sampling_tokens_per_second(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| observation.sampling_tokens_per_second)
        .collect::<Vec<_>>();
    (!samples.is_empty())
        .then(|| throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD))
        .and_then(|stats| stats.clean_median)
}

fn median_abi_logits_ready_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    median_abi_ms_total_from_observations(summary, |observation| observation.logits_ready_ms)
}

fn median_abi_logits_scan_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    median_abi_ms_total_from_observations(summary, |observation| observation.logits_scan_ms)
}

fn median_abi_ms_total_from_observations(
    summary: &AbiDecodeProbeSummary,
    field: impl Fn(&AbiDecodeProbeObservation) -> Option<f64>,
) -> Option<f64> {
    let samples = summary
        .observations
        .iter()
        .filter_map(|observation| {
            let tokens = observation.measured_tokens?;
            (tokens > 0).then(|| field(observation).map(|ms| ms / tokens as f64))?
        })
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    let per_token_ms = median(&samples);
    summary
        .measured_tokens
        .map(|tokens| per_token_ms * tokens as f64)
        .filter(|value| value.is_finite())
}

fn first_abi_non_eval_overhead_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.non_eval_overhead_ms)
}

fn first_abi_decode_call_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.decode_call_ms)
}

fn first_abi_sampling_ms(summary: &AbiDecodeProbeSummary) -> Option<f64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.sampling_ms)
}

fn first_abi_llama_eval_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.llama_eval_count)
}

fn first_abi_llama_graph_reuse_count(summary: &AbiDecodeProbeSummary) -> Option<i64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.llama_graph_reuse_count)
}

fn first_abi_graph_node_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.graph_node_count)
}

fn first_abi_graph_inventory_bucket_overflow_count(summary: &AbiDecodeProbeSummary) -> Option<u64> {
    summary
        .observations
        .iter()
        .find_map(|observation| observation.graph_inventory_bucket_overflow_count)
}

fn first_abi_graph_inventory(
    summary: &AbiDecodeProbeSummary,
) -> Option<Vec<AbiGraphInventoryBucket>> {
    summary
        .observations
        .iter()
        .find(|observation| !observation.graph_inventory.is_empty())
        .map(|observation| observation.graph_inventory.clone())
}

fn abi_decode_probe_error(summary: &AbiDecodeProbeSummary) -> Option<String> {
    let errors = summary
        .observations
        .iter()
        .filter_map(|observation| observation.error.as_deref())
        .collect::<Vec<_>>();
    (!errors.is_empty()).then(|| format!("{} abi decode repeats failed", errors.len()))
}

fn fatal_abi_probe_observation(observation: &AbiDecodeProbeObservation) -> bool {
    observation.error.is_some()
        && observation.status_code.is_some_and(|code| code != 0)
        && observation.tokens_per_second.is_none()
        && observation.elapsed_ms.is_none()
        && observation.llama_eval_ms.is_none()
        && observation.measured_tokens.is_none()
}

fn stderr_tail(stderr: &[u8]) -> Option<String> {
    let stderr = String::from_utf8_lossy(stderr);
    let tail = tail_lines(&stderr, 20);
    (!tail.trim().is_empty()).then_some(tail)
}

fn benchmark_model(
    args: &Args,
    hardware: &HardwareProfile,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
) -> Vec<BenchmarkScenarioSummary> {
    let scenarios = selected_benchmark_scenarios(args);
    let mut summaries = Vec::with_capacity(scenarios.len());
    let mut abort_reason = None;

    for (scenario_index, scenario) in scenarios.into_iter().enumerate() {
        if let Some(reason) = abort_reason.as_deref() {
            summaries.push(skipped_scenario_summary(
                scenario,
                &model.profile,
                recommendation,
                reason,
            ));
            continue;
        }
        let summary = benchmark_scenario(
            args,
            hardware,
            model,
            recommendation,
            model_index,
            scenario_index,
            scenario,
        );
        if fatal_benchmark_runtime_failure(&summary.benchmark) {
            abort_reason = Some(format!(
                "previous scenario {} failed to start the benchmark runtime",
                summary.scenario
            ));
        }
        summaries.push(summary);
    }

    summaries
}

fn benchmark_scenario(
    args: &Args,
    hardware: &HardwareProfile,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
    scenario_index: usize,
    scenario: BenchmarkScenarioSpec,
) -> BenchmarkScenarioSummary {
    let scenario = adapt_scenario_for_model(scenario, recommendation);
    let scenario = calibrate_scenario_prompt(args, model, recommendation, model_index, scenario);
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "scenario_start",
        &format!(
            "scenario={} max_new_tokens={} request_count={} reuse_session={}",
            scenario.name, scenario.max_new_tokens, scenario.request_count, scenario.reuse_session
        ),
    );
    let mut summary = BenchmarkSummary {
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    if let Some(reason) = benchmark_skip_reason(args, model, recommendation, scenario.kind) {
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "scenario_skip",
            &format!("scenario={} reason={reason}", scenario.name),
        );
        summary.skip_reason = Some(reason);
        let result = scenario_summary(
            scenario,
            &model.profile,
            recommendation,
            recommendation,
            summary,
        );
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "scenario_done",
            &scenario_summary_detail(&result),
        );
        return result;
    }

    let initial = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        0,
        DEFAULT_REPEATS,
    );
    // The default recommendation is scored for the default workload context,
    // but validation scenarios intentionally exercise different prompt shapes:
    // short steady decode, first-token/prefill, and KV reuse. Compare each
    // observed benchmark to a recommendation rescored with the measured prompt
    // and generated-token shape from that scenario. Otherwise the validator can
    // manufacture a false miss by judging a short decode run against a
    // long-context/default-chat estimate. This still stays honest: the
    // observation only selects the scenario context being validated, and no
    // observed tok/s is fed back into model-fit scoring.
    let expected = benchmark_expected_for_summary(
        hardware,
        &model.profile,
        recommendation,
        &scenario,
        &initial,
    );
    let summary = finalize_benchmark_summary(initial, &scenario, expected);
    let summary = remeasure_unstable_summary(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        expected,
        summary,
    );
    let summary = confirm_stable_fit_mismatch_summary(
        args,
        model,
        model_index,
        scenario_index,
        &scenario,
        expected,
        summary,
    );
    let scenario_recommendation = benchmark_scenario_recommendation(
        hardware,
        &model.profile,
        recommendation,
        &scenario,
        &summary,
    );
    let result = scenario_summary(
        scenario,
        &model.profile,
        recommendation,
        &scenario_recommendation,
        summary,
    );
    heartbeat(
        Some(model_index),
        &model.input_ref,
        "scenario_done",
        &scenario_summary_detail(&result),
    );
    result
}

fn run_benchmark_repeats(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    repeat_start: usize,
    repeat_count: usize,
) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary {
        attempted: true,
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    for repeat_offset in 0..repeat_count {
        let repeat = repeat_start + repeat_offset;
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "benchmark_repeat_start",
            &format!(
                "scenario={} repeat={} batch_repeat={}/{}",
                scenario.name,
                repeat + 1,
                repeat_offset + 1,
                repeat_count
            ),
        );
        let _status = TerminalStatus::start(
            args.show_progress,
            format!(
                "Benchmarking {} {} repeat {}/{}",
                model.input_ref,
                scenario.name,
                repeat_offset + 1,
                repeat_count
            ),
        );
        let observation =
            run_one_benchmark(args, model, model_index, scenario_index, repeat, scenario);
        heartbeat(
            Some(model_index),
            &model.input_ref,
            "benchmark_repeat_done",
            &benchmark_observation_detail(&observation, scenario),
        );
        if let Some(error) = observation.error.as_ref() {
            summary.errors.push(error.clone());
        }
        let fatal = fatal_benchmark_observation(&observation, scenario);
        summary.observations.push(observation);
        if fatal {
            let reason = format!(
                "aborting {} repeats after runtime startup failure; later repeats would relaunch the same single-stage runtime",
                scenario.name
            );
            heartbeat(
                Some(model_index),
                &model.input_ref,
                "benchmark_repeats_abort",
                &reason,
            );
            summary.errors.push(reason);
            break;
        }
    }
    summary
}

fn skipped_scenario_summary(
    scenario: BenchmarkScenarioSpec,
    model: &ModelProfile,
    recommendation: &ModelRecommendation,
    reason: &str,
) -> BenchmarkScenarioSummary {
    let summary = BenchmarkSummary {
        skip_reason: Some(reason.into()),
        verdict: "skipped".into(),
        ..BenchmarkSummary::default()
    };
    scenario_summary(scenario, model, recommendation, recommendation, summary)
}

fn benchmark_skip_reason(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    scenario: BenchmarkScenarioKind,
) -> Option<String> {
    if model.profile.layer_count.is_none() {
        return Some("model metadata did not include layer count".into());
    }
    if !args.benchmark_all
        && !matches!(
            recommendation.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    {
        return Some(format!(
            "fit status is {:?}; use --benchmark-all to force single-stage benchmark",
            recommendation.fit_status
        ));
    }
    if scenario.uses_decode_tps_prediction()
        && recommendation.estimated_decode_tokens_per_sec.is_none()
    {
        return Some("fit algorithm did not produce a decode tokens/sec estimate".into());
    }
    if scenario == BenchmarkScenarioKind::Prefill
        && recommendation.estimated_prefill_tokens_per_sec.is_none()
    {
        return Some("fit algorithm did not produce a prefill tokens/sec estimate".into());
    }
    if !args.benchmark_all && recommendation.selected_backend == BackendKind::Cpu {
        return Some(
            "fit selected CPU backend; use --benchmark-all to force the single-stage Skippy benchmark"
                .into(),
        );
    }
    None
}

fn adapt_scenario_for_model(
    mut scenario: BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> BenchmarkScenarioSpec {
    // Tiny models are fast enough that a short decode benchmark mostly measures
    // request/runtime jitter. Increase the steady-decode token window when the
    // fit estimate says the active byte footprint is small. This keeps the
    // validator honest: the fit algorithm still predicts from metadata alone,
    // while validation gives each model enough generated tokens for tok/s to be
    // a useful signal.
    if !scenario.kind.is_steady_decode() {
        return scenario;
    }
    if let Some(active_bytes) = recommendation.estimated_active_decode_bytes_per_token {
        scenario.max_new_tokens = steady_decode_tokens_for_active_bytes(active_bytes);
    }
    if scenario.kind == BenchmarkScenarioKind::PrimaryContextSteadyDecode {
        scenario.prompt = primary_context_prompt(recommendation, scenario.max_new_tokens);
    }
    scenario
}

fn steady_decode_tokens_for_active_bytes(active_bytes: u64) -> usize {
    let gib = 1024 * 1024 * 1024;
    if active_bytes < gib / 2 {
        1024
    } else if active_bytes < 2 * gib {
        512
    } else {
        DEFAULT_MAX_NEW_TOKENS
    }
}

fn primary_context_prompt(recommendation: &ModelRecommendation, max_new_tokens: usize) -> String {
    let target_context = recommendation
        .estimated_decode_context_tokens
        .unwrap_or(DEFAULT_CTX_SIZE / 2);
    let generated = u32::try_from(max_new_tokens).unwrap_or(u32::MAX);
    let generated_prefix = average_generated_prefix_tokens(generated);
    let prompt_words = target_context.saturating_sub(generated_prefix).max(64);
    synthetic_word_prompt(prompt_words)
}

fn calibrate_scenario_prompt(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    model_index: usize,
    mut scenario: BenchmarkScenarioSpec,
) -> BenchmarkScenarioSpec {
    if scenario.kind != BenchmarkScenarioKind::PrimaryContextSteadyDecode {
        return scenario;
    }
    match tokenizer_calibrated_primary_context_prompt(args, model, recommendation, &scenario) {
        Ok(candidate) => {
            heartbeat(
                Some(model_index),
                &model.input_ref,
                "scenario_prompt_calibrated",
                &format!(
                    "scenario={} target_ctx={} measured_prompt_tokens={} requested_tokens={} word_count={} sequence={}",
                    scenario.name,
                    display_u32(recommendation.estimated_decode_context_tokens),
                    candidate.prompt_tokens,
                    candidate.requested_tokens,
                    candidate.word_count,
                    candidate.sequence
                ),
            );
            scenario.prompt = candidate.prompt;
        }
        Err(error) => heartbeat(
            Some(model_index),
            &model.input_ref,
            "scenario_prompt_calibration_failed",
            &format!(
                "scenario={} error={error:#}; using uncalibrated synthetic prompt",
                scenario.name
            ),
        ),
    }
    scenario
}

fn tokenizer_calibrated_primary_context_prompt(
    args: &Args,
    model: &PreparedModel,
    recommendation: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
) -> Result<PromptLengthCandidate> {
    let target_prompt_tokens = target_primary_prompt_tokens(recommendation, scenario);
    let word_counts = primary_prompt_candidate_word_counts(target_prompt_tokens);
    let prompts = word_counts
        .iter()
        .map(|word_count| synthetic_word_prompt(*word_count))
        .collect::<Vec<_>>();
    let run_id = format!("model-fit-token-lengths-{}", std::process::id());
    let corpus_path = std::env::temp_dir().join(format!("{run_id}.jsonl"));
    let tsv_path = std::env::temp_dir().join(format!("{run_id}.tsv"));
    write_prompt_candidate_corpus(&corpus_path, &prompts)?;
    run_raw_token_lengths(args, model, scenario, &corpus_path, &tsv_path)?;
    let rows = read_prompt_length_candidates(&tsv_path, &word_counts, &prompts)?;
    select_prompt_length_candidate(rows, target_prompt_tokens, scenario.ctx_size)
}

fn target_primary_prompt_tokens(
    recommendation: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
) -> u32 {
    let target_context = recommendation
        .estimated_decode_context_tokens
        .unwrap_or(DEFAULT_CTX_SIZE / 2);
    let generated = u32::try_from(scenario.max_new_tokens).unwrap_or(u32::MAX);
    target_context
        .saturating_sub(average_generated_prefix_tokens(generated))
        .max(64)
}

fn primary_prompt_candidate_word_counts(target_prompt_tokens: u32) -> Vec<u32> {
    let mut counts = BTreeSet::new();
    counts.insert(64);
    for divisor in [4, 3, 2] {
        counts.insert((target_prompt_tokens / divisor).max(64));
    }
    for (numerator, denominator) in [(2, 3), (3, 4), (1, 1), (5, 4), (3, 2), (2, 1)] {
        counts.insert(
            target_prompt_tokens
                .saturating_mul(numerator)
                .checked_div(denominator)
                .unwrap_or(target_prompt_tokens)
                .max(64),
        );
    }
    counts.into_iter().collect()
}

fn write_prompt_candidate_corpus(path: &Path, prompts: &[String]) -> Result<()> {
    let mut corpus = String::new();
    for (index, prompt) in prompts.iter().enumerate() {
        corpus.push_str(&serde_json::to_string(&json!({
            "id": format!("candidate_{index}"),
            "prompt": prompt,
        }))?);
        corpus.push('\n');
    }
    fs::write(path, corpus).with_context(|| format!("write prompt corpus {}", path.display()))
}

fn run_raw_token_lengths(
    args: &Args,
    model: &PreparedModel,
    scenario: &BenchmarkScenarioSpec,
    corpus_path: &Path,
    tsv_path: &Path,
) -> Result<()> {
    let layer_count = model.profile.layer_count.unwrap_or(1).max(1);
    let output = Command::new(&args.skippy_bench_bin)
        .args([
            "token-lengths",
            "--model-path",
            &model.primary_gguf_path.display().to_string(),
            "--prompt-corpus",
            &corpus_path.display().to_string(),
            "--ctx-size",
            &scenario.ctx_size.to_string(),
            "--generation-limit",
            &scenario.max_new_tokens.to_string(),
            "--layer-end",
            &layer_count.to_string(),
            "--n-gpu-layers=-1",
            "--raw-prompt",
            "--output-tsv",
            &tsv_path.display().to_string(),
        ])
        .output()
        .context("run skippy-bench token-lengths")?;
    if output.status.success() {
        return Ok(());
    }
    bail!(
        "skippy-bench token-lengths failed with status {}; stderr: {}",
        output.status.code().unwrap_or(-1),
        tail_lines(&String::from_utf8_lossy(&output.stderr), 20)
    )
}

fn read_prompt_length_candidates(
    path: &Path,
    word_counts: &[u32],
    prompts: &[String],
) -> Result<Vec<PromptLengthCandidate>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read token lengths {}", path.display()))?;
    let mut rows = Vec::new();
    for line in contents.lines().skip(1).filter(|line| !line.is_empty()) {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() < 11 {
            bail!("token length row had {} fields, expected 11", fields.len());
        }
        let sequence = fields[0].parse::<usize>().context("parse sequence")?;
        let word_count = *word_counts
            .get(sequence)
            .with_context(|| format!("token length sequence {sequence} has no word count"))?;
        let prompt = prompts
            .get(sequence)
            .with_context(|| format!("token length sequence {sequence} has no prompt"))?
            .clone();
        rows.push(PromptLengthCandidate {
            sequence,
            word_count,
            prompt,
            prompt_tokens: fields[6].parse().context("parse prompt_tokens")?,
            requested_tokens: fields[8].parse().context("parse requested_tokens")?,
            fits_context: fields[10].parse().context("parse fits_context")?,
        });
    }
    if rows.is_empty() {
        bail!("skippy-bench token-lengths produced no rows");
    }
    Ok(rows)
}

fn select_prompt_length_candidate(
    rows: Vec<PromptLengthCandidate>,
    target_prompt_tokens: u32,
    ctx_size: u32,
) -> Result<PromptLengthCandidate> {
    let fits = rows
        .iter()
        .filter(|row| row.fits_context && row.requested_tokens <= ctx_size)
        .collect::<Vec<_>>();
    let candidates = if fits.is_empty() {
        rows.iter().collect::<Vec<_>>()
    } else {
        fits
    };
    candidates
        .into_iter()
        .min_by_key(|row| row.prompt_tokens.abs_diff(target_prompt_tokens))
        .cloned()
        .context("select prompt length candidate")
}

fn synthetic_word_prompt(word_count: u32) -> String {
    let mut prompt = String::from(
        "Synthetic context for model-fit primary-context validation. The benchmark uses the tokenizer-reported prompt length, not this word count, when comparing fit.\n",
    );
    for index in 0..word_count {
        if index > 0 {
            prompt.push(' ');
        }
        if index > 0 && index % 32 == 0 {
            prompt.push('\n');
        }
        prompt.push_str("context");
    }
    prompt.push_str("\nExplain how local inference speed depends on active model bytes.");
    prompt
}

fn finalize_benchmark_summary(
    mut summary: BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
) -> BenchmarkSummary {
    let samples = scenario_metric_samples(&summary, scenario);
    summary.raw_sample_count = samples.len();
    summary.successful_repeats = summary
        .observations
        .iter()
        .filter(|observation| observation.error.is_none())
        .count();
    if samples.is_empty() {
        summary.sample_count = 0;
        summary.verdict = if summary.attempted && benchmark_has_runtime_error(&summary) {
            "runtime-error".into()
        } else {
            "error".into()
        };
        return summary;
    }

    let stats = throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD);
    let median = stats.clean_median.expect("non-empty sample stats");
    let min = stats.clean_min.expect("non-empty sample stats");
    let max = stats.clean_max.expect("non-empty sample stats");
    let spread = stats.clean_spread.expect("non-empty sample stats");
    let observed_over_fit = if expected.predicted.is_some_and(|fit| fit > 0.0) {
        expected.predicted.map(|fit| median / fit)
    } else {
        None
    };

    summary.sample_count = stats.clean_sample_count;
    summary.median_tokens_per_sec = Some(median);
    summary.min_tokens_per_sec = Some(min);
    summary.max_tokens_per_sec = Some(max);
    summary.spread_pct = Some(spread * 100.0);
    summary.raw_median_tokens_per_sec = stats.raw_median;
    summary.raw_min_tokens_per_sec = stats.raw_min;
    summary.raw_max_tokens_per_sec = stats.raw_max;
    summary.raw_spread_pct = stats.raw_spread.map(|spread| spread * 100.0);
    add_request_level_steady_decode_stats(&mut summary, scenario);
    summary.denoised_outlier_count = stats.outlier_count;
    summary.observed_over_fit = observed_over_fit;
    summary.verdict = benchmark_verdict(median, observed_over_fit, spread, expected.range);
    summary
}

fn add_request_level_steady_decode_stats(
    summary: &mut BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) {
    if !scenario.kind.is_steady_decode() {
        return;
    }
    let samples = steady_decode_request_tokens_per_sec_samples(summary);
    if samples.is_empty() {
        return;
    }
    let stats = throughput_sample_stats(&samples, DEFAULT_MAX_SPREAD);
    summary.request_sample_count = samples.len();
    summary.request_median_tokens_per_sec = stats.raw_median;
    summary.request_min_tokens_per_sec = stats.raw_min;
    summary.request_max_tokens_per_sec = stats.raw_max;
    summary.request_spread_pct = stats.raw_spread.map(|spread| spread * 100.0);
}

fn remeasure_unstable_summary(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
    mut initial: BenchmarkSummary,
) -> BenchmarkSummary {
    let Some(reason) = remeasure_reason(&initial, scenario) else {
        return initial;
    };
    thread::sleep(DEFAULT_REMEASURE_PAUSE);
    let remeasure = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        scenario,
        DEFAULT_REPEATS,
        DEFAULT_REMEASURE_REPEATS,
    );
    let mut remeasure = finalize_benchmark_summary(remeasure, scenario, expected);
    if accepts_remeasure_summary(&remeasure) {
        remeasure.remeasured = true;
        remeasure.remeasure_reason = Some(reason);
        remeasure.initial_observations = initial.observations;
        return remeasure;
    }
    initial.remeasured = true;
    initial.remeasure_reason = Some(format!(
        "{reason}; remeasure was not stable enough to replace the initial pass"
    ));
    initial.rejected_remeasure_observations = remeasure.observations;
    initial
}

fn remeasure_reason(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<String> {
    if benchmark_has_runtime_error(summary) {
        return None;
    }
    let raw_spread = summary.raw_spread_pct? / 100.0;
    if raw_spread >= DEFAULT_REMEASURE_RAW_SPREAD {
        return Some(format!(
            "raw {} spread {:.1}% exceeded remeasure threshold {:.1}%",
            scenario.name,
            raw_spread * 100.0,
            DEFAULT_REMEASURE_RAW_SPREAD * 100.0
        ));
    }
    if !scenario.kind.is_steady_decode() {
        return None;
    }
    let ordered_drop = ordered_sample_drop(summary, scenario)?;
    (ordered_drop >= DEFAULT_REMEASURE_ORDERED_DROP).then(|| {
        format!(
            "ordered steady-decode samples dropped {:.1}% across repeats",
            ordered_drop * 100.0
        )
    })
}

fn ordered_sample_drop(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<f64> {
    let samples = scenario_metric_samples(summary, scenario);
    let first = samples.first().copied().filter(|sample| *sample > 0.0)?;
    let last = samples.last().copied()?;
    (last < first).then_some((first - last) / first)
}

fn accepts_remeasure_summary(summary: &BenchmarkSummary) -> bool {
    if benchmark_has_runtime_error(summary) {
        return false;
    }
    if summary.successful_repeats < 2 {
        return false;
    }
    summary
        .spread_pct
        .is_some_and(|spread| spread <= DEFAULT_MAX_SPREAD * 100.0)
}

fn confirm_stable_fit_mismatch_summary(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    scenario: &BenchmarkScenarioSpec,
    expected: BenchmarkExpected,
    mut initial: BenchmarkSummary,
) -> BenchmarkSummary {
    let Some(reason) = stable_fit_mismatch_confirmation_reason(&initial, scenario) else {
        return initial;
    };
    thread::sleep(DEFAULT_REMEASURE_PAUSE);
    let confirmation = run_benchmark_repeats(
        args,
        model,
        model_index,
        scenario_index,
        scenario,
        DEFAULT_REPEATS + DEFAULT_REMEASURE_REPEATS,
        DEFAULT_CONFIRM_REPEATS,
    );
    let mut confirmation = finalize_benchmark_summary(confirmation, scenario, expected);
    if accepts_confirmation_summary(&initial, &confirmation) {
        confirmation.remeasured = true;
        confirmation.remeasure_reason = Some(reason);
        confirmation.initial_observations = preserved_initial_observations(initial);
        return confirmation;
    }
    initial.remeasured = true;
    initial.remeasure_reason = Some(format!(
        "{reason}; confirmation did not materially change the stable mismatch"
    ));
    initial.rejected_remeasure_observations = confirmation.observations;
    initial
}

fn stable_fit_mismatch_confirmation_reason(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<String> {
    if !scenario.kind.is_steady_decode() || benchmark_has_runtime_error(summary) {
        return None;
    }
    if !is_stable_summary(summary) {
        return None;
    }
    let ratio = summary.observed_over_fit?;
    let outside_tolerance = (ratio - 1.0).abs() > DEFAULT_TOLERANCE;
    let mismatch_verdict = matches!(
        summary.verdict.as_str(),
        "slower-than-fit" | "faster-than-fit"
    );
    (outside_tolerance || mismatch_verdict).then(|| {
        format!(
            "stable steady-decode fit mismatch ratio {:.3} exceeded tolerance {:.1}%",
            ratio,
            DEFAULT_TOLERANCE * 100.0
        )
    })
}

fn accepts_confirmation_summary(
    initial: &BenchmarkSummary,
    confirmation: &BenchmarkSummary,
) -> bool {
    if !accepts_remeasure_summary(confirmation) {
        return false;
    }
    let Some(initial_ratio) = initial.observed_over_fit else {
        return false;
    };
    let Some(confirmation_ratio) = confirmation.observed_over_fit else {
        return false;
    };
    let Some(initial_median) = initial.median_tokens_per_sec else {
        return false;
    };
    let Some(confirmation_median) = confirmation.median_tokens_per_sec else {
        return false;
    };
    let observed_delta = relative_delta(initial_median, confirmation_median);
    let initial_error = (initial_ratio - 1.0).abs();
    let confirmation_error = (confirmation_ratio - 1.0).abs();
    observed_delta >= DEFAULT_CONFIRM_DELTA || confirmation_error + 0.05 < initial_error
}

fn is_stable_summary(summary: &BenchmarkSummary) -> bool {
    summary
        .spread_pct
        .is_some_and(|spread| spread <= DEFAULT_MAX_SPREAD * 100.0)
}

fn preserved_initial_observations(initial: BenchmarkSummary) -> Vec<BenchmarkObservation> {
    let mut observations = initial.initial_observations;
    observations.extend(initial.observations);
    observations
}

fn relative_delta(left: f64, right: f64) -> f64 {
    let baseline = left.abs().max(right.abs());
    if baseline <= f64::EPSILON {
        0.0
    } else {
        (left - right).abs() / baseline
    }
}

fn benchmark_has_runtime_error(summary: &BenchmarkSummary) -> bool {
    !summary.errors.is_empty()
        || summary
            .observations
            .iter()
            .any(|observation| observation.status_code.is_some_and(|code| code != 0))
        || summary
            .observations
            .iter()
            .any(|observation| observation.error.is_some())
}

fn fatal_benchmark_runtime_failure(summary: &BenchmarkSummary) -> bool {
    summary
        .observations
        .iter()
        .any(fatal_benchmark_observation_without_scenario)
}

fn fatal_benchmark_observation(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> bool {
    fatal_benchmark_observation_without_scenario(observation)
        && benchmark_observation_metric(observation, scenario).is_none()
}

fn fatal_benchmark_observation_without_scenario(observation: &BenchmarkObservation) -> bool {
    // A non-zero `skippy-bench` exit with no request observations and no
    // measured aggregate metric means the stage runtime did not reach the
    // request path. Retrying the same model/scenario launches the same
    // single-stage runtime with the same metadata-derived config, so this is
    // not a throughput sample and should not consume another startup timeout.
    observation.error.is_some()
        && observation.status_code.is_some_and(|code| code != 0)
        && observation.generated_tokens_per_sec.is_none()
        && observation.text_request_elapsed_ms.is_none()
        && observation.request_results.is_empty()
}

fn scenario_metric_samples(
    summary: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Vec<f64> {
    // Scenario sampling intentionally differs by workload shape.
    //
    // Steady decode should represent sustained token generation, so each repeat
    // contributes one aggregate decode-throughput sample. Aggregating generated
    // tokens over aggregate decode time denoises tiny models, where a single
    // request can swing just from scheduler/runtime jitter. This is not
    // "cheating" against the fit estimate: the prediction is unchanged and
    // metadata-only; we are only making the observation less noisy.
    //
    // Prefill is a separate metric from first-token latency. It uses Skippy's
    // request timing fields to compare prompt tokens / prefill elapsed time
    // against `estimated_prefill_tokens_per_sec`. That keeps the validation
    // falsifiable without using observed prefill speed as a scoring input.
    //
    // KV warm reuse cares about the reused final request, so it samples the last
    // request. First-token samples end-to-end request latency in milliseconds:
    // tokenize + prefill + the first decode step. That latency is deliberately
    // kept separate from prefill throughput because single-token decode after a
    // prompt can include graph/session/synchronization costs that sustained
    // steady decode does not see.
    let request_samples = match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::PrimaryContextSteadyDecode => {
            summary
                .observations
                .iter()
                .filter_map(steady_decode_observation_tokens_per_sec)
                .collect::<Vec<_>>()
        }
        BenchmarkScenarioKind::Prefill => summary
            .observations
            .iter()
            .filter_map(prefill_observation_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::KvWarmReuse => summary
            .observations
            .iter()
            .filter_map(|observation| observation.request_results.last())
            .filter_map(|request| request.generated_tokens_per_sec)
            .collect::<Vec<_>>(),
        BenchmarkScenarioKind::FirstToken => summary
            .observations
            .iter()
            .filter_map(|observation| observation.text_request_elapsed_ms)
            .collect::<Vec<_>>(),
    };
    if !request_samples.is_empty() {
        return request_samples;
    }
    summary
        .observations
        .iter()
        .filter_map(|observation| observation.generated_tokens_per_sec)
        .collect()
}

fn steady_decode_observation_tokens_per_sec(observation: &BenchmarkObservation) -> Option<f64> {
    // Prefer decode-only timings from `/v1/text`, excluding tokenization and
    // prefill. If an older benchmark binary did not emit those fields, fall back
    // to the last request's reported throughput so historical validation JSON
    // can still be summarized.
    let generated_tokens = observation
        .request_results
        .iter()
        .map(|request| request.generated_token_count.unwrap_or_default())
        .sum::<u64>();
    let decode_elapsed_ms = observation
        .request_results
        .iter()
        .filter_map(|request| request.decode_elapsed_ms)
        .sum::<f64>();
    if generated_tokens > 0 && decode_elapsed_ms > 0.0 {
        return Some(generated_tokens as f64 / (decode_elapsed_ms / 1000.0));
    }
    observation.request_results.last().and_then(|request| {
        request
            .decode_tokens_per_sec
            .or(request.generated_tokens_per_sec)
    })
}

fn steady_decode_request_tokens_per_sec_samples(summary: &BenchmarkSummary) -> Vec<f64> {
    // Repeat-level steady decode aggregates all generated tokens in a repeat so
    // a single fast or slow request does not dominate the pass/fail verdict.
    // Tiny models can still show a different failure mode: individual requests
    // inside one repeat swing widely even though aggregate throughput looks
    // stable enough after remeasurement. Keep that request-level spread in the
    // report as evidence for confidence decisions. This is not consumed by
    // scoring; it tells developers when the validation path is measuring
    // request/runtime variability more than metadata-visible matmul traffic.
    summary
        .observations
        .iter()
        .flat_map(|observation| observation.request_results.iter())
        .filter_map(|request| {
            request
                .decode_tokens_per_sec
                .or(request.generated_tokens_per_sec)
        })
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect()
}

fn prefill_observation_tokens_per_sec(observation: &BenchmarkObservation) -> Option<f64> {
    let prompt_tokens = observation
        .request_results
        .iter()
        .map(|request| request.prompt_token_count.unwrap_or_default())
        .sum::<u64>();
    let prefill_elapsed_ms = observation
        .request_results
        .iter()
        .filter_map(|request| request.prefill_elapsed_ms)
        .sum::<f64>();
    if prompt_tokens > 0 && prefill_elapsed_ms > 0.0 {
        return Some(prompt_tokens as f64 / (prefill_elapsed_ms / 1000.0));
    }
    None
}

fn benchmark_scenario_recommendation(
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    fallback: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> ModelRecommendation {
    if !matches!(
        scenario.kind,
        BenchmarkScenarioKind::SteadyDecode
            | BenchmarkScenarioKind::PrimaryContextSteadyDecode
            | BenchmarkScenarioKind::Prefill
            | BenchmarkScenarioKind::FirstToken
            | BenchmarkScenarioKind::KvWarmReuse
    ) {
        return fallback.clone();
    }
    let Some(context_tokens) = prediction_context_tokens(scenario, benchmark) else {
        return fallback.clone();
    };
    score_model_for_context_tokens(
        hardware,
        profile,
        &selection_config(&primary_workload_profile()),
        context_tokens,
    )
}

fn benchmark_expected_for_summary(
    hardware: &HardwareProfile,
    profile: &ModelProfile,
    fallback: &ModelRecommendation,
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> BenchmarkExpected {
    let recommendation =
        benchmark_scenario_recommendation(hardware, profile, fallback, scenario, benchmark);
    BenchmarkExpected {
        predicted: scenario_prediction(scenario, &recommendation),
        range: scenario_prediction_range(scenario, &recommendation),
    }
}

fn prediction_context_tokens(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> Option<u32> {
    let prompt_tokens = median_prompt_token_count(benchmark)?;
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::PrimaryContextSteadyDecode => {
            let generated = median_generated_tokens_per_request(benchmark, scenario)?;
            Some(prompt_tokens.saturating_add(average_generated_prefix_tokens(generated)))
        }
        BenchmarkScenarioKind::KvWarmReuse => {
            let generated = median_generated_tokens_per_request(benchmark, scenario)?;
            let current_decode =
                prompt_tokens.saturating_add(average_generated_prefix_tokens(generated));
            if !scenario.reuse_session {
                return Some(current_decode);
            }
            let prior_requests = u32::try_from(scenario.request_count.saturating_sub(1)).ok()?;
            // `skippy-bench local-single` reuses the same session id for the
            // warm-reuse scenario. By the final request, previous requests have
            // already appended their prompt and generated tokens to the KV
            // state. The observed metric for this scenario is the last
            // request's decode throughput, so the metadata estimate should
            // charge the prior cached sequence plus the average prefix length
            // within the current generated run. This is a causal-attention
            // shape fact, not calibration against the measured tokens/sec.
            let prior_cached =
                prior_requests.saturating_mul(prompt_tokens.saturating_add(generated));
            Some(prior_cached.saturating_add(current_decode))
        }
        BenchmarkScenarioKind::Prefill | BenchmarkScenarioKind::FirstToken => Some(prompt_tokens),
    }
}

fn average_generated_prefix_tokens(generated_tokens: u32) -> u32 {
    // During sampled decode, generated token i attends to the prompt plus the
    // i previously emitted tokens. Averaged across an N-token generation that
    // is prompt + (N - 1) / 2. The scorer only accepts an integer context
    // proxy, so round the generated-prefix half up by one token for odd/even
    // boundaries instead of introducing a hidden fractional fudge factor.
    generated_tokens.saturating_sub(1).div_ceil(2)
}

fn median_generated_tokens_per_request(
    benchmark: &BenchmarkSummary,
    scenario: &BenchmarkScenarioSpec,
) -> Option<u32> {
    let mut samples = benchmark
        .observations
        .iter()
        .flat_map(|observation| observation.request_results.iter())
        .filter_map(|request| request.generated_token_count)
        .filter_map(|count| u32::try_from(count).ok())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        samples = benchmark
            .observations
            .iter()
            .filter_map(|observation| {
                let generated = observation.generated_token_count?;
                let request_count = observation
                    .request_count
                    .or_else(|| u64::try_from(scenario.request_count).ok())
                    .filter(|count| *count > 0)?;
                u32::try_from(generated / request_count).ok()
            })
            .collect();
    }
    if samples.is_empty() {
        return u32::try_from(scenario.max_new_tokens).ok();
    }
    samples.sort_unstable();
    Some(samples[samples.len() / 2])
}

fn median_prompt_token_count(benchmark: &BenchmarkSummary) -> Option<u32> {
    let mut samples = benchmark
        .observations
        .iter()
        .filter_map(|observation| observation.prompt_token_count)
        .filter_map(|count| u32::try_from(count).ok())
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable();
    Some(samples[samples.len() / 2])
}

fn scenario_summary(
    scenario: BenchmarkScenarioSpec,
    model: &ModelProfile,
    primary_recommendation: &ModelRecommendation,
    prediction_recommendation: &ModelRecommendation,
    mut benchmark: BenchmarkSummary,
) -> BenchmarkScenarioSummary {
    let predicted = scenario_prediction(&scenario, prediction_recommendation);
    let predicted_range = scenario_prediction_range(&scenario, prediction_recommendation);
    let primary_predicted = scenario_prediction(&scenario, primary_recommendation);
    let primary_predicted_range = scenario_prediction_range(&scenario, primary_recommendation);
    let observed = scenario_observed(&scenario, &benchmark);
    let observed_over_fit = match (observed, predicted) {
        (Some(observed), Some(predicted)) if predicted > 0.0 => Some(observed / predicted),
        _ => None,
    };
    let primary_observed_over_fit = match (observed, primary_predicted) {
        (Some(observed), Some(predicted)) if predicted > 0.0 => Some(observed / predicted),
        _ => None,
    };
    let verdict = scenario_level_verdict(
        &scenario,
        &benchmark,
        observed,
        observed_over_fit,
        predicted_range,
    );
    // `BenchmarkSummary` is produced before scenario-specific rescoring is
    // possible, because the rescore needs the measured prompt-token count from
    // the benchmark itself. Keep the raw timing samples intact, but make the
    // embedded comparison fields match the scenario wrapper. Otherwise JSON
    // consumers and the Markdown table can accidentally report the generic
    // workload estimate while the scenario verdict was judged against the
    // prompt-shape-correct estimate.
    benchmark.observed_over_fit = observed_over_fit;
    benchmark.verdict = verdict.clone();
    BenchmarkScenarioSummary {
        scenario: scenario.name.into(),
        fit_metric: scenario.fit_metric.into(),
        predicted,
        predicted_range,
        prediction_source: prediction_source(primary_recommendation, prediction_recommendation),
        primary_predicted,
        primary_predicted_range,
        primary_observed_over_fit,
        prediction_context_tokens: prediction_context_tokens(&scenario, &benchmark),
        prediction_decode_cost_breakdown: prediction_recommendation.decode_cost_breakdown.clone(),
        observed,
        observed_over_fit,
        observed_over_abi: benchmark.observed_over_abi,
        first_token_breakdown: first_token_breakdown(
            &scenario,
            model,
            prediction_recommendation,
            &benchmark,
        ),
        verdict,
        benchmark,
    }
}

fn prediction_source(
    primary_recommendation: &ModelRecommendation,
    prediction_recommendation: &ModelRecommendation,
) -> &'static str {
    if std::ptr::eq(primary_recommendation, prediction_recommendation) {
        "primary_recommendation"
    } else {
        "scenario_rescored_for_benchmark_prompt_shape"
    }
}

fn apply_observed_over_abi(
    benchmarks: &mut [BenchmarkScenarioSummary],
    abi_decode_probe: Option<&AbiDecodeProbeSummary>,
) {
    let abi_tokens_per_second = abi_decode_probe.and_then(|probe| probe.tokens_per_second);
    for benchmark in benchmarks {
        let observed_over_abi = ratio(benchmark.observed, abi_tokens_per_second);
        benchmark.observed_over_abi = observed_over_abi;
        benchmark.benchmark.observed_over_abi = observed_over_abi;
    }
}

fn first_token_breakdown(
    scenario: &BenchmarkScenarioSpec,
    model: &ModelProfile,
    recommendation: &ModelRecommendation,
    benchmark: &BenchmarkSummary,
) -> Option<FirstTokenBreakdown> {
    if scenario.kind != BenchmarkScenarioKind::FirstToken {
        return None;
    }
    let prompt_token_count = median_prompt_token_count(benchmark).map(u64::from);
    let observed_tokenize_ms =
        median_request_value(benchmark, |request| request.tokenize_elapsed_ms);
    let observed_prefill_ms = median_request_value(benchmark, |request| request.prefill_elapsed_ms);
    let observed_decode_ms = median_request_value(benchmark, |request| request.decode_elapsed_ms);
    let observed_total_ms =
        median_observation_value(benchmark, |observation| observation.text_request_elapsed_ms);
    let predicted_decode_ms = recommendation
        .estimated_first_token_decode_ms
        .map(f64::from);
    let predicted_overhead_ms = recommendation
        .estimated_first_token_overhead_ms
        .map(f64::from);
    let predicted_sampler_ms = recommendation
        .estimated_first_token_sampler_ms
        .map(f64::from);
    let predicted_sampled_decode_ms = predicted_decode_ms.map(|decode| {
        decode
            + predicted_overhead_ms.unwrap_or_default()
            + predicted_sampler_ms.unwrap_or_default()
    });
    // `/v1/text` measures the first decode call with sampling included. In
    // Skippy that call eventually reaches `skippy_decode_step_sampled()`,
    // which runs `skippy_sync_chat_sampling_history()` before applying the
    // sampler chain. The sync loop accepts each prompt token into the sampler
    // history on the first sampled decode after prefill; subsequent decode
    // steps only accept the newly generated token. We report the residual here
    // instead of hiding it in a tuned estimator constant so validation can show
    // whether first-token misses scale with prompt length and vocabulary size.
    let observed_sampled_decode_residual_ms =
        match (observed_decode_ms, predicted_sampled_decode_ms) {
            (Some(observed), Some(predicted)) => Some((observed - predicted).max(0.0)),
            _ => None,
        };
    let observed_sampled_decode_residual_us_per_prompt_token = observed_sampled_decode_residual_ms
        .zip(prompt_token_count)
        .and_then(|(residual_ms, tokens)| {
            (tokens > 0).then_some(residual_ms * 1000.0 / tokens as f64)
        });
    let observed_sum = observed_tokenize_ms.unwrap_or_default()
        + observed_prefill_ms.unwrap_or_default()
        + observed_decode_ms.unwrap_or_default();
    let observed_unattributed_ms = observed_total_ms.map(|total| (total - observed_sum).max(0.0));
    Some(FirstTokenBreakdown {
        prompt_token_count,
        tokenizer_vocab_size: model.tokenizer.vocab_size,
        chat_template_available: model.tokenizer.chat_template_available,
        predicted_prefill_ms: recommendation
            .estimated_first_token_prefill_ms
            .map(f64::from),
        predicted_decode_ms,
        predicted_overhead_ms,
        predicted_sampler_ms,
        predicted_sampled_decode_ms,
        observed_tokenize_ms,
        observed_prefill_ms,
        observed_decode_ms,
        observed_sampled_decode_residual_ms,
        observed_sampled_decode_residual_us_per_prompt_token,
        observed_unattributed_ms,
    })
}

fn median_request_value(
    benchmark: &BenchmarkSummary,
    value: impl Fn(&BenchmarkRequestObservation) -> Option<f64>,
) -> Option<f64> {
    let mut samples = benchmark
        .observations
        .iter()
        .flat_map(|observation| observation.request_results.iter())
        .filter_map(value)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&samples))
}

fn scenario_level_verdict(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    predicted_range: Option<(f64, f64)>,
) -> String {
    if matches!(
        benchmark.verdict.as_str(),
        "skipped" | "error" | "runtime-error" | "inconclusive-noisy"
    ) {
        return benchmark.verdict.clone();
    }
    let Some(observed) = observed else {
        return "error".into();
    };
    let verdict = benchmark_verdict(
        observed,
        observed_over_fit,
        benchmark.spread_pct.unwrap_or_default() / 100.0,
        predicted_range,
    );
    if scenario.kind != BenchmarkScenarioKind::FirstToken {
        return verdict;
    }
    match verdict.as_str() {
        "slower-than-fit" => "faster-than-fit".into(),
        "faster-than-fit" => "slower-than-fit".into(),
        _ => verdict,
    }
}

fn scenario_observed(
    scenario: &BenchmarkScenarioSpec,
    benchmark: &BenchmarkSummary,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::PrimaryContextSteadyDecode => {
            benchmark.median_tokens_per_sec
        }
        BenchmarkScenarioKind::Prefill => benchmark.median_tokens_per_sec,
        BenchmarkScenarioKind::FirstToken => {
            median_observation_value(benchmark, |observation| observation.text_request_elapsed_ms)
        }
        BenchmarkScenarioKind::KvWarmReuse => median_observation_value(benchmark, |observation| {
            observation
                .request_results
                .last()
                .and_then(|request| request.generated_tokens_per_sec)
        }),
    }
}

fn median_observation_value(
    benchmark: &BenchmarkSummary,
    value: impl Fn(&BenchmarkObservation) -> Option<f64>,
) -> Option<f64> {
    let mut samples = benchmark
        .observations
        .iter()
        .filter_map(value)
        .collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&samples))
}

fn scenario_prediction(
    scenario: &BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode
        | BenchmarkScenarioKind::PrimaryContextSteadyDecode
        | BenchmarkScenarioKind::KvWarmReuse => recommendation
            .estimated_decode_tokens_per_sec
            .map(f64::from),
        BenchmarkScenarioKind::Prefill => recommendation
            .estimated_prefill_tokens_per_sec
            .map(f64::from),
        BenchmarkScenarioKind::FirstToken => recommendation.estimated_first_token_ms.map(f64::from),
    }
}

fn scenario_prediction_range(
    scenario: &BenchmarkScenarioSpec,
    recommendation: &ModelRecommendation,
) -> Option<(f64, f64)> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode
        | BenchmarkScenarioKind::PrimaryContextSteadyDecode
        | BenchmarkScenarioKind::KvWarmReuse => recommendation
            .estimated_decode_tokens_per_sec_range
            .map(|range| (f64::from(range.lower), f64::from(range.upper))),
        BenchmarkScenarioKind::Prefill => None,
        BenchmarkScenarioKind::FirstToken => recommendation
            .estimated_first_token_ms_range
            .map(|range| (f64::from(range.lower_ms), f64::from(range.upper_ms))),
    }
}

fn run_one_benchmark(
    args: &Args,
    model: &PreparedModel,
    model_index: usize,
    scenario_index: usize,
    repeat: usize,
    scenario: &BenchmarkScenarioSpec,
) -> BenchmarkObservation {
    let run_id = benchmark_run_id(model_index, scenario.name, repeat);
    let report_json_path = std::env::temp_dir().join(format!("{run_id}-report.json"));
    let stdout_json_path = std::env::temp_dir().join(format!("{run_id}.json"));
    let mut observation = BenchmarkObservation {
        repeat,
        run_id: run_id.clone(),
        command: Vec::new(),
        status_code: None,
        wall_seconds: 0.0,
        prompt_token_count: None,
        generated_tokens_per_sec: None,
        generated_token_count: None,
        text_request_elapsed_ms: None,
        request_count: None,
        reuse_session: None,
        request_results: Vec::new(),
        stdout_json_path: Some(stdout_json_path.clone()),
        report_json_path: report_json_path.clone(),
        stderr_tail: None,
        error: None,
    };

    let layer_count = model
        .profile
        .layer_count
        .expect("benchmark skip reason checked layer count");
    let Ok(port_base) = benchmark_port_base(args, model_index, scenario_index, repeat) else {
        observation.error = Some("port allocation overflow".into());
        return observation;
    };
    let command_args = benchmark_command_args(
        args,
        model,
        layer_count,
        port_base,
        &run_id,
        &report_json_path,
        scenario,
    );
    observation.command = command_display(&args.skippy_bench_bin, &command_args);

    let started = Instant::now();
    let output = Command::new(&args.skippy_bench_bin)
        .args(&command_args)
        .output();
    observation.wall_seconds = started.elapsed().as_secs_f64();

    match output {
        Ok(output) => read_benchmark_output(output, stdout_json_path, &mut observation),
        Err(err) => observation.error = Some(format!("failed to start skippy-bench: {err}")),
    }
    observation
}

fn benchmark_run_id(model_index: usize, scenario: &str, repeat: usize) -> String {
    format!(
        "model-fit-validate-{}-{model_index}-{scenario}-{repeat}",
        std::process::id()
    )
}

fn read_benchmark_output(
    output: std::process::Output,
    stdout_json_path: PathBuf,
    observation: &mut BenchmarkObservation,
) {
    observation.status_code = output.status.code();
    if !output.stderr.is_empty() {
        observation.stderr_tail = Some(tail_lines(&String::from_utf8_lossy(&output.stderr), 40));
    }
    if let Err(err) = fs::write(&stdout_json_path, &output.stdout) {
        observation.error = Some(format!("write benchmark stdout: {err}"));
        return;
    }
    if !output.status.success() {
        observation.error = Some(format!(
            "benchmark exited with status {}",
            output.status.code().unwrap_or(-1)
        ));
        return;
    }
    match serde_json::from_slice::<Value>(&output.stdout) {
        Ok(value) => apply_benchmark_json(&value, observation),
        Err(err) => observation.error = Some(format!("parse skippy-bench output JSON: {err}")),
    }
}

fn apply_benchmark_json(value: &Value, observation: &mut BenchmarkObservation) {
    observation.generated_tokens_per_sec = value
        .get("generated_tokens_per_sec")
        .and_then(Value::as_f64);
    observation.generated_token_count = value.get("generated_token_count").and_then(Value::as_u64);
    observation.prompt_token_count = value
        .get("request_results")
        .and_then(Value::as_array)
        .and_then(|results| results.first())
        .and_then(|result| result.get("prompt_token_count"))
        .and_then(Value::as_u64);
    observation.text_request_elapsed_ms =
        value.get("text_request_elapsed_ms").and_then(Value::as_f64);
    observation.request_count = value.get("request_count").and_then(Value::as_u64);
    observation.reuse_session = value.get("reuse_session").and_then(Value::as_bool);
    observation.request_results = value
        .get("request_results")
        .and_then(Value::as_array)
        .map(|results| results.iter().map(request_observation_from_json).collect())
        .unwrap_or_default();
    if observation.generated_tokens_per_sec.is_none() {
        observation.error = Some("skippy-bench output omitted generated_tokens_per_sec".into());
    }
}

fn request_observation_from_json(value: &Value) -> BenchmarkRequestObservation {
    BenchmarkRequestObservation {
        request_id: string_field(value, "request_id"),
        session_id: string_field(value, "session_id"),
        elapsed_ms: value.get("elapsed_ms").and_then(Value::as_f64),
        tokenize_elapsed_ms: value.get("tokenize_elapsed_ms").and_then(Value::as_f64),
        prefill_elapsed_ms: value.get("prefill_elapsed_ms").and_then(Value::as_f64),
        decode_elapsed_ms: value.get("decode_elapsed_ms").and_then(Value::as_f64),
        prompt_token_count: value.get("prompt_token_count").and_then(Value::as_u64),
        generated_token_count: value.get("generated_token_count").and_then(Value::as_u64),
        generated_tokens_per_sec: value
            .get("generated_tokens_per_sec")
            .and_then(Value::as_f64),
        decode_tokens_per_sec: value.get("decode_tokens_per_sec").and_then(Value::as_f64),
    }
}

fn benchmark_command_args(
    args: &Args,
    model: &PreparedModel,
    layer_count: u32,
    port_base: u16,
    run_id: &str,
    report_json_path: &Path,
    scenario: &BenchmarkScenarioSpec,
) -> Vec<String> {
    let mut command_args = vec![
        "local-single".into(),
        "--metrics-server-bin".into(),
        args.metrics_server_bin.display().to_string(),
        "--stage-server-bin".into(),
        args.skippy_server_bin.display().to_string(),
        "--model-path".into(),
        model.primary_gguf_path.display().to_string(),
        "--model-id".into(),
        model.input_ref.clone(),
        "--ctx-size".into(),
        scenario.ctx_size.to_string(),
        "--n-gpu-layers=-1".into(),
        "--layer-end".into(),
        layer_count.to_string(),
        "--warmup-new-tokens".into(),
        scenario.warmup_tokens.to_string(),
        "--max-new-tokens".into(),
        scenario.max_new_tokens.to_string(),
        "--request-count".into(),
        scenario.request_count.to_string(),
        "--prompt".into(),
        scenario.prompt.clone(),
        "--run-id".into(),
        run_id.to_string(),
        "--metrics-http-addr".into(),
        format!("127.0.0.1:{port_base}"),
        "--metrics-otlp-grpc-addr".into(),
        format!("127.0.0.1:{}", port_base + 1000),
        "--stage-bind-addr".into(),
        format!("127.0.0.1:{}", port_base + 2000),
        "--output".into(),
        report_json_path.display().to_string(),
        "--startup-timeout-secs".into(),
        "300".into(),
    ];
    if scenario.reuse_session {
        command_args.push("--reuse-session".into());
    }
    command_args
}

fn benchmark_port_base(
    args: &Args,
    model_index: usize,
    scenario_index: usize,
    repeat: usize,
) -> Result<u16> {
    let repeats_per_scenario =
        DEFAULT_REPEATS + DEFAULT_REMEASURE_REPEATS + DEFAULT_CONFIRM_REPEATS;
    let scenario_count = selected_benchmark_scenarios(args).len().max(1);
    args.base_port
        .checked_add(
            (model_index * scenario_count * repeats_per_scenario
                + scenario_index * repeats_per_scenario
                + repeat) as u16
                * 10,
        )
        .context("port allocation overflow")
}

fn benchmark_verdict(
    median: f64,
    observed_over_fit: Option<f64>,
    spread: f64,
    predicted_range: Option<(f64, f64)>,
) -> String {
    if observed_over_fit.is_none() {
        return "observed-only".into();
    }
    if spread > DEFAULT_MAX_SPREAD {
        return "inconclusive-noisy".into();
    }
    let Some(ratio) = observed_over_fit else {
        return "error".into();
    };
    let within_tolerance = (ratio - 1.0).abs() <= DEFAULT_TOLERANCE;
    let within_range = predicted_range
        .map(|(lower, upper)| median >= lower && median <= upper)
        .unwrap_or(true);
    if within_tolerance && within_range {
        "match".into()
    } else if within_range && (ratio - 1.0).abs() <= DEFAULT_TOLERANCE + spread {
        "inconclusive-noisy".into()
    } else if ratio < 1.0 {
        "slower-than-fit".into()
    } else {
        "faster-than-fit".into()
    }
}

fn load_hardware_profile(args: &Args) -> Result<LoadedHardware> {
    heartbeat(
        None,
        "hardware",
        "hardware_start",
        "building hardware profile",
    );
    let survey = mesh_llm_system::hardware::survey();
    let (benchmark_outputs, facts, raw_json) = if let Some(path) = args.gpu_benchmark_json.as_ref()
    {
        heartbeat(
            None,
            "hardware",
            "gpu_benchmark_json_start",
            &format!("path={}", path.display()),
        );
        let bytes = read_json_input(path)?;
        let raw_json: Value = serde_json::from_slice(&bytes).context("parse GPU benchmark JSON")?;
        let (outputs, facts) = parse_gpu_benchmark_json(&raw_json, &survey)?;
        heartbeat(
            None,
            "hardware",
            "gpu_benchmark_json_done",
            &format!("outputs={}", outputs.len()),
        );
        (outputs, facts, raw_json)
    } else {
        let benchmark = run_local_gpu_benchmark(args, &survey)?;
        let outputs = benchmark.outputs;
        let facts = default_facts_with_backend(&survey, outputs.len(), benchmark.backend);
        let raw_json = json!({
            "source": "model-fit-validate:auto_gpu_benchmark",
            "runner_backend": benchmark.backend,
            "outputs": outputs,
        });
        (outputs, facts, raw_json)
    };
    let default_backend = facts
        .first()
        .and_then(|fact| fact.backend)
        .unwrap_or_else(|| infer_backend_from_survey(&survey));
    let profile = hardware_profile_from_gpu_benchmark(GpuBenchmarkHardwareInput {
        memory: memory_profile(&survey, &facts),
        cpu: cpu_profile(),
        default_backend,
        accelerators: facts,
        benchmark_outputs: benchmark_outputs.clone(),
    })?;
    heartbeat(
        None,
        "hardware",
        "hardware_done",
        &format!(
            "accelerators={} backend={:?}",
            profile.accelerators.len(),
            default_backend
        ),
    );
    Ok(LoadedHardware {
        profile,
        benchmark_outputs,
        raw_json,
    })
}

fn parse_gpu_benchmark_json(
    raw_json: &Value,
    survey: &HardwareSurvey,
) -> Result<(Vec<GpuBenchmarkOutput>, Vec<GpuBenchmarkAcceleratorFacts>)> {
    match serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_json.clone()) {
        Ok(outputs) if !outputs.is_empty() => {
            return Ok((outputs.clone(), default_facts(survey, outputs.len())));
        }
        _ => {}
    }
    if let Some(outputs) = raw_json
        .get("outputs")
        .and_then(|raw_outputs| {
            serde_json::from_value::<Vec<GpuBenchmarkOutput>>(raw_outputs.clone()).ok()
        })
        .filter(|outputs| !outputs.is_empty())
    {
        return Ok((outputs.clone(), default_facts(survey, outputs.len())));
    }
    parse_gpus_command_json(raw_json, survey)
}

fn parse_gpus_command_json(
    raw_json: &Value,
    survey: &HardwareSurvey,
) -> Result<(Vec<GpuBenchmarkOutput>, Vec<GpuBenchmarkAcceleratorFacts>)> {
    let gpus = raw_json
        .get("gpus")
        .and_then(Value::as_array)
        .context("GPU benchmark JSON must be a raw BenchmarkOutput array or contain gpus[]")?;
    let mut outputs = Vec::new();
    let mut facts = Vec::new();
    for gpu in gpus {
        let Some(p90_gbps) = gpu.get("mem_bandwidth_gbps").and_then(Value::as_f64) else {
            continue;
        };
        if p90_gbps <= 0.0 {
            continue;
        }
        outputs.push(gpu_output_from_command_json(gpu, p90_gbps));
        facts.push(gpu_facts_from_command_json(gpu, survey));
    }
    if outputs.is_empty() {
        bail!("GPU benchmark JSON did not include any positive mem_bandwidth_gbps values");
    }
    Ok((outputs, facts))
}

fn gpu_output_from_command_json(gpu: &Value, p90_gbps: f64) -> GpuBenchmarkOutput {
    GpuBenchmarkOutput {
        device: string_field(gpu, "name").unwrap_or_else(|| "gpu".into()),
        buffer_mb: 0,
        runs: 0,
        p50_gbps: p90_gbps,
        p90_gbps,
        decode_effective_gbps: gpu.get("decode_effective_gbps").and_then(Value::as_f64),
        decode_fixed_overhead_ms: gpu.get("decode_fixed_overhead_ms").and_then(Value::as_f64),
        decode_runtime_overhead_ms: gpu
            .get("decode_runtime_overhead_ms")
            .and_then(Value::as_f64),
        post_prefill_decode_overhead_ms: gpu
            .get("post_prefill_decode_overhead_ms")
            .and_then(Value::as_f64),
        compute_tflops_fp32: gpu.get("compute_tflops_fp32").and_then(Value::as_f64),
        compute_tflops_fp16: gpu.get("compute_tflops_fp16").and_then(Value::as_f64),
        prefill_matmul_tflops_fp16: gpu
            .get("prefill_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        prefill_ubatch_matmul_tflops_fp16: gpu
            .get("prefill_ubatch_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        prefill_moe_matmul_tflops_fp16: gpu
            .get("prefill_moe_matmul_tflops_fp16")
            .and_then(Value::as_f64),
        sampler_history_us_per_token: gpu
            .get("sampler_history_us_per_token")
            .and_then(Value::as_f64),
        sampler_vocab_us_per_token: gpu
            .get("sampler_vocab_us_per_token")
            .and_then(Value::as_f64),
        decode_kernel_probes: gpu
            .get("decode_kernel_probes")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default(),
        noise_pct: 0.0,
        runtime_s: 0.0,
        rated_gbps: None,
        rated_estimated: None,
        efficiency_pct: None,
        bus_width_bits: None,
        mem_clock_mhz: None,
        gcn_arch: None,
        hbm: None,
    }
}

fn gpu_facts_from_command_json(
    gpu: &Value,
    survey: &HardwareSurvey,
) -> GpuBenchmarkAcceleratorFacts {
    let total_memory_bytes = gpu
        .get("vram_bytes")
        .and_then(Value::as_u64)
        .or_else(|| nonzero(survey.vram_bytes));
    let reserved_bytes = gpu.get("reserved_bytes").and_then(Value::as_u64);
    let unified_memory = gpu
        .get("unified_memory")
        .and_then(Value::as_bool)
        .unwrap_or_else(|| survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory));
    let name = string_field(gpu, "name").or_else(|| survey.gpu_name.clone());
    let backend = string_field(gpu, "backend_device")
        .as_deref()
        .map(infer_backend_from_device)
        .filter(|backend| *backend != BackendKind::Unknown)
        .or_else(|| Some(infer_backend_from_name(name.as_deref())));
    GpuBenchmarkAcceleratorFacts {
        name,
        kind: if unified_memory {
            AcceleratorKind::IntegratedGpu
        } else {
            AcceleratorKind::DiscreteGpu
        },
        backend,
        total_memory_bytes,
        available_memory_bytes: total_memory_bytes
            .map(|total| total.saturating_sub(reserved_bytes.unwrap_or(0))),
        unified_memory,
    }
}

fn default_facts(survey: &HardwareSurvey, count: usize) -> Vec<GpuBenchmarkAcceleratorFacts> {
    default_facts_with_backend(survey, count, BackendKind::Unknown)
}

fn default_facts_with_backend(
    survey: &HardwareSurvey,
    count: usize,
    runner_backend: BackendKind,
) -> Vec<GpuBenchmarkAcceleratorFacts> {
    (0..count)
        .map(|index| default_fact(survey, index, runner_backend))
        .collect()
}

fn default_fact(
    survey: &HardwareSurvey,
    index: usize,
    runner_backend: BackendKind,
) -> GpuBenchmarkAcceleratorFacts {
    let gpu = survey.gpus.get(index);
    let unified_memory = gpu
        .map(|gpu| gpu.unified_memory)
        .unwrap_or_else(|| survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory));
    let total_memory_bytes = gpu
        .and_then(|gpu| nonzero(gpu.vram_bytes))
        .or_else(|| survey.gpu_vram.get(index).copied().and_then(nonzero))
        .or_else(|| nonzero(survey.vram_bytes));
    let reserved_bytes = gpu
        .and_then(|gpu| gpu.reserved_bytes)
        .or_else(|| survey.gpu_reserved.get(index).copied().flatten());
    let name = gpu
        .map(|gpu| gpu.display_name.clone())
        .filter(|name| !name.is_empty())
        .or_else(|| survey.gpu_name.clone());
    let backend = gpu
        .and_then(|gpu| gpu.backend_device.as_deref())
        .map(infer_backend_from_device)
        .filter(|backend| *backend != BackendKind::Unknown)
        .or_else(|| {
            let inferred = infer_backend_from_name(name.as_deref());
            (inferred != BackendKind::Unknown).then_some(inferred)
        })
        .or_else(|| (runner_backend != BackendKind::Unknown).then_some(runner_backend));
    GpuBenchmarkAcceleratorFacts {
        name,
        kind: if unified_memory {
            AcceleratorKind::IntegratedGpu
        } else {
            AcceleratorKind::DiscreteGpu
        },
        backend,
        total_memory_bytes,
        available_memory_bytes: total_memory_bytes
            .map(|total| total.saturating_sub(reserved_bytes.unwrap_or(0))),
        unified_memory,
    }
}

fn memory_profile(
    survey: &HardwareSurvey,
    facts: &[GpuBenchmarkAcceleratorFacts],
) -> MemoryProfile {
    let detected_unified_total = facts
        .iter()
        .filter(|fact| fact.unified_memory)
        .filter_map(|fact| fact.total_memory_bytes)
        .max();
    let detected_unified_available = facts
        .iter()
        .filter(|fact| fact.unified_memory)
        .filter_map(|fact| fact.available_memory_bytes)
        .max();
    let total = system_total_memory_bytes()
        .or(detected_unified_total)
        .or_else(|| nonzero(survey.vram_bytes));
    let available = system_available_memory_bytes()
        .or(detected_unified_available)
        .or(total);
    let has_unified = facts.iter().any(|fact| fact.unified_memory)
        || survey.is_soc
        || survey.gpus.iter().any(|gpu| gpu.unified_memory);
    MemoryProfile {
        total_system_bytes: total,
        available_system_bytes: available,
        total_unified_bytes: has_unified.then_some(total).flatten(),
        available_unified_bytes: has_unified.then_some(available).flatten(),
    }
}

fn cpu_profile() -> CpuProfile {
    CpuProfile {
        physical_cores: None,
        logical_cores: std::thread::available_parallelism()
            .ok()
            .and_then(|count| u32::try_from(count.get()).ok()),
        memory_bandwidth_bytes_per_sec: None,
        compute_tflops_fp16: None,
        post_prefill_decode_overhead_ms: None,
        prefill_matmul_tflops_fp16: None,
        prefill_ubatch_matmul_tflops_fp16: None,
        prefill_moe_matmul_tflops_fp16: None,
        sampler_history_us_per_token: None,
        sampler_vocab_us_per_token: None,
    }
}

fn run_local_gpu_benchmark(args: &Args, survey: &HardwareSurvey) -> Result<LocalGpuBenchmark> {
    heartbeat(
        None,
        "hardware",
        "gpu_benchmark_start",
        "running local GPU benchmark",
    );
    let _status = TerminalStatus::start(
        args.show_progress,
        "Benchmarking local GPU memory bandwidth".into(),
    );
    let runner = benchmark_runner_for_survey(survey)?;
    let runner_backend = backend_kind_from_runner(runner.backend);
    let outputs = mesh_llm_gpu_bench::run_benchmark_with_options(
        runner,
        Duration::from_secs(300),
        mesh_llm_gpu_bench::BenchmarkOptions {
            // Keep the validator's automatic machine profile focused on the
            // portable facts that every fit needs: measured memory bandwidth,
            // launch overhead, prefill compute probes, sampler overhead, VRAM,
            // and accelerator identity. Validation then appends model-shaped
            // probes derived from the GGUF being tested.
            //
            // Standard/deep GGML probe mode is useful for manual backend
            // analysis and for `mesh-llm gpus benchmark`, but it is intentionally
            // not the automatic validation path. Those modes sweep generic GGML
            // graph shapes, which can make a smoke validation spend minutes in
            // hardware profiling before it has even looked at the model. More
            // importantly, generic probes are not a better source of truth than
            // metadata-shaped probes below: for a sparse MoE model, for example,
            // we derive the expert count, active experts, expert width, hidden
            // width, tensor type, and repeated layer depth directly from GGUF
            // metadata and run that exact graph.
            //
            // This keeps the estimator honest:
            // - hardware facts come from observed benchmark data, not marketing
            //   bandwidth or backend-specific constants;
            // - model-shaped corrections come from source-faithful graph probes
            //   keyed by metadata dimensions, not filenames or observed model
            //   throughput;
            // - validation remains repeatable enough to run as a smoke check on
            //   CUDA/Metal hosts without burning cycles on unrelated shapes.
            probe_depth: mesh_llm_gpu_bench::ProbeDepth::HardwareOnly,
        },
    )
    .context("run local GPU benchmark")?;
    heartbeat(
        None,
        "hardware",
        "gpu_benchmark_done",
        &format!("outputs={}", outputs.len()),
    );
    Ok(LocalGpuBenchmark {
        outputs,
        backend: runner_backend,
    })
}

fn backend_kind_from_runner(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> BackendKind {
    match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => BackendKind::Metal,
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => BackendKind::Cuda,
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => BackendKind::Rocm,
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => BackendKind::Vulkan,
    }
}

fn benchmark_runner_for_survey(
    survey: &HardwareSurvey,
) -> Result<mesh_llm_gpu_bench::BenchmarkRunner> {
    let gpu_name = survey
        .gpu_name
        .as_deref()
        .or_else(|| survey.gpus.first().map(|gpu| gpu.display_name.as_str()));
    let gpu_count = if survey.gpu_count > 0 {
        survey.gpu_count
    } else {
        u8::try_from(survey.gpus.len()).unwrap_or(u8::MAX)
    };
    let is_soc = survey.is_soc || survey.gpus.iter().any(|gpu| gpu.unified_memory);
    mesh_llm_gpu_bench::runner_for(std::env::consts::OS, gpu_count, gpu_name, is_soc)
        .context("could not infer GPU benchmark backend from local hardware")
}

fn infer_backend_from_survey(survey: &HardwareSurvey) -> BackendKind {
    if survey
        .gpus
        .iter()
        .filter_map(|gpu| gpu.backend_device.as_deref())
        .any(|device| infer_backend_from_device(device) == BackendKind::Metal)
        || std::env::consts::OS == "macos"
    {
        return BackendKind::Metal;
    }
    infer_backend_from_name(survey.gpu_name.as_deref())
}

fn infer_backend_from_device(device: &str) -> BackendKind {
    let upper = device.to_ascii_uppercase();
    if upper.starts_with("MTL") || upper.contains("METAL") {
        BackendKind::Metal
    } else if upper.contains("CUDA") || upper.contains("NVIDIA") {
        BackendKind::Cuda
    } else if upper.contains("HIP") || upper.contains("ROCM") || upper.contains("AMD") {
        BackendKind::Rocm
    } else if upper.contains("VULKAN") {
        BackendKind::Vulkan
    } else {
        BackendKind::Unknown
    }
}

fn infer_backend_from_name(name: Option<&str>) -> BackendKind {
    let Some(name) = name else {
        return BackendKind::Unknown;
    };
    let upper = name.to_ascii_uppercase();
    if upper.contains("APPLE") || upper.contains("METAL") {
        BackendKind::Metal
    } else if upper.contains("NVIDIA") {
        BackendKind::Cuda
    } else if upper.contains("AMD") || upper.contains("RADEON") {
        BackendKind::Rocm
    } else {
        BackendKind::Unknown
    }
}

fn system_total_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "macos")]
    {
        return command_u64("sysctl", &["-n", "hw.memsize"]);
    }
    #[cfg(target_os = "linux")]
    {
        return fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_value_bytes(&text, "MemTotal:"));
    }
    #[allow(unreachable_code)]
    None
}

fn system_available_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|text| meminfo_value_bytes(&text, "MemAvailable:"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn meminfo_value_bytes(text: &str, key: &str) -> Option<u64> {
    text.lines()
        .find(|line| line.starts_with(key))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kib| kib.saturating_mul(1024))
}

#[cfg(target_os = "macos")]
fn command_u64(program: &str, args: &[&str]) -> Option<u64> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then_some(output.stdout)
        .and_then(|stdout| String::from_utf8(stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok())
}

fn nonzero(value: u64) -> Option<u64> {
    (value > 0).then_some(value)
}

fn selection_config(workload: &WorkloadProfile) -> SelectionConfig {
    let mut config = SelectionConfig {
        workload: workload.clone(),
        ..SelectionConfig::default()
    };
    // The validator uses DEFAULT_CTX_SIZE as runtime capacity, but the
    // steady-decode speed estimate is about the active prompt/past length for a
    // request. Keep the workload's default `expected_prompt_tokens` intact so
    // the primary recommendation predicts the same occupied KV length that the
    // validator's primary-context prompt generator targets. Memory fit still
    // sizes the KV allocation through the normal context requirement path.
    config.weights = config.workload.default_weights();
    config
}

fn decode_context_tokens_for_validation(config: &SelectionConfig, model: &ModelProfile) -> u32 {
    let required = config
        .workload
        .requirements
        .min_context_tokens
        .or(config.workload.interaction.expected_prompt_tokens)
        .unwrap_or(4096)
        .max(1);
    let target = model
        .context_length
        .map_or(required, |native| required.min(native));
    // The model-shaped full-token probes are validation evidence for the
    // source graph that Skippy/llama.cpp submits, not only for the logical
    // attention length a request has occupied so far. Source inspection matters
    // here: `llama_kv_cache::cpy_k()` and `cpy_v()` return GGML_SET_ROWS nodes
    // whose output tensors are the layer KV-cache buffers. Those buffers are
    // shaped by the runtime context capacity (`--ctx-size` / workload context
    // requirement), and ABI graph inventory reports their full tensor bytes even
    // in single-token steady decode. For mid-size models the transformer
    // matmuls dominate and this distinction is easy to miss. For tiny models,
    // these capacity-shaped cache/runtime nodes can be a large part of token
    // time, so probing only `expected_prompt_tokens` under-measures the
    // source-visible decode boundary.
    target.max(1)
}

fn active_decode_context_tokens_for_validation(
    config: &SelectionConfig,
    model: &ModelProfile,
    capacity_tokens: u32,
) -> u32 {
    let prompt_tokens = config
        .workload
        .interaction
        .expected_prompt_tokens
        .unwrap_or_else(|| capacity_tokens.saturating_div(2))
        .max(1);
    let generated_prefix_tokens = config
        .workload
        .interaction
        .expected_output_tokens
        .unwrap_or_default()
        .saturating_div(2);
    let occupied = prompt_tokens.saturating_add(generated_prefix_tokens).max(1);
    let native = model.context_length.unwrap_or(capacity_tokens);
    let capped = occupied.min(capacity_tokens).min(native).max(1);

    // llama.cpp does not usually build decode attention over the full KV
    // allocation. `llama_kv_cache::get_n_kv()` pads the currently occupied
    // cells with `max(n_pad, 256)` so graph topology remains reusable. Mirror
    // that source rule for synthetic full-token probes: SET_ROWS still sees the
    // allocation capacity through `context_tokens`, while Flash Attention, K/V
    // cache views, and KQ mask shape use this padded active n_kv.
    const LLAMA_DECODE_N_KV_PAD: u32 = 256;
    let padded = capped.saturating_add(LLAMA_DECODE_N_KV_PAD - 1) / LLAMA_DECODE_N_KV_PAD
        * LLAMA_DECODE_N_KV_PAD;
    padded.min(capacity_tokens).min(native).max(1)
}

fn score_workloads(
    hardware: &HardwareProfile,
    model: &ModelProfile,
) -> Vec<WorkloadRecommendation> {
    workload_profiles()
        .into_iter()
        .map(|(workload, profile)| WorkloadRecommendation {
            workload: workload.into(),
            recommendation: score_model(hardware, model, &selection_config(&profile)),
        })
        .collect()
}

fn workload_profiles() -> Vec<(&'static str, WorkloadProfile)> {
    vec![
        ("chat", WorkloadProfile::chat()),
        ("coding_agent", WorkloadProfile::coding_agent()),
        ("tool_calling", WorkloadProfile::tool_calling()),
        ("summarization", WorkloadProfile::summarization()),
        ("embedding", WorkloadProfile::embedding()),
        ("reranking", WorkloadProfile::reranking()),
        ("vision_chat", WorkloadProfile::vision_chat()),
        ("general_generation", WorkloadProfile::general_generation()),
    ]
}

fn primary_workload_label() -> &'static str {
    "chat"
}

fn primary_workload_profile() -> WorkloadProfile {
    WorkloadProfile::chat()
}

fn validation_prompt() -> &'static str {
    "You are validating local model throughput. Write a concise explanation of how memory bandwidth affects token generation speed."
}

fn benchmark_scenarios() -> Vec<BenchmarkScenarioSpec> {
    vec![
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::SteadyDecode,
            name: "steady_decode",
            fit_metric: "estimated_decode_tokens_per_sec",
            prompt: validation_prompt().into(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            warmup_tokens: DEFAULT_WARMUP_TOKENS,
            request_count: 3,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::PrimaryContextSteadyDecode,
            name: "steady_decode_primary_context",
            fit_metric: "estimated_decode_tokens_per_sec",
            prompt: validation_prompt().into(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            // This scenario intentionally validates a single request at the
            // fitter's primary/default decode context. The short
            // steady_decode scenario can safely use warmup plus repeated
            // independent sessions because each session occupies only a small
            // number of KV cells. At primary context, doing the same thing
            // changes the experiment: a long warmup session and several long
            // measured sessions can coexist in the server's KV arena and make
            // an otherwise valid single local request fail with KV exhaustion.
            //
            // The fitter prediction being checked here is "how fast does one
            // local request decode once its prompt has occupied the normal
            // target context?", not "how many unrelated primary-context
            // sessions can the runtime keep resident at once". Keep benchmark
            // mechanics from becoming the source of a false local-fit failure.
            warmup_tokens: 0,
            request_count: 1,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::Prefill,
            name: "prefill",
            fit_metric: "estimated_prefill_tokens_per_sec",
            prompt: first_token_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: FIRST_TOKEN_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 1,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::FirstToken,
            name: "first_token",
            fit_metric: "estimated_first_token_ms",
            prompt: first_token_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: FIRST_TOKEN_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 1,
            reuse_session: false,
        },
        BenchmarkScenarioSpec {
            kind: BenchmarkScenarioKind::KvWarmReuse,
            name: "kv_warm_reuse",
            fit_metric: "warm_reuse_second_request_tokens_per_sec",
            prompt: kv_reuse_prompt(),
            ctx_size: DEFAULT_CTX_SIZE,
            max_new_tokens: KV_WARM_REUSE_MAX_NEW_TOKENS,
            warmup_tokens: 0,
            request_count: 2,
            reuse_session: true,
        },
    ]
}

fn selected_benchmark_scenarios(args: &Args) -> Vec<BenchmarkScenarioSpec> {
    let scenarios = benchmark_scenarios();
    if args.benchmark_scenarios.is_empty() {
        return scenarios;
    }
    if args
        .benchmark_scenarios
        .iter()
        .any(|scenario| scenario == "all")
    {
        return scenarios;
    }
    let mut selected = Vec::new();
    for requested in &args.benchmark_scenarios {
        let scenario = scenarios
            .iter()
            .find(|scenario| scenario.name == requested)
            .cloned()
            .expect("scenario names are validated during argument parsing");
        if !selected
            .iter()
            .any(|existing: &BenchmarkScenarioSpec| existing.name == scenario.name)
        {
            selected.push(scenario);
        }
    }
    selected
}

fn fit_input_contract() -> FitInputContract {
    FitInputContract {
        hardware_fields_consumed: vec![
            "memory.available_system_bytes",
            "memory.available_unified_bytes",
            "accelerators.kind",
            "accelerators.backend",
            "accelerators.available_memory_bytes",
            "accelerators.memory_bandwidth_bytes_per_sec",
            "accelerators.decode_effective_bandwidth_bytes_per_sec",
            "accelerators.decode_fixed_overhead_ms",
            "accelerators.decode_runtime_overhead_ms",
            "accelerators.post_prefill_decode_overhead_ms",
            "accelerators.bandwidth_source",
            "accelerators.benchmark_noise_pct",
            "accelerators.compute_tflops_fp16",
            "accelerators.prefill_matmul_tflops_fp16",
            "accelerators.prefill_ubatch_matmul_tflops_fp16",
            "accelerators.prefill_moe_matmul_tflops_fp16",
            "accelerators.sampler_history_us_per_token",
            "accelerators.sampler_vocab_us_per_token",
            "accelerators.unified_memory",
            "cpu.memory_bandwidth_bytes_per_sec",
            "cpu.compute_tflops_fp16",
            "cpu.post_prefill_decode_overhead_ms",
            "cpu.prefill_matmul_tflops_fp16",
            "cpu.prefill_ubatch_matmul_tflops_fp16",
            "cpu.prefill_moe_matmul_tflops_fp16",
            "cpu.sampler_history_us_per_token",
            "cpu.sampler_vocab_us_per_token",
        ],
        model_fields_consumed: vec![
            "architecture",
            "architecture_class",
            "weight_coverage",
            "file_size_bytes",
            "tensor_bytes",
            "base_resident_bytes",
            "expert_tensor_bytes",
            "tensor_group_bytes",
            "tensor_matmul",
            "quantization",
            "layer_count",
            "hidden_size",
            "ffn_size",
            "attention_heads",
            "kv_heads",
            "key_length",
            "value_length",
            "context_length",
            "expert_count",
            "expert_used_count",
            "rope",
            "tokenizer",
            "capability_evidence",
        ],
        validation_backend: "skippy-bench local-single plus abi-decode-probe using skippy-server/llama.cpp full-model inference and the native skippy decode benchmark ABI",
        validation_note: "Validation observations exercise real GGML/llama.cpp model execution. They are reported as evidence only; observed model throughput and ABI probe throughput are not fed back into metadata-only fit scoring.",
    }
}

fn first_token_prompt() -> String {
    let seed = "Summarize this benchmark context into one operational takeaway: local inference fit depends on model bytes, active layers, KV cache pressure, backend overhead, and memory bandwidth.";
    std::iter::repeat_n(seed, 96).collect::<Vec<_>>().join("\n")
}

fn kv_reuse_prompt() -> String {
    "You are an agent inside a short tool loop. Inspect the same project context again and answer with the next concrete action.".into()
}

fn primary_download_path(
    artifact: &ResolvedModelArtifact,
    downloaded_paths: &[PathBuf],
) -> Result<PathBuf> {
    let primary = Path::new(&artifact.primary_file);
    downloaded_paths
        .iter()
        .find(|path| path.ends_with(primary))
        .or_else(|| downloaded_paths.first())
        .cloned()
        .with_context(|| format!("download produced no files for {}", artifact.canonical_ref))
}

fn download_progress(args: &Args, model_ref: &str) -> Option<ModelDownloadProgress> {
    args.show_progress.then(|| {
        let renderer = TerminalDownloadProgress::new(model_ref);
        ModelDownloadProgress::new(move |event| renderer.report(event))
    })
}

fn summarize(args: &Args, models: &[ModelValidationReport], tolerance: f64) -> ValidationSummary {
    let mut summary = ValidationSummary {
        model_count: models.len(),
        ..ValidationSummary::default()
    };
    let mut ratios = Vec::new();
    for model in models {
        count_model_summary(model, tolerance, &mut summary, &mut ratios);
    }
    ratios.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    summary.median_observed_over_fit = (!ratios.is_empty()).then(|| median(&ratios));
    summary.mean_observed_over_fit = mean(&ratios);
    summary.median_absolute_percent_error = median_absolute_percent_error(&ratios);
    summary.scenario_summaries = summarize_scenarios(args, models, tolerance);
    summary
}

fn summarize_scenarios(
    args: &Args,
    models: &[ModelValidationReport],
    tolerance: f64,
) -> Vec<ScenarioValidationSummary> {
    selected_benchmark_scenarios(args)
        .into_iter()
        .map(|scenario| summarize_scenario(models, scenario.name, tolerance))
        .collect()
}

fn summarize_scenario(
    models: &[ModelValidationReport],
    scenario: &str,
    tolerance: f64,
) -> ScenarioValidationSummary {
    let mut summary = ScenarioValidationSummary {
        scenario: scenario.into(),
        ..ScenarioValidationSummary::default()
    };
    let mut ratios = Vec::new();

    for model in models {
        let Some(benchmark) = model
            .benchmarks
            .iter()
            .find(|entry| entry.scenario == scenario)
        else {
            continue;
        };
        summary.sample_count += usize::from(benchmark.observed_over_fit.is_some());
        count_scenario_verdict(model, benchmark, tolerance, &mut summary, &mut ratios);
    }

    ratios.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    summary.median_observed_over_fit = (!ratios.is_empty()).then(|| median(&ratios));
    summary.mean_observed_over_fit = mean(&ratios);
    summary.median_absolute_percent_error = median_absolute_percent_error(&ratios);
    summary
}

fn count_scenario_verdict(
    model: &ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
    tolerance: f64,
    summary: &mut ScenarioValidationSummary,
    ratios: &mut Vec<f64>,
) {
    match benchmark.verdict.as_str() {
        "match" => summary.matched_count += 1,
        "slower-than-fit" => summary.slower_than_fit_count += 1,
        "faster-than-fit" => summary.faster_than_fit_count += 1,
        "inconclusive-noisy" => summary.noisy_count += 1,
        "skipped" => summary.skipped_count += 1,
        "runtime-error" => summary.runtime_error_count += 1,
        "error" => summary.error_count += 1,
        _ => {}
    }
    if let Some(classification) = steady_decode_classification(model, benchmark) {
        count_decode_probe_classification(classification, summary);
    }
    if steady_decode_accuracy_exclusion(model, benchmark).is_some() {
        return;
    }
    if !accuracy_gated_verdict(&benchmark.verdict) {
        return;
    }
    if let Some(ratio) = benchmark.observed_over_fit {
        if (ratio - 1.0).abs() <= tolerance {
            summary.within_tolerance_count += 1;
        }
        ratios.push(ratio);
    }
}

fn steady_decode_accuracy_exclusion<'a>(
    model: &'a ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
) -> Option<&'a str> {
    if benchmark.scenario != "steady_decode" || !accuracy_gated_verdict(&benchmark.verdict) {
        return None;
    }
    steady_decode_accuracy_exclusion_for_model(model)
}

fn steady_decode_classification<'a>(
    model: &'a ModelValidationReport,
    benchmark: &BenchmarkScenarioSummary,
) -> Option<&'a str> {
    if benchmark.scenario != "steady_decode" {
        return None;
    }
    steady_decode_classification_for_model(model)
}

fn steady_decode_classification_for_model(model: &ModelValidationReport) -> Option<&str> {
    model
        .decode_probe_diagnostic
        .as_ref()
        .map(|diagnostic| diagnostic.classification.as_str())
}

fn steady_decode_accuracy_exclusion_for_model(model: &ModelValidationReport) -> Option<&str> {
    let classification = steady_decode_classification_for_model(model)?;
    // The ±10% accuracy score is supposed to evaluate the metadata-only fit
    // estimate against a stable local decode path. When Skippy's observed
    // benchmark and the ABI decode probe disagree, that row is still valuable
    // evidence, but it is not clean estimator evidence: it mixes metadata cost,
    // probe representativeness, runtime/server path overhead, cache/session
    // behavior, and backend runtime state.
    //
    // Keep these cases visible as separate summary buckets instead of widening
    // the tolerance or silently counting them as fit failures. That follows the
    // repo-wide empirical rule: report residual misses honestly and avoid
    // tuning the estimator around a noisy local run.
    match classification {
        "steady_path_overhead_mismatch"
        | "steady_path_overhead_mismatch_noisy"
        | "estimate_and_probe_agree_noisy_requests"
        | "primary_estimate_and_probe_agree_context_rescore_unstable"
        | "scenario_estimate_and_probe_agree_noisy_requests"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch"
        | "sampler_sync_residual"
        | "sampler_sync_residual_noisy"
        | "llama_source_boundary_residual"
        | "llama_source_boundary_residual_noisy"
        | "decode_submission_residual"
        | "decode_submission_residual_noisy"
        | "decode_and_sampler_residual"
        | "decode_and_sampler_residual_noisy"
        | "context_rescore_unstable"
        | "abi_probe_noisy"
        | "validation_request_noise"
        | "missing_representative_decode_probe"
        | "probe_not_representative"
        | "unstable_probe_geometry" => Some(classification),
        _ => None,
    }
}

fn count_decode_probe_classification(
    classification: &str,
    summary: &mut ScenarioValidationSummary,
) {
    match classification {
        "metadata_estimate_miss" => summary.metadata_estimate_miss_count += 1,
        "steady_path_overhead_mismatch"
        | "steady_path_overhead_mismatch_noisy"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch"
        | "sampler_sync_residual"
        | "sampler_sync_residual_noisy"
        | "llama_source_boundary_residual"
        | "llama_source_boundary_residual_noisy"
        | "decode_submission_residual"
        | "decode_submission_residual_noisy"
        | "decode_and_sampler_residual"
        | "decode_and_sampler_residual_noisy" => summary.runtime_path_mismatch_count += 1,
        "abi_probe_noisy"
        | "estimate_and_probe_agree_noisy_requests"
        | "primary_estimate_and_probe_agree_context_rescore_unstable"
        | "scenario_estimate_and_probe_agree_noisy_requests"
        | "validation_request_noise"
        | "missing_representative_decode_probe"
        | "probe_not_representative"
        | "unstable_probe_geometry" => {
            summary.probe_mismatch_count += 1;
        }
        _ => {}
    }
}

fn count_decode_probe_classification_for_model(
    classification: &str,
    summary: &mut ValidationSummary,
) {
    match classification {
        "metadata_estimate_miss" => summary.metadata_estimate_miss_count += 1,
        "steady_path_overhead_mismatch"
        | "steady_path_overhead_mismatch_noisy"
        | "runtime_path_mismatch"
        | "mixed_estimate_and_runtime_mismatch"
        | "sampler_sync_residual"
        | "sampler_sync_residual_noisy"
        | "llama_source_boundary_residual"
        | "llama_source_boundary_residual_noisy"
        | "decode_submission_residual"
        | "decode_submission_residual_noisy"
        | "decode_and_sampler_residual"
        | "decode_and_sampler_residual_noisy" => summary.runtime_path_mismatch_count += 1,
        "abi_probe_noisy"
        | "estimate_and_probe_agree_noisy_requests"
        | "primary_estimate_and_probe_agree_context_rescore_unstable"
        | "scenario_estimate_and_probe_agree_noisy_requests"
        | "validation_request_noise"
        | "missing_representative_decode_probe"
        | "probe_not_representative"
        | "unstable_probe_geometry" => {
            summary.probe_mismatch_count += 1;
        }
        _ => {}
    }
}

fn count_model_summary(
    model: &ModelValidationReport,
    tolerance: f64,
    summary: &mut ValidationSummary,
    ratios: &mut Vec<f64>,
) {
    match model.benchmark.verdict.as_str() {
        "match" => summary.matched_count += 1,
        "slower-than-fit" => summary.slower_than_fit_count += 1,
        "faster-than-fit" => summary.faster_than_fit_count += 1,
        "inconclusive-noisy" => summary.noisy_count += 1,
        "skipped" => summary.skipped_count += 1,
        "runtime-error" => summary.runtime_error_count += 1,
        "error" => summary.error_count += 1,
        _ => {}
    }
    if model.benchmark.attempted {
        summary.benchmarked_count += 1;
    }
    if let Some(classification) = steady_decode_classification_for_model(model) {
        count_decode_probe_classification_for_model(classification, summary);
    }
    if steady_decode_accuracy_exclusion_for_model(model).is_some() {
        return;
    }
    if !accuracy_gated_verdict(&model.benchmark.verdict) {
        return;
    }
    if let Some(ratio) = model.benchmark.observed_over_fit {
        if (ratio - 1.0).abs() <= tolerance {
            summary.within_tolerance_count += 1;
        }
        ratios.push(ratio);
    }
}

fn accuracy_gated_verdict(verdict: &str) -> bool {
    matches!(verdict, "match" | "slower-than-fit" | "faster-than-fit")
}

fn median_absolute_percent_error(ratios: &[f64]) -> Option<f64> {
    if ratios.is_empty() {
        return None;
    }
    let mut errors = ratios
        .iter()
        .map(|ratio| (ratio - 1.0).abs() * 100.0)
        .collect::<Vec<_>>();
    errors.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    Some(median(&errors))
}

fn error_report(input_ref: String, error: String) -> ModelValidationReport {
    ModelValidationReport {
        input_ref,
        resolved_ref: None,
        artifact: None,
        downloaded_paths: Vec::new(),
        primary_gguf_path: None,
        model_profile: None,
        recommendation: None,
        fit_interpretation: None,
        runtime_diagnostic: None,
        recommendations: Vec::new(),
        abi_decode_probe: None,
        context_aligned_abi_decode_probe: None,
        decode_probe_diagnostic: None,
        graph_inventory_diagnostic: None,
        operation_bucket_diagnostic: None,
        model_specific_decode_kernel_probes: Vec::new(),
        model_specific_probe_errors: Vec::new(),
        benchmarks: Vec::new(),
        benchmark: BenchmarkSummary {
            verdict: "error".into(),
            errors: vec![error.clone()],
            ..BenchmarkSummary::default()
        },
        errors: vec![error],
    }
}

fn print_markdown_table(rows: &[ModelValidationReport]) {
    println!(
        "| model_ref | fit | meaning | backend | primary tok/s | primary ctx | steady scenario tok/s | steady ctx | primary-context scenario tok/s | primary-context ctx | abi tok/s | sample/sync tok/s | abi overhead | steady range | steady median | steady observed/scenario-fit | steady observed/primary-fit | primary-context median | primary-context observed/scenario-fit | primary-context observed/primary-fit | steady/abi | decode diag | steady | primary-context | first-token | kv-reuse |"
    );
    println!(
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|"
    );
    for row in rows {
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            fit_status(row),
            display_fit_meaning(row),
            display_selected_backend(row),
            display_estimated_tps(row),
            display_estimated_decode_context(row),
            display_steady_estimated_tps(row),
            display_steady_prediction_context(row),
            display_scenario_estimated_tps(row, "steady_decode_primary_context"),
            display_scenario_prediction_context(row, "steady_decode_primary_context"),
            display_abi_decode_probe(row),
            display_abi_sampling_probe(row),
            display_abi_non_eval_overhead(row),
            display_steady_estimated_range(row),
            display_steady_observed(row),
            display_steady_observed_over_fit(row),
            display_steady_observed_over_primary(row),
            display_scenario_observed(row, "steady_decode_primary_context"),
            display_scenario_observed_over_fit(row, "steady_decode_primary_context"),
            display_scenario_observed_over_primary(row, "steady_decode_primary_context"),
            display_observed_over_abi(row),
            display_decode_probe_classification(row),
            scenario_verdict(row, "steady_decode"),
            scenario_verdict(row, "steady_decode_primary_context"),
            scenario_verdict(row, "first_token"),
            scenario_verdict(row, "kv_warm_reuse"),
        );
    }
    print_dense_probe_ladder_table(rows);
    print_graph_inventory_diagnostic_table(rows);
    print_operation_bucket_diagnostic_table(rows);
}

fn print_graph_inventory_diagnostic_table(rows: &[ModelValidationReport]) {
    let rows_with_inventory = rows
        .iter()
        .filter(|row| {
            row.graph_inventory_diagnostic
                .as_ref()
                .is_some_and(|diagnostic| diagnostic.available)
        })
        .collect::<Vec<_>>();
    if rows_with_inventory.is_empty() {
        return;
    }

    println!();
    println!("Graph inventory diagnostic");
    println!(
        "| model_ref | status | ABI graph nodes | selected probe nodes | probe/ABI nodes | selected probe layers | selected ctx | ABI ctx | synthetic/ABI bucket misses | synthetic/ABI node delta | synthetic/ABI src0 delta | synthetic/ABI src1 delta | synthetic/ABI output delta | transformer src0/meta | transformer+unclassified src0/meta | unclassified matmul src0 | transformer nodes graph/meta | transformer ms / ABI ms | notes |"
    );
    println!(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    );
    for row in rows_with_inventory {
        let diagnostic = row
            .graph_inventory_diagnostic
            .as_ref()
            .expect("filtered row has graph inventory diagnostic");
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            diagnostic.status,
            display_option_u64(diagnostic.graph_node_count),
            display_option_u64(diagnostic.selected_probe_node_count),
            display_option_ratio(diagnostic.selected_probe_nodes_over_abi),
            display_option_u32(diagnostic.selected_transformer_probe_layers),
            display_option_u32(diagnostic.selected_probe_context_tokens),
            display_option_u32(diagnostic.abi_graph_context_tokens),
            display_option_u64(diagnostic.selected_probe_inventory_bucket_mismatch_count),
            display_option_u64(diagnostic.selected_probe_inventory_abs_node_delta),
            display_option_u64(diagnostic.selected_probe_inventory_abs_src0_delta_bytes),
            display_option_u64(diagnostic.selected_probe_inventory_abs_src1_delta_bytes),
            display_option_u64(diagnostic.selected_probe_inventory_abs_output_delta_bytes),
            display_option_ratio(diagnostic.graph_transformer_src0_over_metadata),
            display_option_ratio(diagnostic.graph_transformer_plus_unclassified_src0_over_metadata),
            diagnostic.graph_unclassified_matmul_src0_bytes,
            display_graph_node_ratio(diagnostic),
            display_option_ratio(diagnostic.estimated_transformer_over_abi),
            display_graph_inventory_notes(diagnostic),
        );
    }
}

fn print_operation_bucket_diagnostic_table(rows: &[ModelValidationReport]) {
    let rows_with_buckets = rows
        .iter()
        .filter(|row| {
            row.operation_bucket_diagnostic
                .as_ref()
                .is_some_and(|diagnostic| diagnostic.available)
        })
        .collect::<Vec<_>>();
    if rows_with_buckets.is_empty() {
        return;
    }

    println!();
    println!("Operation bucket diagnostic");
    println!(
        "| model_ref | bucket | source | est ms | est share | graph families | graph nodes | graph src0 | graph src1 | graph output | src0/meta | notes |"
    );
    println!("|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|");
    for row in rows_with_buckets {
        let diagnostic = row
            .operation_bucket_diagnostic
            .as_ref()
            .expect("filtered row has operation bucket diagnostic");
        for bucket in &diagnostic.buckets {
            println!(
                "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                row.input_ref,
                bucket.bucket,
                bucket.source,
                display_option_ms(bucket.estimated_ms),
                display_option_ratio(bucket.estimated_share_of_selected_ms),
                bucket.graph_families.join(", "),
                bucket.graph_node_count,
                bucket.graph_src0_bytes,
                bucket.graph_src1_bytes,
                bucket.graph_output_bytes,
                display_option_ratio(bucket.graph_src0_over_metadata),
                display_operation_bucket_notes(bucket),
            );
        }
    }
}

fn print_dense_probe_ladder_table(rows: &[ModelValidationReport]) {
    let rows_with_dense_probes = rows
        .iter()
        .filter(|row| dense_probe_ladder_available(row))
        .collect::<Vec<_>>();
    if rows_with_dense_probes.is_empty() {
        return;
    }

    println!();
    println!("Dense probe ladder diagnostic");
    println!(
        "| model_ref | observed tok/s | fit tok/s | selected probe | l1 tok/s (GB/s) | l4 tok/s (GB/s) | l8 tok/s (GB/s) | l16 tok/s (GB/s) | full-depth tok/s (GB/s) |"
    );
    println!("|---|---:|---:|---|---:|---:|---:|---:|---:|");
    for row in rows_with_dense_probes {
        println!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |",
            row.input_ref,
            display_steady_observed(row),
            display_steady_estimated_tps(row),
            display_selected_dense_probe(row),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(1)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(4)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(8)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::Layers(16)),
            display_dense_probe_ladder_cell(row, DenseProbeLadderSlot::FullDepth),
        );
    }
}

#[derive(Clone, Copy, Debug)]
enum DenseProbeLadderSlot {
    Layers(u32),
    FullDepth,
}

fn dense_probe_ladder_available(row: &ModelValidationReport) -> bool {
    selected_dense_transformer_group(row).is_some()
        && row
            .model_specific_decode_kernel_probes
            .iter()
            .any(is_dense_llama_graph_probe)
}

fn display_selected_dense_probe(row: &ModelValidationReport) -> String {
    selected_dense_transformer_group(row)
        .and_then(|group| group.probe_name.as_deref())
        .map(|name| format!("`{name}`"))
        .unwrap_or_else(|| "-".into())
}

fn display_dense_probe_ladder_cell(
    row: &ModelValidationReport,
    slot: DenseProbeLadderSlot,
) -> String {
    let Some(probe) = dense_probe_for_ladder_slot(row, slot) else {
        return "-".into();
    };
    let Some(implied_tps) = dense_probe_implied_tokens_per_second(row, probe) else {
        return format!("- ({:.0})", probe.effective_gbps);
    };
    format!("{implied_tps:.1} ({:.0})", probe.effective_gbps)
}

fn dense_probe_for_ladder_slot(
    row: &ModelValidationReport,
    slot: DenseProbeLadderSlot,
) -> Option<&DecodeKernelProbe> {
    let group = selected_dense_transformer_group(row)?;
    let target_layers = match slot {
        DenseProbeLadderSlot::Layers(layers) => layers,
        DenseProbeLadderSlot::FullDepth => row.model_profile.as_ref()?.layer_count?,
    };
    row.model_specific_decode_kernel_probes
        .iter()
        .filter(|probe| {
            is_dense_llama_graph_probe(probe)
                && dense_probe_layers(probe) == target_layers
                && group.probe_rows.is_none_or(|rows| probe.rows == rows)
                && group.probe_cols.is_none_or(|cols| probe.cols == cols)
                && probe.tensor_type.eq_ignore_ascii_case(&group.tensor_type)
        })
        .max_by(|left, right| {
            left.effective_gbps
                .partial_cmp(&right.effective_gbps)
                .unwrap_or(Ordering::Equal)
        })
}

fn dense_probe_implied_tokens_per_second(
    row: &ModelValidationReport,
    probe: &DecodeKernelProbe,
) -> Option<f64> {
    // This is a diagnostic, not an alternate scoring path. It asks:
    // "If this synthetic dense graph row supplied only the transformer-block
    // timing, and every other cost term from the already-produced recommendation
    // stayed the same, what tok/s would the model-fit estimate imply?"
    //
    // That framing lets the report compare l1/l4/l8/l16/full-depth probe rows
    // against observed steady decode without silently changing model-fit's
    // deterministic selector. It also exposes synthetic graph artifacts: if a
    // full-depth row whipsaws while observed tok/s is stable, the row is
    // validation evidence, not a better estimator.
    let recommendation = row.recommendation.as_ref()?;
    let breakdown = recommendation.decode_cost_breakdown.as_ref()?;
    let group = selected_dense_transformer_group(row)?;
    let model_layers = f64::from(row.model_profile.as_ref()?.layer_count?);
    let probe_layers = f64::from(dense_probe_layers(probe).max(1));
    let probe_elapsed_ms = probe.elapsed_ms?;
    let variable_probe_ms = (probe_elapsed_ms - f64::from(breakdown.fixed_overhead_ms)).max(0.0);
    let candidate_block_ms = variable_probe_ms * (model_layers / probe_layers);
    let original_block_ms = f64::from(group.bandwidth_ms);
    let other_bandwidth_ms = (f64::from(breakdown.bandwidth_ms) - original_block_ms).max(0.0);
    let candidate_bandwidth_ms = other_bandwidth_ms + candidate_block_ms;
    let overhead_ms = f64::from(breakdown.fixed_overhead_ms)
        + f64::from(breakdown.runtime_overhead_ms)
        + f64::from(breakdown.measured_graph_overhead_ms)
        + f64::from(breakdown.architecture_overhead_ms)
        + f64::from(breakdown.sampled_decode_sampler_ms);
    let candidate_total_ms =
        candidate_bandwidth_ms.max(f64::from(breakdown.compute_ms)) + overhead_ms;
    (candidate_total_ms > 0.0).then_some(1000.0 / candidate_total_ms)
}

fn selected_dense_transformer_group(
    row: &ModelValidationReport,
) -> Option<&model_fit::DecodeCostGroupBreakdown> {
    row.recommendation
        .as_ref()?
        .decode_cost_breakdown
        .as_ref()?
        .groups
        .iter()
        .find(|group| group.group == "transformer_block" && group.probe_name.is_some())
}

fn is_dense_llama_graph_probe(probe: &DecodeKernelProbe) -> bool {
    let name = probe.name.to_ascii_lowercase();
    name.contains("llama_graph")
}

fn dense_probe_layers(probe: &DecodeKernelProbe) -> u32 {
    let name = probe.name.to_ascii_lowercase();
    let Some((_, suffix)) = name.split_once("_llama_graph_l") else {
        return 1;
    };
    let digits = suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse::<u32>().unwrap_or(1).max(1)
}

fn steady_benchmark_for_decode_diagnostic(
    benchmarks: &[BenchmarkScenarioSummary],
) -> Option<&BenchmarkScenarioSummary> {
    // Prefer the normal steady-decode scenario when present because it is the
    // stable smoke-test shape used by broad validation. Some focused runs only
    // request `steady_decode_primary_context` so the prompt length matches the
    // primary workload estimate. That scenario is still sampled steady decode:
    // it exercises the same `skippy_decode_step_sampled()` path and is valid
    // evidence for ABI-vs-observed and sampler/logits-sync diagnostics.
    benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == "steady_decode")
        .or_else(|| {
            benchmarks
                .iter()
                .find(|benchmark| benchmark.scenario == "steady_decode_primary_context")
        })
}

fn decode_probe_diagnostic(
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
    steady_benchmark: Option<&BenchmarkScenarioSummary>,
    model_specific_probe_errors: &[String],
) -> Option<DecodeProbeDiagnostic> {
    if !matches!(
        recommendation.fit_status,
        FitStatus::FitsLocal | FitStatus::FitsWithWarning
    ) {
        return None;
    }

    let predicted = recommendation
        .estimated_decode_tokens_per_sec
        .map(f64::from);
    let scenario_predicted = steady_benchmark.and_then(|benchmark| benchmark.predicted);
    let abi = abi_probe.and_then(|probe| probe.tokens_per_second);
    let observed = steady_benchmark
        .and_then(|benchmark| benchmark.observed)
        .or_else(|| {
            steady_benchmark.and_then(|benchmark| benchmark.benchmark.median_tokens_per_sec)
        });
    let observed_over_fit = ratio(observed, predicted);
    let observed_over_scenario_fit = ratio(observed, scenario_predicted);
    let abi_over_fit = ratio(abi, predicted);
    let abi_over_scenario_fit = ratio(abi, scenario_predicted);
    let observed_over_abi = ratio(observed, abi);
    let observed_vs_fit = throughput_ratio_verdict(observed_over_fit);
    let observed_vs_scenario_fit = throughput_ratio_verdict(observed_over_scenario_fit);
    let abi_vs_fit = throughput_ratio_verdict(abi_over_fit);
    let abi_vs_scenario_fit = throughput_ratio_verdict(abi_over_scenario_fit);
    let observed_vs_abi = throughput_ratio_verdict(observed_over_abi);
    let predicted_decode_submission_ms_per_token =
        predicted_decode_submission_ms_per_token(recommendation);
    let abi_decode_call_ms_per_token = abi_decode_call_ms_per_token(abi_probe);
    let decode_submission_residual_ms_per_token = positive_delta(
        abi_decode_call_ms_per_token,
        predicted_decode_submission_ms_per_token,
    );
    let decode_submission_residual_share_of_predicted =
        residual_share_of_selected_fit(decode_submission_residual_ms_per_token, recommendation);
    let predicted_sampler_sync_ms_per_token = predicted_sampler_sync_ms_per_token(recommendation);
    let abi_sampling_ms_per_token = abi_sampling_ms_per_token(abi_probe);
    let abi_logits_ready_ms_per_token = abi_logits_ready_ms_per_token(abi_probe);
    let abi_logits_scan_ms_per_token = abi_logits_scan_ms_per_token(abi_probe);
    let abi_sampling_over_selected_fit = abi_sampling_over_selected_fit(recommendation, abi_probe);
    let sampler_sync_residual_ms_per_token = positive_delta(
        abi_sampling_ms_per_token,
        predicted_sampler_sync_ms_per_token,
    );
    let sampler_sync_residual_share_of_predicted =
        residual_share_of_selected_fit(sampler_sync_residual_ms_per_token, recommendation);
    let selected_full_token_handoff_probe = selected_full_token_handoff_probe(recommendation);
    let selected_full_token_source_sampled_probe =
        selected_full_token_source_sampled_probe(recommendation);
    let selected_fit_probe_count = selected_fit_probe_count(recommendation);
    let selected_fit_probe_max_spread_pct = selected_fit_probe_max_spread_pct(recommendation);
    let abi_probe_noisy = abi_probe
        .and_then(|probe| probe.spread_pct)
        .is_some_and(|spread| spread > DEFAULT_MAX_SPREAD * 100.0);
    let request_spread_pct = steady_benchmark
        .and_then(|benchmark| benchmark.benchmark.request_spread_pct)
        .filter(|spread| spread.is_finite());
    let request_decode_noisy =
        request_spread_pct.is_some_and(|spread| spread > DEFAULT_MAX_SPREAD * 100.0);
    let classification = decode_probe_classification(DecodeProbeClassificationInput {
        predicted,
        abi,
        observed,
        missing_representative_model_probe: missing_representative_model_probe(
            recommendation,
            model_specific_probe_errors,
        ),
        abi_probe_noisy,
        request_decode_noisy,
        request_spread_pct,
        observed_over_fit,
        observed_over_scenario_fit,
        abi_over_fit,
        abi_over_scenario_fit,
        observed_over_abi,
        abi_sampling_over_selected_fit,
        selected_full_token_handoff_probe,
        selected_full_token_source_sampled_probe,
        decode_submission_residual_share_of_predicted,
        sampler_sync_residual_share_of_predicted,
    });
    let notes = decode_probe_notes(DecodeProbeNotesInput {
        observed_over_fit,
        abi_over_fit,
        observed_over_abi,
        decode_submission_residual_ms_per_token,
        abi_decode_call_ms_per_token,
        sampler_sync_residual_ms_per_token,
        abi_sampling_over_selected_fit,
        abi_logits_ready_ms_per_token,
        abi_logits_scan_ms_per_token,
        request_spread_pct: steady_benchmark
            .and_then(|benchmark| benchmark.benchmark.request_spread_pct),
        selected_fit_probe_max_spread_pct,
        selected_full_token_source_sampled_probe,
        classification: &classification,
    });
    Some(DecodeProbeDiagnostic {
        predicted_tokens_per_second: predicted,
        scenario_predicted_tokens_per_second: scenario_predicted,
        abi_tokens_per_second: abi,
        observed_tokens_per_second: observed,
        observed_over_fit,
        observed_over_scenario_fit,
        abi_over_fit,
        abi_over_scenario_fit,
        observed_over_abi,
        scenario_prediction_source: steady_benchmark.map(|benchmark| benchmark.prediction_source),
        observed_vs_fit,
        observed_vs_scenario_fit,
        abi_vs_fit,
        abi_vs_scenario_fit,
        observed_vs_abi,
        predicted_decode_submission_ms_per_token,
        abi_decode_call_ms_per_token,
        decode_submission_residual_ms_per_token,
        decode_submission_residual_share_of_predicted,
        predicted_sampler_sync_ms_per_token,
        abi_sampling_ms_per_token,
        abi_logits_ready_ms_per_token,
        abi_logits_scan_ms_per_token,
        abi_sampling_over_selected_fit,
        sampler_sync_residual_ms_per_token,
        sampler_sync_residual_share_of_predicted,
        selected_full_token_handoff_probe,
        selected_full_token_source_sampled_probe,
        selected_fit_probe_count,
        selected_fit_probe_max_spread_pct,
        classification,
        notes,
    })
}

fn selected_fit_probe_count(recommendation: &ModelRecommendation) -> usize {
    recommendation
        .decode_cost_breakdown
        .as_ref()
        .map(|breakdown| {
            breakdown
                .groups
                .iter()
                .filter(|group| group.probe_name.is_some())
                .count()
        })
        .unwrap_or_default()
}

fn selected_fit_probe_max_spread_pct(recommendation: &ModelRecommendation) -> Option<f64> {
    recommendation
        .decode_cost_breakdown
        .as_ref()?
        .groups
        .iter()
        .filter_map(|group| group.probe_spread_pct)
        .filter(|spread| spread.is_finite())
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
}

fn predicted_decode_submission_ms_per_token(recommendation: &ModelRecommendation) -> Option<f64> {
    recommendation
        .decode_cost_breakdown
        .as_ref()?
        .groups
        .iter()
        .find(|group| group.group == "decode_submission")
        .map(|group| f64::from(group.bandwidth_ms))
        .filter(|value| value.is_finite())
}

fn abi_decode_call_ms_per_token(abi_probe: Option<&AbiDecodeProbeSummary>) -> Option<f64> {
    let probe = abi_probe?;
    let decode_call_ms = probe.decode_call_ms?;
    let measured_tokens = probe.measured_tokens?;
    (measured_tokens > 0).then_some(decode_call_ms / measured_tokens as f64)
}

fn predicted_sampler_sync_ms_per_token(recommendation: &ModelRecommendation) -> Option<f64> {
    let breakdown = recommendation.decode_cost_breakdown.as_ref()?;
    let logits_readback_ms = breakdown
        .groups
        .iter()
        .filter(|group| group.group == "logits_readback")
        .map(|group| f64::from(group.bandwidth_ms))
        .sum::<f64>();
    Some(f64::from(breakdown.sampled_decode_sampler_ms) + logits_readback_ms)
}

fn abi_sampling_ms_per_token(abi_probe: Option<&AbiDecodeProbeSummary>) -> Option<f64> {
    let probe = abi_probe?;
    let sampling_ms = probe.sampling_ms?;
    let measured_tokens = probe.measured_tokens?;
    (measured_tokens > 0).then_some(sampling_ms / measured_tokens as f64)
}

fn abi_logits_ready_ms_per_token(abi_probe: Option<&AbiDecodeProbeSummary>) -> Option<f64> {
    let probe = abi_probe?;
    let logits_ready_ms = probe.logits_ready_ms?;
    let measured_tokens = probe.measured_tokens?;
    (measured_tokens > 0).then_some(logits_ready_ms / measured_tokens as f64)
}

fn abi_logits_scan_ms_per_token(abi_probe: Option<&AbiDecodeProbeSummary>) -> Option<f64> {
    let probe = abi_probe?;
    let logits_scan_ms = probe.logits_scan_ms?;
    let measured_tokens = probe.measured_tokens?;
    (measured_tokens > 0).then_some(logits_scan_ms / measured_tokens as f64)
}

fn abi_sampling_over_selected_fit(
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
) -> Option<f64> {
    let breakdown = recommendation.decode_cost_breakdown.as_ref()?;
    let selected_ms = f64::from(breakdown.selected_time_ms);
    if selected_ms <= 0.0 || !selected_ms.is_finite() {
        return None;
    }
    abi_sampling_ms_per_token(abi_probe).map(|sampling_ms| sampling_ms / selected_ms)
}

fn residual_share_of_selected_fit(
    residual_ms_per_token: Option<f64>,
    recommendation: &ModelRecommendation,
) -> Option<f64> {
    residual_ms_per_token
        .zip(recommendation.decode_cost_breakdown.as_ref())
        .and_then(|(residual, breakdown)| {
            (breakdown.selected_time_ms > 0.0)
                .then_some(residual / f64::from(breakdown.selected_time_ms))
        })
        .filter(|share| share.is_finite())
}

fn selected_full_token_handoff_probe(recommendation: &ModelRecommendation) -> bool {
    recommendation
        .decode_cost_breakdown
        .as_ref()
        .is_some_and(|breakdown| {
            breakdown.groups.iter().any(|group| {
                group.group == "full_token_graph"
                    && group.source.starts_with("probe_full_token_handoff")
            })
        })
}

fn selected_full_token_source_sampled_probe(recommendation: &ModelRecommendation) -> bool {
    recommendation
        .decode_cost_breakdown
        .as_ref()
        .is_some_and(|breakdown| {
            breakdown.groups.iter().any(|group| {
                group.group == "full_token_graph"
                    && group.source.starts_with("probe_full_token_source_sampled")
            })
        })
}

fn positive_delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right)
        .map(|(left, right)| (left - right).max(0.0))
        .filter(|value| value.is_finite())
}

fn missing_representative_model_probe(
    recommendation: &ModelRecommendation,
    model_specific_probe_errors: &[String],
) -> bool {
    let warned_about_representative_probe = recommendation.warnings.iter().any(|warning| {
        warning.contains("shape-representative decode kernel probe")
            || warning.contains("tok/s confidence cannot be high")
    });
    let model_probe_failed = model_specific_probe_errors.iter().any(|error| {
        error.contains("dense graph probe did not produce a supported result")
            || error.contains("linear attention graph probe did not produce a supported result")
            || error.contains("MoE graph probe did not produce a supported result")
            || error.contains("output projection probe did not produce a supported result")
    });
    warned_about_representative_probe && model_probe_failed
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0.0 => Some(numerator / denominator),
        _ => None,
    }
}

fn throughput_ratio_verdict(ratio: Option<f64>) -> String {
    let Some(ratio) = ratio else {
        return "missing".into();
    };
    if (ratio - 1.0).abs() <= DEFAULT_TOLERANCE {
        "match".into()
    } else if ratio < 1.0 {
        "slower-than-reference".into()
    } else {
        "faster-than-reference".into()
    }
}

fn decode_probe_classification(input: DecodeProbeClassificationInput) -> String {
    if input.predicted.is_none() {
        return "missing_fit_estimate".into();
    }
    if input.observed.is_none() {
        return "missing_observed_benchmark".into();
    }
    if input.abi.is_none() {
        return "missing_abi_probe".into();
    }

    let fit_observed_matches = ratio_matches(input.observed_over_fit);
    let fit_abi_matches = ratio_matches(input.abi_over_fit);
    let scenario_observed_matches = ratio_matches(input.observed_over_scenario_fit);
    let scenario_abi_matches = ratio_matches(input.abi_over_scenario_fit);
    let abi_observed_matches = ratio_matches(input.observed_over_abi);
    let context_rescore_less_representative =
        context_rescore_is_less_representative_than_primary(&input);
    if context_rescore_less_representative
        && fit_observed_matches
        && fit_abi_matches
        && abi_observed_matches
    {
        // The validation scenario has a shorter prompt/context shape than the
        // primary workload target, so the scenario rescore can be less
        // representative than the main recommendation. When the primary
        // estimate, ABI token loop, and observed request median all agree, keep
        // that positive evidence visible instead of collapsing the row into a
        // generic "context rescore unstable" bucket. The row still stays out of
        // accuracy gates because either the ABI or request path was noisy enough
        // to make the scenario comparison unclean.
        return "primary_estimate_and_probe_agree_context_rescore_unstable".into();
    }
    if input.request_decode_noisy && fit_observed_matches && fit_abi_matches && abi_observed_matches
    {
        // The point estimate agrees, but the individual requests are swinging
        // too much to count this as clean +/-10% evidence. This happens on tiny
        // models where the full serving path can move by more than the model
        // core cost. Keep the row useful as a positive diagnostic, but do not
        // let it inflate the accuracy-gated sample set.
        return "estimate_and_probe_agree_noisy_requests".into();
    }
    if input.request_decode_noisy
        && !fit_observed_matches
        && scenario_observed_matches
        && scenario_abi_matches
        && abi_observed_matches
    {
        // Primary fit and benchmark-scenario fit intentionally answer
        // different questions when the workload context differs from the
        // benchmark prompt shape. If the scenario-scoped estimate, ABI probe,
        // and observed steady median all agree, keep that positive evidence
        // visible while still excluding the row from +/-10% accuracy gates due
        // request spread. This is validation reporting only; observed tok/s is
        // never fed back into model-fit scoring.
        return "scenario_estimate_and_probe_agree_noisy_requests".into();
    }
    if !fit_observed_matches
        && scenario_observed_matches
        && scenario_abi_matches
        && abi_observed_matches
    {
        // The primary fit estimate and the benchmark-scenario estimate can be
        // intentionally different. The primary recommendation answers the
        // configured workload question, usually with the preferred context
        // window. The scenario rescore answers the exact validation prompt
        // shape. When the scenario estimate, ABI probe, and observed request
        // path agree, the benchmark validates the metadata/probe model for that
        // prompt shape; it does not falsify the primary estimate at a different
        // context. Keep this out of metadata_estimate_miss so the report does
        // not imply that a source-grounded context charge is wrong just because
        // the validation prompt was shorter.
        return "primary_context_differs_from_benchmark".into();
    }
    let decode_submission_residual = decode_submission_residual_exceeds_tolerance(
        input.decode_submission_residual_share_of_predicted,
    ) || selected_full_token_handoff_covers_abi_sampling(&input);
    let sampler_sync_residual =
        sampler_sync_residual_exceeds_tolerance(input.sampler_sync_residual_share_of_predicted);
    let source_boundary_residual_is_representative = !fit_observed_matches
        && ratio_is_slower(input.abi_over_fit)
        && (abi_observed_matches || input.request_decode_noisy);
    if decode_submission_residual
        && sampler_sync_residual
        && source_boundary_residual_is_representative
    {
        return if input.request_decode_noisy {
            "decode_and_sampler_residual_noisy".into()
        } else {
            "decode_and_sampler_residual".into()
        };
    }
    if decode_submission_residual && source_boundary_residual_is_representative {
        return if input.request_decode_noisy {
            "decode_submission_residual_noisy".into()
        } else {
            "decode_submission_residual".into()
        };
    }
    if sampler_sync_residual && source_boundary_residual_is_representative {
        if input.selected_full_token_source_sampled_probe {
            return if input.request_decode_noisy {
                "llama_source_boundary_residual_noisy".into()
            } else {
                "llama_source_boundary_residual".into()
            };
        }
        return if input.request_decode_noisy {
            "sampler_sync_residual_noisy".into()
        } else {
            "sampler_sync_residual".into()
        };
    }
    if context_rescore_less_representative && (input.abi_probe_noisy || input.request_decode_noisy)
    {
        // Context-aligned rescoring is a validation diagnostic, not the primary
        // recommendation. Keep it from masking source-boundary residuals above:
        // when the ABI token loop and observed steady decode are both slower
        // than the metadata estimate, the useful miss class is the source path
        // that the synthetic probes did not cover. Fall back to this generic
        // bucket only when no source-grounded residual explains the mismatch.
        return "context_rescore_unstable".into();
    }
    if input.abi_probe_noisy {
        return "abi_probe_noisy".into();
    }
    if input.request_decode_noisy && !fit_observed_matches {
        if request_noise_cannot_explain_observed_abi_gap(
            input.observed_over_abi,
            input.request_spread_pct,
        ) {
            // This is still not clean metadata-estimator evidence: the steady
            // benchmark path is noisy. But when the median steady result is
            // much slower than the ABI decode probe by more than the observed
            // request spread plus the normal +/-10% tolerance, calling the row
            // "noise" hides the important miss class. The source-grounded
            // interpretation is that the full request/serving path contains
            // overhead not represented by the model-shaped decode graph probe.
            return "steady_path_overhead_mismatch_noisy".into();
        }
        // Repeat medians can look stable after remeasurement while the
        // individual steady-decode requests inside those repeats still swing
        // widely. That is especially visible on tiny models where fixed server
        // scheduling, sampler/session work, and backend runtime state are a
        // large fraction of the request. Treat those rows as validation-path
        // evidence rather than clean metadata-estimator accuracy evidence.
        // This uses the same spread threshold as the rest of the validator; it
        // is not a model-specific correction and it is never fed back into
        // model-fit scoring.
        return "validation_request_noise".into();
    }
    match (fit_observed_matches, fit_abi_matches, abi_observed_matches) {
        (true, true, true) => "estimate_and_probe_agree".into(),
        (false, _, true) if input.missing_representative_model_probe => {
            "missing_representative_decode_probe".into()
        }
        (false, _, true) => "metadata_estimate_miss".into(),
        (false, true, false) => {
            if ratio_is_slower(input.observed_over_fit) && ratio_is_slower(input.observed_over_abi)
            {
                "steady_path_overhead_mismatch".into()
            } else {
                "runtime_path_mismatch".into()
            }
        }
        (false, false, false) => "mixed_estimate_and_runtime_mismatch".into(),
        (true, false, false) => "probe_not_representative".into(),
        (true, false, true) => "probe_differs_but_observed_matches_fit".into(),
        (true, true, false) => "unstable_probe_geometry".into(),
    }
}

fn selected_full_token_handoff_covers_abi_sampling(input: &DecodeProbeClassificationInput) -> bool {
    input.selected_full_token_handoff_probe && ratio_matches(input.abi_sampling_over_selected_fit)
}

fn context_rescore_is_less_representative_than_primary(
    input: &DecodeProbeClassificationInput,
) -> bool {
    let Some(primary_ratio) = input.observed_over_fit else {
        return false;
    };
    let Some(scenario_ratio) = input.observed_over_scenario_fit else {
        return false;
    };
    if !primary_ratio.is_finite() || !scenario_ratio.is_finite() {
        return false;
    }
    let primary_error = (primary_ratio - 1.0).abs();
    let scenario_error = (scenario_ratio - 1.0).abs();
    scenario_error > primary_error + DEFAULT_TOLERANCE
}

fn ratio_matches(ratio: Option<f64>) -> bool {
    ratio.is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
}

fn ratio_is_slower(ratio: Option<f64>) -> bool {
    ratio.is_some_and(|ratio| ratio < 1.0 - DEFAULT_TOLERANCE)
}

fn request_noise_cannot_explain_observed_abi_gap(
    observed_over_abi: Option<f64>,
    request_spread_pct: Option<f64>,
) -> bool {
    let Some(ratio) = observed_over_abi else {
        return false;
    };
    if !ratio.is_finite() || ratio >= 1.0 - DEFAULT_TOLERANCE {
        return false;
    }
    let gap_pct = (1.0 - ratio).max(0.0) * 100.0;
    let spread_pct = request_spread_pct.unwrap_or_default().max(0.0);
    gap_pct > spread_pct + DEFAULT_TOLERANCE * 100.0
}

fn sampler_sync_residual_exceeds_tolerance(residual_share: Option<f64>) -> bool {
    residual_share.is_some_and(|share| share.is_finite() && share > DEFAULT_TOLERANCE)
}

fn decode_submission_residual_exceeds_tolerance(residual_share: Option<f64>) -> bool {
    residual_share.is_some_and(|share| share.is_finite() && share > DEFAULT_TOLERANCE)
}

fn decode_probe_notes(input: DecodeProbeNotesInput<'_>) -> Vec<String> {
    let mut notes = Vec::new();
    notes.push(
        "ABI probe is validation evidence only; it is not fed into metadata-only fit scoring."
            .into(),
    );
    if let Some(ratio) = input.observed_over_fit {
        notes.push(format!(
            "Observed steady decode is {:.1}% of the metadata estimate.",
            ratio * 100.0
        ));
    }
    if let Some(ratio) = input.abi_over_fit {
        notes.push(format!(
            "ABI decode probe is {:.1}% of the metadata estimate.",
            ratio * 100.0
        ));
    }
    if let Some(ratio) = input.observed_over_abi {
        notes.push(format!(
            "Observed steady decode is {:.1}% of the ABI decode probe.",
            ratio * 100.0
        ));
    }
    if let Some(spread) = input.request_spread_pct {
        notes.push(format!(
            "Per-request steady decode spread was {spread:.1}% across benchmark requests."
        ));
    }
    if let Some(spread) = input.selected_fit_probe_max_spread_pct {
        notes.push(format!(
            "Largest selected model-fit decode probe spread was {spread:.1}% across timed probe runs."
        ));
    }
    if input.selected_full_token_source_sampled_probe {
        notes.push(
            "The selected model-fit probe is a source-sampled synthetic full-token graph: it times source-side decode bookkeeping, graph submission, async logits extraction, scheduler synchronization, and a CPU logits scan, but it still does not load a real llama_context or benchmark the target model."
                .into(),
        );
    }
    if let Some(residual) = input.decode_submission_residual_ms_per_token {
        notes.push(format!(
            "ABI decode-call time exceeds predicted synthetic decode submission by {residual:.3} ms/token."
        ));
    }
    if let Some(decode_call_ms) = input.abi_decode_call_ms_per_token {
        notes.push(format!(
            "ABI decode-call time is {decode_call_ms:.3} ms/token before logits/sampling."
        ));
    }
    if let Some(residual) = input.sampler_sync_residual_ms_per_token {
        notes.push(format!(
            "ABI sampling/logits-sync time exceeds predicted sampler+logits handoff by {residual:.3} ms/token."
        ));
    }
    if let Some(ratio) = input.abi_sampling_over_selected_fit {
        notes.push(format!(
            "ABI sampling/logits-sync time is {:.1}% of the selected metadata decode estimate.",
            ratio * 100.0
        ));
    }
    match (
        input.abi_logits_ready_ms_per_token,
        input.abi_logits_scan_ms_per_token,
    ) {
        (Some(ready_ms), Some(scan_ms)) => notes.push(format!(
            "ABI splits that sampling/logits-sync boundary into {ready_ms:.3} ms/token waiting for llama_get_logits_ith() readiness and {scan_ms:.3} ms/token scanning the CPU-visible vocab row."
        )),
        (Some(ready_ms), None) => notes.push(format!(
            "ABI reports {ready_ms:.3} ms/token waiting for llama_get_logits_ith() readiness; vocab scan timing was unavailable."
        )),
        (None, Some(scan_ms)) => notes.push(format!(
            "ABI reports {scan_ms:.3} ms/token scanning the CPU-visible vocab row; logits readiness timing was unavailable."
        )),
        (None, None) => {}
    }
    match input.classification {
        "metadata_estimate_miss" => notes.push(
            "ABI and observed decode agree, so the miss points at metadata cost modeling.".into(),
        ),
        "missing_representative_decode_probe" => notes.push(
            "ABI and observed decode agree, but model-shaped decode probes were unavailable for this row, so the metadata estimate fell back to less representative timing evidence."
                .into(),
        ),
        "validation_request_noise" => notes.push(
            "The benchmark's per-request steady decode spread exceeded the validator noise threshold, so this row is not clean evidence that the metadata estimator is wrong."
                .into(),
        ),
        "estimate_and_probe_agree_noisy_requests" => notes.push(
            "The metadata estimate, ABI probe, and steady median agree, but per-request spread exceeded the validator noise threshold, so this row is positive diagnostic evidence rather than clean accuracy-gated evidence."
                .into(),
        ),
        "primary_estimate_and_probe_agree_context_rescore_unstable" => notes.push(
            "The primary workload-context estimate, ABI probe, and steady median agree, but the benchmark-context rescore is less representative on a noisy row; keep this as positive primary-scope evidence, not clean scenario-scope accuracy evidence."
                .into(),
        ),
        "scenario_estimate_and_probe_agree_noisy_requests" => notes.push(
            "The benchmark-scenario estimate, ABI probe, and steady median agree, but per-request spread exceeded the validator noise threshold; this is positive scenario-scope evidence, not clean primary metadata-estimator accuracy evidence."
                .into(),
        ),
        "primary_context_differs_from_benchmark" => notes.push(
            "The benchmark-scenario estimate, ABI probe, and steady median agree, while the primary workload estimate uses a different context target. This validates the scenario rescore without treating the primary context charge as a metadata miss."
                .into(),
        ),
        "context_rescore_unstable" => notes.push(
            "The benchmark-context rescore is less representative than the primary workload-context estimate on a noisy row; keep this as scenario-scope evidence, not clean metadata-estimator accuracy evidence."
                .into(),
        ),
        "steady_path_overhead_mismatch_noisy" => notes.push(
            "Per-request spread is high, but it is too small to explain the observed-vs-ABI gap; treat this as noisy evidence of request/serving-path overhead rather than clean metadata-estimator error."
                .into(),
        ),
        "sampler_sync_residual" => notes.push(
            "ABI decode shows the missing token time is in source-visible logits synchronization and sampler work after the decode graph is submitted; this is validation evidence for a benchmark/probe gap, not a fitted model correction."
                .into(),
        ),
        "sampler_sync_residual_noisy" => notes.push(
            "The steady requests are noisy, but ABI decode shows a large source-visible logits synchronization/sampler residual; keep the row out of clean accuracy gates while tracking this miss class explicitly."
                .into(),
        ),
        "llama_source_boundary_residual" => notes.push(
            "ABI and observed decode agree, but the source-sampled synthetic full-token graph is still faster than the real llama.cpp boundary where skippy_decode_tokens()/llama_decode() returns and skippy_greedy_sample() calls llama_get_logits_ith(). This is a source-representation gap in the probe, not an observed-throughput correction."
                .into(),
        ),
        "llama_source_boundary_residual_noisy" => notes.push(
            "The steady requests are noisy, but ABI and observed decode agree closely enough to expose a source-representation gap: the source-sampled synthetic full-token graph is faster than the real llama.cpp decode/logits-readiness boundary. Keep this row out of clean accuracy gates and use it to guide better source-shaped probes."
                .into(),
        ),
        "decode_submission_residual" => notes.push(
            "The selected full-token handoff probe already accounts for ABI sampling/logits visibility, while ABI total decode is still slower than the metadata estimate. The remaining gap is before sampling, in the source-visible decode submission path such as skippy_decode_tokens()/llama_decode(), not in a fitted scorer correction."
                .into(),
        ),
        "decode_submission_residual_noisy" => notes.push(
            "The steady requests are noisy, but the selected full-token handoff probe already accounts for ABI sampling/logits visibility while ABI total decode is still slower than the metadata estimate. Track this as decode-submission residual evidence and keep it out of clean accuracy gates."
                .into(),
        ),
        "decode_and_sampler_residual" => notes.push(
            "ABI timing shows missing token time on both source-visible sides of the boundary: the decode-call path before logits/sampling and the logits synchronization/sampler path after graph submission. This is validation evidence for probe coverage, not an observed-throughput correction."
                .into(),
        ),
        "decode_and_sampler_residual_noisy" => notes.push(
            "The steady requests are noisy, but ABI timing shows missing token time on both source-visible sides of the boundary: decode-call submission and logits synchronization/sampler work. Keep this row out of clean accuracy gates while tracking the residual explicitly."
                .into(),
        ),
        "abi_probe_noisy" => notes.push(
            "The ABI decode probe exceeded the validator spread threshold, so this row is reported as probe-noise evidence rather than clean estimator accuracy evidence."
                .into(),
        ),
        "steady_path_overhead_mismatch" => notes.push(
            "ABI and metadata agree, but steady decode is slower; this points at validation/runtime path overhead such as server scheduling, sampler/session work, or benchmark scenario overhead rather than the metadata-only estimator."
                .into(),
        ),
        "runtime_path_mismatch" | "mixed_estimate_and_runtime_mismatch" => notes.push(
            "Observed decode diverges from the ABI probe, so runtime path, graph, cache, or benchmark shape needs inspection.".into(),
        ),
        "probe_not_representative" => notes.push(
            "The metadata estimate matched observed decode, but the ABI probe did not represent full runtime throughput.".into(),
        ),
        _ => {}
    }
    notes
}

fn graph_inventory_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    model_specific_probes: &[DecodeKernelProbe],
    abi_probe: Option<&AbiDecodeProbeSummary>,
    abi_probe_kind: AbiDecodeProbeKind,
) -> Option<GraphInventoryDiagnostic> {
    let abi_probe = abi_probe?;
    let comparisons = graph_inventory_comparisons(profile, abi_probe);
    let metadata_transformer_weight_bytes = graph_metadata_transformer_bytes(profile);
    let graph_transformer_weight_src0_bytes = graph_transformer_src0_bytes(profile, abi_probe);
    let graph_unclassified_matmul_src0_bytes = graph_family_src0_bytes(abi_probe, "matmul");
    let graph_transformer_plus_unclassified_src0_bytes =
        graph_transformer_weight_src0_bytes.saturating_add(graph_unclassified_matmul_src0_bytes);
    let metadata_transformer_matmul_nodes = graph_metadata_transformer_nodes(profile);
    let graph_transformer_matmul_nodes = graph_transformer_node_count(profile, abi_probe);
    let transformer_group = selected_graph_probe_group(profile, recommendation);
    let selected_transformer_probe = transformer_group.and_then(|group| group.probe_name.clone());
    let selected_transformer_probe_layers = selected_transformer_probe
        .as_deref()
        .map(graph_probe_layers_from_name);
    let selected_probe_context_tokens = selected_transformer_probe
        .as_deref()
        .and_then(graph_probe_context_from_name);
    let abi_graph_context_tokens = abi_graph_context_tokens(abi_probe);
    let selected_probe_node_count = selected_transformer_probe.as_deref().and_then(|name| {
        model_specific_probes
            .iter()
            .find(|probe| probe.name == name)
            .and_then(|probe| probe.graph_node_count)
    });
    let selected_probe_inventory_delta =
        if graph_contexts_match(selected_probe_context_tokens, abi_graph_context_tokens) {
            selected_transformer_probe.as_deref().and_then(|name| {
                model_specific_probes
                    .iter()
                    .find(|probe| probe.name == name)
                    .and_then(|probe| selected_probe_inventory_delta(probe, abi_probe))
            })
        } else {
            None
        };
    let selected_probe_nodes_over_abi = ratio(
        selected_probe_node_count.map(|value| value as f64),
        abi_probe.graph_node_count.map(|value| value as f64),
    );
    let estimated_transformer_block_ms =
        transformer_group.map(|group| f64::from(group.bandwidth_ms));
    let abi_ms_per_token = match (abi_probe.measured_tokens, abi_probe.elapsed_ms) {
        (Some(tokens), Some(elapsed_ms)) if tokens > 0 => Some(elapsed_ms / tokens as f64),
        _ => abi_probe
            .tokens_per_second
            .filter(|tps| *tps > 0.0)
            .map(|tps| 1000.0 / tps),
    };
    let estimated_transformer_over_abi = ratio(estimated_transformer_block_ms, abi_ms_per_token);
    let graph_transformer_src0_over_metadata = ratio_u64(
        graph_transformer_weight_src0_bytes,
        metadata_transformer_weight_bytes,
    );
    let graph_transformer_plus_unclassified_src0_over_metadata = ratio_u64(
        graph_transformer_plus_unclassified_src0_bytes,
        metadata_transformer_weight_bytes,
    );
    let mut notes = graph_inventory_notes(GraphInventoryNotesInput {
        metadata_transformer_weight_bytes,
        graph_transformer_weight_src0_bytes,
        graph_unclassified_matmul_src0_bytes,
        metadata_transformer_matmul_nodes,
        graph_transformer_matmul_nodes,
        selected_probe_layers: selected_transformer_probe_layers,
        estimated_transformer_over_abi,
        selected_probe_nodes_over_abi,
    });
    if abi_probe.graph_inventory_bucket_overflow_count.unwrap_or(0) > 0 {
        notes.push(
            "Graph inventory bucket overflowed; comparisons are partial and should not drive estimator changes."
                .into(),
        );
    }
    if abi_probe_kind == AbiDecodeProbeKind::ContextAligned {
        notes.push(
            "Graph inventory uses a diagnostic ABI probe warmed to the selected synthetic active-context depth; this probe is not used by the fitter or the observed-vs-ABI throughput summary."
                .into(),
        );
    }
    match (selected_probe_context_tokens, abi_graph_context_tokens) {
        (Some(selected_context), Some(abi_context))
            if !graph_contexts_match(selected_probe_context_tokens, abi_graph_context_tokens) =>
        {
            notes.push(format!(
                "Synthetic/ABI bucket deltas are suppressed because the selected synthetic probe used n_kv/context {selected_context} while the ABI graph inventory used n_kv/context {abi_context}."
            ));
        }
        _ => {}
    }
    match selected_probe_inventory_delta {
        Some(delta) if delta.bucket_mismatch_count > 0 => {
            notes.push(format!(
                "Selected synthetic probe differs from ABI graph inventory in {} buckets; this points at probe topology/layout representation rather than GGUF tensor grouping.",
                delta.bucket_mismatch_count
            ));
        }
        _ => {}
    }
    let available = !abi_probe.graph_inventory.is_empty();
    let status = graph_inventory_status(
        &comparisons,
        selected_transformer_probe_layers,
        graph_unclassified_matmul_src0_bytes,
        graph_transformer_plus_unclassified_src0_over_metadata,
    );
    Some(GraphInventoryDiagnostic {
        available,
        status,
        graph_node_count: abi_probe.graph_node_count,
        graph_inventory_bucket_overflow_count: abi_probe.graph_inventory_bucket_overflow_count,
        selected_transformer_probe,
        selected_transformer_probe_layers,
        selected_probe_context_tokens,
        abi_graph_context_tokens,
        selected_probe_node_count,
        selected_probe_nodes_over_abi,
        selected_probe_inventory_bucket_mismatch_count: selected_probe_inventory_delta
            .map(|delta| delta.bucket_mismatch_count),
        selected_probe_inventory_abs_node_delta: selected_probe_inventory_delta
            .map(|delta| delta.abs_node_delta),
        selected_probe_inventory_abs_src0_delta_bytes: selected_probe_inventory_delta
            .map(|delta| delta.abs_src0_delta_bytes),
        selected_probe_inventory_abs_src1_delta_bytes: selected_probe_inventory_delta
            .map(|delta| delta.abs_src1_delta_bytes),
        selected_probe_inventory_abs_output_delta_bytes: selected_probe_inventory_delta
            .map(|delta| delta.abs_output_delta_bytes),
        metadata_transformer_matmul_nodes,
        graph_transformer_matmul_nodes,
        metadata_transformer_weight_bytes,
        graph_transformer_weight_src0_bytes,
        graph_unclassified_matmul_src0_bytes,
        graph_transformer_src0_over_metadata,
        graph_transformer_plus_unclassified_src0_over_metadata,
        estimated_transformer_block_ms,
        abi_ms_per_token,
        estimated_transformer_over_abi,
        comparisons,
        notes,
    })
}

fn selected_graph_probe_group<'a>(
    profile: &ModelProfile,
    recommendation: &'a ModelRecommendation,
) -> Option<&'a model_fit::DecodeCostGroupBreakdown> {
    let transformer_cost_group = graph_transformer_cost_group(profile);
    recommendation
        .decode_cost_breakdown
        .as_ref()
        .and_then(|breakdown| {
            breakdown
                .groups
                .iter()
                .find(|group| group.group == "full_token_graph")
                .or_else(|| {
                    breakdown
                        .groups
                        .iter()
                        .find(|group| group.group == transformer_cost_group)
                })
        })
}

fn selected_graph_probe_context_tokens(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
) -> Option<u32> {
    selected_graph_probe_group(profile, recommendation)
        .and_then(|group| group.probe_name.as_deref())
        .and_then(graph_probe_context_from_name)
}

fn graph_inventory_comparisons(
    profile: &ModelProfile,
    abi_probe: &AbiDecodeProbeSummary,
) -> Vec<GraphInventoryComparison> {
    let mut comparisons = vec![graph_inventory_comparison(
        "attention_matmul",
        profile.tensor_matmul.attention.bytes,
        profile.tensor_matmul.attention.shape.logical_matrix_count,
        abi_probe,
        "attention_matmul",
    )];

    if profile.architecture_class == model_fit::ModelArchitectureClass::SparseMoeTransformer {
        // llama.cpp does not lower sparse MoE expert FFN as the same graph
        // family as dense FFN. The routed expert path uses
        // GGML_OP_MUL_MAT_ID: one graph node per up/gate/down expert matrix
        // group for each layer, with the expert dimension packed behind the
        // tensor and selected by ids at runtime. Comparing GGUF
        // `expert_feed_forward` bytes against the dense `ffn_matmul` family
        // made the validator report a false inventory mismatch even when the
        // graph had exactly the routed expert bytes we expected.
        comparisons.push(graph_inventory_comparison(
            "expert_moe_matmul_id",
            profile.tensor_matmul.expert_feed_forward.bytes,
            sparse_moe_expected_expert_matmul_id_nodes(profile),
            abi_probe,
            "moe_matmul_id",
        ));
    } else {
        comparisons.push(graph_inventory_comparison(
            "ffn_matmul",
            profile.tensor_matmul.feed_forward.bytes,
            profile
                .tensor_matmul
                .feed_forward
                .shape
                .logical_matrix_count,
            abi_probe,
            "ffn_matmul",
        ));
    }

    comparisons.push(graph_inventory_comparison(
        "output_matmul",
        graph_expected_output_bytes(profile),
        graph_expected_output_nodes(profile),
        abi_probe,
        "output_matmul",
    ));

    comparisons
}

fn selected_probe_inventory_delta(
    probe: &DecodeKernelProbe,
    abi_probe: &AbiDecodeProbeSummary,
) -> Option<SelectedProbeInventoryDelta> {
    if probe.graph_inventory.is_empty() || abi_probe.graph_inventory.is_empty() {
        return None;
    }
    let synthetic = synthetic_graph_inventory_totals(&probe.graph_inventory);
    let abi = abi_graph_inventory_totals(&abi_probe.graph_inventory);
    let keys = synthetic
        .keys()
        .chain(abi.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut delta = SelectedProbeInventoryDelta::default();
    for key in keys {
        let left = synthetic.get(&key).copied().unwrap_or_default();
        let right = abi.get(&key).copied().unwrap_or_default();
        let node_delta = abs_delta_u64(left.node_count, right.node_count);
        let src0_delta = abs_delta_u64(left.src0_bytes, right.src0_bytes);
        let src1_delta = abs_delta_u64(left.src1_bytes, right.src1_bytes);
        let output_delta = abs_delta_u64(left.output_bytes, right.output_bytes);
        if node_delta > 0 || src0_delta > 0 || src1_delta > 0 || output_delta > 0 {
            delta.bucket_mismatch_count += 1;
            delta.abs_node_delta = delta.abs_node_delta.saturating_add(node_delta);
            delta.abs_src0_delta_bytes = delta.abs_src0_delta_bytes.saturating_add(src0_delta);
            delta.abs_src1_delta_bytes = delta.abs_src1_delta_bytes.saturating_add(src1_delta);
            delta.abs_output_delta_bytes =
                delta.abs_output_delta_bytes.saturating_add(output_delta);
        }
    }
    Some(delta)
}

fn graph_contexts_match(selected_context: Option<u32>, abi_context: Option<u32>) -> bool {
    match (selected_context, abi_context) {
        (Some(selected), Some(abi)) => ratio(Some(f64::from(selected)), Some(f64::from(abi)))
            .is_some_and(|value| (value - 1.0).abs() <= DEFAULT_TOLERANCE),
        _ => true,
    }
}

fn abi_graph_context_tokens(abi_probe: &AbiDecodeProbeSummary) -> Option<u32> {
    abi_graph_context_tokens_for_family(abi_probe, "kv_cache")
        .or_else(|| abi_graph_context_tokens_for_family(abi_probe, "attention_runtime"))
}

fn abi_graph_context_tokens_for_family(
    abi_probe: &AbiDecodeProbeSummary,
    family: &str,
) -> Option<u32> {
    let mut dimensions = abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .flat_map(|bucket| bucket.ne.iter().copied())
        .filter_map(|value| u32::try_from(value).ok())
        .filter(|value| *value > 1)
        .collect::<Vec<_>>();
    dimensions.sort_unstable();
    dimensions.dedup();
    if family == "kv_cache" && dimensions.len() >= 2 {
        // The KV-cache inventory contains both allocation-capacity tensors
        // (`kv_width x ctx_size`) and active attention views
        // (`head_dim x n_kv x kv_heads`). For context comparability with
        // synthetic `_nkv` probes, the active view is the useful value. In the
        // current llama.cpp graph it is the largest KV dimension below the
        // capacity dimension.
        return dimensions.iter().rev().nth(1).copied();
    }
    dimensions.last().copied()
}

fn synthetic_graph_inventory_totals(
    buckets: &[mesh_llm_gpu_bench::DecodeGraphInventoryBucket],
) -> BTreeMap<String, GraphInventoryTotals> {
    buckets
        .iter()
        .map(|bucket| {
            (
                graph_inventory_bucket_key(
                    bucket.family.as_deref(),
                    bucket.ggml_op,
                    bucket.ggml_type,
                    &bucket.ne,
                ),
                GraphInventoryTotals {
                    node_count: bucket.node_count.unwrap_or_default(),
                    src0_bytes: bucket.src0_bytes.unwrap_or_default(),
                    src1_bytes: bucket.src1_bytes.unwrap_or_default(),
                    output_bytes: bucket.output_bytes.unwrap_or_default(),
                },
            )
        })
        .collect()
}

fn abi_graph_inventory_totals(
    buckets: &[AbiGraphInventoryBucket],
) -> BTreeMap<String, GraphInventoryTotals> {
    buckets
        .iter()
        .map(|bucket| {
            (
                graph_inventory_bucket_key(
                    bucket.family.as_deref(),
                    bucket.ggml_op,
                    bucket.ggml_type,
                    &bucket.ne,
                ),
                GraphInventoryTotals {
                    node_count: bucket.node_count.unwrap_or_default(),
                    src0_bytes: bucket.src0_bytes.unwrap_or_default(),
                    src1_bytes: bucket.src1_bytes.unwrap_or_default(),
                    output_bytes: bucket.output_bytes.unwrap_or_default(),
                },
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
struct GraphInventoryTotals {
    node_count: u64,
    src0_bytes: u64,
    src1_bytes: u64,
    output_bytes: u64,
}

fn graph_inventory_bucket_key(
    family: Option<&str>,
    ggml_op: Option<i64>,
    ggml_type: Option<u64>,
    ne: &[i64],
) -> String {
    format!(
        "{}|{}|{}|{}",
        family.unwrap_or(""),
        ggml_op
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        ggml_type
            .map(|value| value.to_string())
            .unwrap_or_else(|| "-".into()),
        ne.iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("x")
    )
}

fn abs_delta_u64(left: u64, right: u64) -> u64 {
    left.max(right) - left.min(right)
}

fn graph_transformer_cost_group(profile: &ModelProfile) -> &'static str {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => "sparse_transformer_block",
        _ => "transformer_block",
    }
}

fn sparse_moe_expected_expert_matmul_id_nodes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.expert_feed_forward.bytes == 0 {
        return 0;
    }

    profile
        .layer_count
        .filter(|layers| *layers > 0)
        .map(|layers| u64::from(layers).saturating_mul(3))
        .unwrap_or_else(|| {
            profile
                .tensor_matmul
                .expert_feed_forward
                .shape
                .logical_matrix_count
        })
}

fn graph_inventory_comparison(
    name: &'static str,
    metadata_weight_bytes: u64,
    metadata_node_count: u64,
    abi_probe: &AbiDecodeProbeSummary,
    family: &str,
) -> GraphInventoryComparison {
    let graph_weight_src0_bytes = graph_family_src0_bytes(abi_probe, family);
    let graph_node_count = graph_family_node_count(abi_probe, family);
    GraphInventoryComparison {
        name,
        metadata_weight_bytes,
        metadata_node_count,
        graph_weight_src0_bytes,
        graph_node_count,
        src0_over_metadata: ratio_u64(graph_weight_src0_bytes, metadata_weight_bytes),
        node_count_delta: i64::try_from(graph_node_count).unwrap_or(i64::MAX)
            - i64::try_from(metadata_node_count).unwrap_or(i64::MAX),
    }
}

fn graph_expected_output_bytes(profile: &ModelProfile) -> u64 {
    if profile.tensor_matmul.output.bytes > 0 || profile.tensor_group_bytes.output_bytes > 0 {
        return profile
            .tensor_matmul
            .output
            .bytes
            .max(profile.tensor_group_bytes.output_bytes);
    }
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::DenseTransformer
        | model_fit::ModelArchitectureClass::SparseMoeTransformer
        | model_fit::ModelArchitectureClass::Unknown => profile.tensor_group_bytes.embedding_bytes,
        _ => 0,
    }
}

fn graph_expected_output_nodes(profile: &ModelProfile) -> u64 {
    if graph_expected_output_bytes(profile) == 0 {
        0
    } else {
        profile
            .tensor_matmul
            .output
            .shape
            .logical_matrix_count
            .max(1)
    }
}

fn graph_family_src0_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.src0_bytes)
        .sum()
}

fn graph_family_node_count(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.node_count)
        .sum()
}

fn graph_inventory_status(
    comparisons: &[GraphInventoryComparison],
    selected_probe_layers: Option<u32>,
    graph_unclassified_matmul_src0_bytes: u64,
    graph_transformer_plus_unclassified_src0_over_metadata: Option<f64>,
) -> String {
    let inventory_mismatch = comparisons.iter().any(|comparison| {
        comparison
            .src0_over_metadata
            .is_some_and(|ratio| (ratio - 1.0).abs() > DEFAULT_TOLERANCE)
            || comparison.node_count_delta != 0
    });
    if inventory_mismatch
        && graph_unclassified_matmul_src0_bytes > 0
        && graph_transformer_plus_unclassified_src0_over_metadata
            .is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
    {
        "metadata_inventory_has_unclassified_matmul".into()
    } else if inventory_mismatch
        && graph_transformer_plus_unclassified_src0_over_metadata
            .is_some_and(|ratio| ratio < 1.0 - DEFAULT_TOLERANCE)
    {
        "metadata_inventory_missing_transformer_matmul".into()
    } else if inventory_mismatch {
        "metadata_inventory_mismatch".into()
    } else if selected_probe_layers == Some(1) {
        "metadata_inventory_matches_probe_depth_risk".into()
    } else {
        "metadata_inventory_matches".into()
    }
}

fn graph_inventory_notes(input: GraphInventoryNotesInput) -> Vec<String> {
    let mut notes = Vec::new();
    if ratio_u64(
        input.graph_transformer_weight_src0_bytes,
        input.metadata_transformer_weight_bytes,
    )
    .is_some_and(|ratio| (ratio - 1.0).abs() <= DEFAULT_TOLERANCE)
        && input.metadata_transformer_matmul_nodes == input.graph_transformer_matmul_nodes
    {
        notes.push(
            "GGUF tensor inventory matches the llama.cpp transformer matmul graph; the miss is likely timing/probe representation, not tensor grouping."
                .into(),
        );
    }
    if input.selected_probe_layers == Some(1) && input.graph_transformer_matmul_nodes > 0 {
        notes.push(
            "The selected transformer timing comes from a one-layer synthetic graph while the native decode graph contains the full repeated-layer matmul inventory."
                .into(),
        );
    }
    if input
        .selected_probe_nodes_over_abi
        .is_some_and(|ratio| ratio < 1.0 - DEFAULT_TOLERANCE)
    {
        notes.push(
            "The selected synthetic decode probe has fewer GGML graph nodes than the ABI llama.cpp decode graph; treat residual tok/s miss as possible probe-topology under-representation before changing bandwidth scoring."
                .into(),
        );
    }
    if input.graph_unclassified_matmul_src0_bytes > 0 {
        notes.push(format!(
            "Native graph has {:.1} MiB of unclassified GGML_OP_MUL_MAT src0 bytes; inspect source node names before treating a known-family mismatch as missing model work.",
            input.graph_unclassified_matmul_src0_bytes as f64 / 1024.0 / 1024.0
        ));
    }
    if let Some(ratio) = input.estimated_transformer_over_abi {
        notes.push(format!(
            "Estimated transformer-block time is {:.1}% of total ABI ms/token.",
            ratio * 100.0
        ));
    }
    notes
}

fn operation_bucket_diagnostic(
    profile: &ModelProfile,
    recommendation: &ModelRecommendation,
    abi_probe: Option<&AbiDecodeProbeSummary>,
) -> Option<OperationBucketDiagnostic> {
    let abi_probe = abi_probe?;
    let breakdown = recommendation.decode_cost_breakdown.as_ref();
    let selected_ms = breakdown.map(|breakdown| f64::from(breakdown.selected_time_ms));
    let abi_ms = abi_ms_per_token(abi_probe);
    let buckets = operation_bucket_rows(profile, breakdown, abi_probe, selected_ms);
    let raw_graph_families = graph_operation_family_rows(abi_probe);
    let available = !buckets.is_empty() || !raw_graph_families.is_empty();
    let notes = operation_bucket_notes(breakdown.is_some());
    Some(OperationBucketDiagnostic {
        available,
        estimated_selected_ms_per_token: selected_ms,
        abi_ms_per_token: abi_ms,
        estimated_over_abi: ratio(selected_ms, abi_ms),
        buckets,
        raw_graph_families,
        notes,
    })
}

fn operation_bucket_rows(
    profile: &ModelProfile,
    breakdown: Option<&model_fit::DecodeCostBreakdown>,
    abi_probe: &AbiDecodeProbeSummary,
    selected_ms: Option<f64>,
) -> Vec<OperationBucketRow> {
    operation_bucket_specs(profile)
        .into_iter()
        .map(|spec| operation_bucket_row(spec, breakdown, abi_probe, selected_ms))
        .collect()
}

fn operation_bucket_specs(profile: &ModelProfile) -> Vec<OperationBucketSpec> {
    let mut specs = Vec::new();
    if profile.architecture_class == model_fit::ModelArchitectureClass::SparseMoeTransformer {
        specs.push(OperationBucketSpec {
            bucket: "sparse_transformer_block",
            graph_families: &["attention_matmul", "moe_matmul_id"],
            cost_group: "sparse_transformer_block",
            metadata_weight_bytes: graph_metadata_transformer_bytes(profile),
            note: "Sparse MoE scoring charges the llama.cpp token graph as attention plus routed expert GGML_OP_MUL_MAT_ID work; dense FFN buckets are not the right comparison for expert tensors.",
        });
        specs.push(OperationBucketSpec {
            bucket: "moe_router_and_runtime",
            graph_families: &["moe_runtime"],
            cost_group: "moe_router_and_runtime",
            metadata_weight_bytes: profile.tensor_matmul.feed_forward.bytes,
            note: "MoE router/gating work is already timed inside the sparse transformer block probe when that probe is selected; this row reports llama.cpp graph inventory only and must not borrow KV fallback timing or observed tok/s.",
        });
    } else {
        specs.push(OperationBucketSpec {
            bucket: "transformer_block",
            graph_families: &["attention_matmul", "ffn_matmul"],
            cost_group: "transformer_block",
            metadata_weight_bytes: graph_metadata_transformer_bytes(profile),
            note: "Scoring charges this as one scheduled llama.cpp token graph; the attention/FFN family split is diagnostic and must not become architecture-name logic.",
        });
    }
    specs.push(OperationBucketSpec {
        bucket: "output_matmul",
        graph_families: &["output_matmul"],
        cost_group: "output_matmul",
        metadata_weight_bytes: graph_expected_output_bytes(profile),
        note: "Output projection is separate from the repeated transformer block because vocab-sized logits can have a very different matrix shape.",
    });
    specs.push(OperationBucketSpec {
        bucket: "kv_and_activation",
        graph_families: &[
            "kv_cache",
            "attention_runtime",
            "ffn_runtime",
            "normalization",
        ],
        cost_group: "kv_and_activation",
        metadata_weight_bytes: 0,
        note: "Runtime buckets are source graph work, but model-fit currently estimates them as one metadata-derived non-weight group rather than independent per-op timings.",
    });
    specs.push(OperationBucketSpec {
        bucket: "unclassified_matmul",
        graph_families: &["matmul"],
        cost_group: "unclassified_matmul",
        metadata_weight_bytes: 0,
        note: "Unclassified matmul means the ABI graph saw GGML_OP_MUL_MAT nodes whose source names did not match the current diagnostic families; this is evidence to improve structural classification, not a model-family correction.",
    });
    specs
}

fn operation_bucket_row(
    spec: OperationBucketSpec,
    breakdown: Option<&model_fit::DecodeCostBreakdown>,
    abi_probe: &AbiDecodeProbeSummary,
    selected_ms: Option<f64>,
) -> OperationBucketRow {
    let group = breakdown.and_then(|breakdown| {
        breakdown
            .groups
            .iter()
            .find(|group| group.group == spec.cost_group)
    });
    let estimated_ms = group.map(|group| f64::from(group.bandwidth_ms));
    let estimated_traffic_bytes = group.map(|group| group.traffic_bytes).unwrap_or(0);
    OperationBucketRow {
        bucket: spec.bucket,
        source: group
            .map(|group| group.source.clone())
            .unwrap_or_else(|| "graph_inventory_only".into()),
        graph_families: spec.graph_families.to_vec(),
        estimated_ms,
        estimated_traffic_bytes,
        metadata_weight_bytes: spec.metadata_weight_bytes,
        graph_node_count: spec
            .graph_families
            .iter()
            .map(|family| graph_family_node_count(abi_probe, family))
            .sum(),
        graph_src0_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_src0_bytes(abi_probe, family))
            .sum(),
        graph_src1_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_src1_bytes(abi_probe, family))
            .sum(),
        graph_output_bytes: spec
            .graph_families
            .iter()
            .map(|family| graph_family_output_bytes(abi_probe, family))
            .sum(),
        graph_src0_over_metadata: ratio_u64(
            spec.graph_families
                .iter()
                .map(|family| graph_family_src0_bytes(abi_probe, family))
                .sum(),
            spec.metadata_weight_bytes,
        ),
        estimated_share_of_selected_ms: ratio(estimated_ms, selected_ms),
        notes: vec![spec.note.into()],
    }
}

fn graph_metadata_transformer_bytes(profile: &ModelProfile) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => profile
            .tensor_matmul
            .attention
            .bytes
            .saturating_add(profile.tensor_matmul.expert_feed_forward.bytes),
        _ => profile
            .tensor_matmul
            .attention
            .bytes
            .saturating_add(profile.tensor_matmul.feed_forward.bytes),
    }
}

fn graph_metadata_transformer_nodes(profile: &ModelProfile) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => profile
            .tensor_matmul
            .attention
            .shape
            .logical_matrix_count
            .saturating_add(sparse_moe_expected_expert_matmul_id_nodes(profile)),
        _ => profile
            .tensor_matmul
            .attention
            .shape
            .logical_matrix_count
            .saturating_add(
                profile
                    .tensor_matmul
                    .feed_forward
                    .shape
                    .logical_matrix_count,
            ),
    }
}

fn graph_transformer_src0_bytes(profile: &ModelProfile, abi_probe: &AbiDecodeProbeSummary) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => {
            graph_family_src0_bytes(abi_probe, "attention_matmul")
                .saturating_add(graph_family_src0_bytes(abi_probe, "moe_matmul_id"))
        }
        _ => graph_family_src0_bytes(abi_probe, "attention_matmul")
            .saturating_add(graph_family_src0_bytes(abi_probe, "ffn_matmul")),
    }
}

fn graph_transformer_node_count(profile: &ModelProfile, abi_probe: &AbiDecodeProbeSummary) -> u64 {
    match profile.architecture_class {
        model_fit::ModelArchitectureClass::SparseMoeTransformer => {
            graph_family_node_count(abi_probe, "attention_matmul")
                .saturating_add(graph_family_node_count(abi_probe, "moe_matmul_id"))
        }
        _ => graph_family_node_count(abi_probe, "attention_matmul")
            .saturating_add(graph_family_node_count(abi_probe, "ffn_matmul")),
    }
}

fn graph_operation_family_rows(abi_probe: &AbiDecodeProbeSummary) -> Vec<GraphOperationFamilyRow> {
    let mut rows = BTreeMap::<String, GraphOperationFamilyRow>::new();
    for bucket in &abi_probe.graph_inventory {
        let family = bucket.family.clone().unwrap_or_else(|| "unknown".into());
        let row = rows
            .entry(family.clone())
            .or_insert_with(|| GraphOperationFamilyRow {
                family,
                node_count: 0,
                src0_bytes: 0,
                src1_bytes: 0,
                output_bytes: 0,
                element_count: 0,
            });
        row.node_count = row
            .node_count
            .saturating_add(bucket.node_count.unwrap_or(0));
        row.src0_bytes = row
            .src0_bytes
            .saturating_add(bucket.src0_bytes.unwrap_or(0));
        row.src1_bytes = row
            .src1_bytes
            .saturating_add(bucket.src1_bytes.unwrap_or(0));
        row.output_bytes = row
            .output_bytes
            .saturating_add(bucket.output_bytes.unwrap_or(0));
        row.element_count = row
            .element_count
            .saturating_add(bucket.element_count.unwrap_or(0));
    }
    rows.into_values().collect()
}

fn operation_bucket_notes(has_breakdown: bool) -> Vec<String> {
    let mut notes = vec![
        "Operation buckets are llama.cpp/GGML graph families, not model families or filename rules."
            .into(),
        "These rows are validation diagnostics; observed benchmark throughput is not fed back into metadata-only scoring."
            .into(),
    ];
    if !has_breakdown {
        notes.push(
            "No decode cost breakdown was available, so rows report graph inventory without estimated bucket timing."
                .into(),
        );
    }
    notes
}

fn ratio_u64(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn graph_family_src1_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.src1_bytes)
        .sum()
}

fn graph_family_output_bytes(abi_probe: &AbiDecodeProbeSummary, family: &str) -> u64 {
    abi_probe
        .graph_inventory
        .iter()
        .filter(|bucket| bucket.family.as_deref() == Some(family))
        .filter_map(|bucket| bucket.output_bytes)
        .sum()
}

fn abi_ms_per_token(abi_probe: &AbiDecodeProbeSummary) -> Option<f64> {
    match (abi_probe.measured_tokens, abi_probe.elapsed_ms) {
        (Some(tokens), Some(elapsed_ms)) if tokens > 0 => Some(elapsed_ms / tokens as f64),
        _ => abi_probe
            .tokens_per_second
            .filter(|tps| *tps > 0.0)
            .map(|tps| 1000.0 / tps),
    }
}

fn graph_probe_layers_from_name(name: &str) -> u32 {
    for marker in [
        "_llama_graph_l",
        "_full_token_source_sampled_l",
        "_full_token_handoff_l",
        "_full_token_l",
        "_submission_l",
        "_moe_block_graph_l",
        "_moe_graph_l",
        "_linear_attn_graph_r",
    ] {
        if let Some(layers) = graph_probe_layers_after_marker(name, marker) {
            return layers;
        }
    }
    1
}

fn graph_probe_context_from_name(name: &str) -> Option<u32> {
    graph_probe_width_after_marker(name, "_nkv")
        .or_else(|| graph_probe_width_after_marker(name, "_ctx"))
}

fn graph_probe_width_after_marker(name: &str, marker: &str) -> Option<u32> {
    let (_, suffix) = name.split_once(marker)?;
    suffix
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>()
        .parse::<u32>()
        .ok()
        .filter(|value| *value > 0)
}

fn graph_probe_layers_after_marker(name: &str, marker: &str) -> Option<u32> {
    let (_, suffix) = name.split_once(marker)?;
    Some(
        suffix
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse::<u32>()
            .unwrap_or(1)
            .max(1),
    )
}

fn display_observed_over_abi(row: &ModelValidationReport) -> String {
    row.decode_probe_diagnostic
        .as_ref()
        .and_then(|diagnostic| diagnostic.observed_over_abi)
        .map(|ratio| format!("{ratio:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_decode_probe_classification(row: &ModelValidationReport) -> String {
    row.decode_probe_diagnostic
        .as_ref()
        .map(|diagnostic| diagnostic.classification.clone())
        .unwrap_or_else(|| "-".into())
}

fn display_fit_meaning(row: &ModelValidationReport) -> String {
    row.fit_interpretation
        .as_ref()
        .map(|fit| fit.summary.clone())
        .unwrap_or_else(|| "-".into())
}

fn display_selected_backend(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .map(|rec| match rec.selected_accelerator.as_deref() {
            Some(accelerator) => format!("{:?} ({accelerator})", rec.selected_backend),
            None => format!("{:?}", rec.selected_backend),
        })
        .unwrap_or_else(|| "-".into())
}

fn display_abi_decode_probe(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.tokens_per_second)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_abi_sampling_probe(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.sampling_tokens_per_second)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_abi_non_eval_overhead(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.abi_decode_probe
        .as_ref()
        .and_then(|probe| probe.non_eval_overhead_pct)
        .map(|pct| format!("{pct:.1}%"))
        .unwrap_or_else(|| "-".into())
}

fn scenario_verdict(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .map(|benchmark| benchmark.verdict.clone())
        .unwrap_or_else(|| "-".into())
}

fn scenario_summary_by_name<'a>(
    row: &'a ModelValidationReport,
    scenario: &str,
) -> Option<&'a BenchmarkScenarioSummary> {
    row.benchmarks
        .iter()
        .find(|benchmark| benchmark.scenario == scenario)
}

fn fit_status(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .map(|rec| format!("{:?}", rec.fit_status))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_tps(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_decode_context(row: &ModelValidationReport) -> String {
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_context_tokens)
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_steady_estimated_tps(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.predicted)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| display_estimated_tps(row))
}

fn display_steady_prediction_context(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.prediction_context_tokens)
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_scenario_estimated_tps(row: &ModelValidationReport, scenario: &str) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    scenario_summary_by_name(row, scenario)
        .and_then(|benchmark| benchmark.predicted)
        .map(|tps| format!("{tps:.1}"))
        .unwrap_or_else(|| "-".into())
}

fn display_scenario_prediction_context(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .and_then(|benchmark| benchmark.prediction_context_tokens)
        .map(|tokens| tokens.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_steady_estimated_range(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.predicted_range)
        .map(|range| format!("{:.1}-{:.1}", range.0, range.1))
        .unwrap_or_else(|| display_estimated_range(row))
}

fn display_steady_observed(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.observed)
        .or(row.benchmark.median_tokens_per_sec)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_steady_observed_over_fit(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.observed_over_fit)
        .or(row.benchmark.observed_over_fit)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_steady_observed_over_primary(row: &ModelValidationReport) -> String {
    scenario_summary_by_name(row, "steady_decode")
        .and_then(|benchmark| benchmark.primary_observed_over_fit)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_scenario_observed(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .and_then(|benchmark| benchmark.observed)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_scenario_observed_over_fit(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .and_then(|benchmark| benchmark.observed_over_fit)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_scenario_observed_over_primary(row: &ModelValidationReport, scenario: &str) -> String {
    scenario_summary_by_name(row, scenario)
        .and_then(|benchmark| benchmark.primary_observed_over_fit)
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_estimated_range(row: &ModelValidationReport) -> String {
    if !row_is_local_fit(row) {
        return "-".into();
    }
    row.recommendation
        .as_ref()
        .and_then(|rec| rec.estimated_decode_tokens_per_sec_range)
        .map(|range| format!("{:.1}-{:.1}", range.lower, range.upper))
        .unwrap_or_else(|| "-".into())
}

fn row_is_local_fit(row: &ModelValidationReport) -> bool {
    row.recommendation.as_ref().is_some_and(|rec| {
        matches!(
            rec.fit_status,
            FitStatus::FitsLocal | FitStatus::FitsWithWarning
        )
    })
}

fn display_opt(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_option_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_option_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_option_ratio(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "-".into())
}

fn display_option_ms(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.4}"))
        .unwrap_or_else(|| "-".into())
}

fn display_graph_node_ratio(diagnostic: &GraphInventoryDiagnostic) -> String {
    format!(
        "{}/{}",
        diagnostic.graph_transformer_matmul_nodes, diagnostic.metadata_transformer_matmul_nodes
    )
}

fn display_graph_inventory_notes(diagnostic: &GraphInventoryDiagnostic) -> String {
    if diagnostic.notes.is_empty() {
        return "-".into();
    }
    diagnostic
        .notes
        .iter()
        .map(|note| note.replace('|', "/"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn display_operation_bucket_notes(bucket: &OperationBucketRow) -> String {
    bucket
        .notes
        .first()
        .map(|note| note.replace('|', "/"))
        .unwrap_or_else(|| "-".into())
}

fn heartbeat(model_index: Option<usize>, model_ref: &str, phase: &str, detail: &str) {
    let index = model_index
        .map(|index| index.to_string())
        .unwrap_or_else(|| "-".into());
    let detail = detail.replace(['\r', '\n'], " ");
    eprintln!(
        "[model-fit-validate] model_index={index} phase={phase} model_ref={:?} {detail}",
        model_ref
    );
}

fn profile_summary(profile: &ModelProfile) -> String {
    format!(
        "architecture={} layers={} hidden={} ctx={} quant={} params={} file_bytes={}",
        profile.architecture.as_deref().unwrap_or("-"),
        display_u32(profile.layer_count),
        display_u32(profile.hidden_size),
        display_u32(profile.context_length),
        profile.quantization.as_deref().unwrap_or("-"),
        display_u64(profile.parameter_count),
        profile.file_size_bytes
    )
}

fn display_u32(value: Option<u32>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn display_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".into())
}

fn abi_probe_observation_detail(observation: &AbiDecodeProbeObservation) -> String {
    format!(
        "repeat={} status={} tok_s={} sample_sync_tok_s={} elapsed_ms={} decode_call_ms={} sampling_ms={} logits_ready_ms={} logits_scan_ms={} overhead_ms={} error={}",
        observation.repeat + 1,
        status_label(observation.status_code),
        display_opt(observation.tokens_per_second),
        display_opt(observation.sampling_tokens_per_second),
        display_opt(observation.elapsed_ms),
        display_opt(observation.decode_call_ms),
        display_opt(observation.sampling_ms),
        display_opt(observation.logits_ready_ms),
        display_opt(observation.logits_scan_ms),
        display_opt(observation.non_eval_overhead_ms),
        observation.error.as_deref().unwrap_or("-")
    )
}

fn benchmark_observation_detail(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> String {
    format!(
        "scenario={} repeat={} status={} wall_s={:.2} metric={} error={}",
        scenario.name,
        observation.repeat + 1,
        status_label(observation.status_code),
        observation.wall_seconds,
        display_opt(benchmark_observation_metric(observation, scenario)),
        observation.error.as_deref().unwrap_or("-")
    )
}

fn benchmark_observation_metric(
    observation: &BenchmarkObservation,
    scenario: &BenchmarkScenarioSpec,
) -> Option<f64> {
    match scenario.kind {
        BenchmarkScenarioKind::SteadyDecode | BenchmarkScenarioKind::PrimaryContextSteadyDecode => {
            steady_decode_observation_tokens_per_sec(observation)
        }
        BenchmarkScenarioKind::Prefill => prefill_observation_tokens_per_sec(observation),
        BenchmarkScenarioKind::FirstToken => observation.text_request_elapsed_ms,
        BenchmarkScenarioKind::KvWarmReuse => observation
            .request_results
            .last()
            .and_then(|request| request.generated_tokens_per_sec),
    }
}

fn scenario_summary_detail(summary: &BenchmarkScenarioSummary) -> String {
    format!(
        "scenario={} verdict={} observed={} observed_over_fit={}",
        summary.scenario,
        summary.verdict,
        display_opt(summary.observed),
        display_opt(summary.observed_over_fit)
    )
}

fn status_label(status_code: Option<i32>) -> String {
    status_code
        .map(|code| code.to_string())
        .unwrap_or_else(|| "-".into())
}

fn parse_local_model(value: &str) -> Result<LocalModelInput> {
    let mut fields = BTreeMap::new();
    for pair in value.split(',') {
        let Some((key, value)) = pair.split_once('=') else {
            bail!("model field must be key=value: {pair}");
        };
        fields.insert(key.trim(), value.trim());
    }
    Ok(LocalModelInput {
        model_ref: required_field(&fields, "ref")?.to_string(),
        gguf_path: PathBuf::from(required_field(&fields, "path")?),
    })
}

fn required_field<'a>(fields: &'a BTreeMap<&str, &str>, key: &str) -> Result<&'a str> {
    fields
        .get(key)
        .copied()
        .with_context(|| format!("missing model field {key}"))
}

fn input_label(input: &ModelInput) -> String {
    match input {
        ModelInput::Ref(model_ref) => model_ref.clone(),
        ModelInput::Local(local) => local.model_ref.clone(),
    }
}

fn read_json_input(path: &Path) -> Result<Vec<u8>> {
    if path == Path::new("-") {
        use std::io::Read;
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .context("read JSON from stdin")?;
        return Ok(bytes);
    }
    fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn write_json_report(path: &Path, report: &ValidationReport) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn command_display(bin: &Path, args: &[String]) -> Vec<String> {
    std::iter::once(bin.display().to_string())
        .chain(args.iter().cloned())
        .collect()
}

#[derive(Clone)]
struct TerminalDownloadProgress {
    model_ref: Arc<String>,
    state: Arc<Mutex<TerminalDownloadProgressState>>,
}

#[derive(Default)]
struct TerminalDownloadProgressState {
    last_draw: Option<Instant>,
    active_line: bool,
}

impl TerminalDownloadProgress {
    fn new(model_ref: &str) -> Self {
        Self {
            model_ref: Arc::new(model_ref.to_string()),
            state: Arc::new(Mutex::new(TerminalDownloadProgressState::default())),
        }
    }

    fn report(&self, event: ModelDownloadProgressEvent) {
        match event {
            ModelDownloadProgressEvent::Ensuring {
                file,
                index,
                total_files,
                total_bytes,
            } => self.draw(
                format!(
                    "Ensuring {} [{}/{}] {}{}",
                    self.model_ref,
                    index,
                    total_files,
                    file,
                    total_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                false,
            ),
            ModelDownloadProgressEvent::Started {
                file, total_bytes, ..
            } => self.draw(
                format!(
                    "Downloading {} {}{}",
                    self.model_ref,
                    file,
                    total_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                false,
            ),
            ModelDownloadProgressEvent::Progress {
                file,
                downloaded_bytes,
                total_bytes,
                bytes_per_sec,
            } => self.draw(
                download_progress_line(
                    &self.model_ref,
                    &file,
                    downloaded_bytes,
                    total_bytes,
                    bytes_per_sec,
                ),
                false,
                false,
            ),
            ModelDownloadProgressEvent::Ready {
                file,
                index,
                total_files,
                size_bytes,
                ..
            } => self.draw(
                format!(
                    "Ready {} [{}/{}] {}{}",
                    self.model_ref,
                    index,
                    total_files,
                    file,
                    size_bytes
                        .map(|bytes| format!(" ({})", format_bytes(bytes)))
                        .unwrap_or_default()
                ),
                true,
                true,
            ),
            ModelDownloadProgressEvent::Complete { .. } => {}
        }
    }

    fn draw(&self, message: String, force: bool, finish_line: bool) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        let now = Instant::now();
        if !force
            && state
                .last_draw
                .is_some_and(|last| now.duration_since(last) < Duration::from_millis(150))
        {
            return;
        }
        state.last_draw = Some(now);
        state.active_line = !finish_line;
        eprint!("\r\x1b[2K{message}");
        if finish_line {
            eprintln!();
        }
        let _ = std::io::stderr().flush();
    }
}

impl Drop for TerminalDownloadProgress {
    fn drop(&mut self) {
        if Arc::strong_count(&self.state) != 1 {
            return;
        }
        match self.state.lock() {
            Ok(state) if state.active_line => eprintln!(),
            _ => {}
        }
    }
}

struct TerminalStatus {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl TerminalStatus {
    fn start(enabled: bool, message: String) -> Self {
        if !enabled {
            return Self {
                done: Arc::new(AtomicBool::new(true)),
                thread: None,
            };
        }
        let done = Arc::new(AtomicBool::new(false));
        let done_thread = Arc::clone(&done);
        let thread = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut index = 0usize;
            while !done_thread.load(AtomicOrdering::Relaxed) {
                eprint!("\r\x1b[2K{} {}", frames[index % frames.len()], message);
                let _ = std::io::stderr().flush();
                index += 1;
                thread::sleep(Duration::from_millis(120));
            }
        });
        Self {
            done,
            thread: Some(thread),
        }
    }

    fn finish(&mut self) {
        self.done.store(true, AtomicOrdering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        eprint!("\r\x1b[2K");
        let _ = std::io::stderr().flush();
    }
}

impl Drop for TerminalStatus {
    fn drop(&mut self) {
        self.finish();
    }
}

fn download_progress_line(
    model_ref: &str,
    file: &str,
    downloaded: u64,
    total: Option<u64>,
    bytes_per_sec: Option<f64>,
) -> String {
    let speed = bytes_per_sec
        .filter(|speed| *speed > 0.0)
        .map(|speed| format!(" at {}/s", format_bytes(speed as u64)))
        .unwrap_or_default();
    if let Some(total) = total.filter(|total| *total > 0) {
        let percent = (downloaded.min(total) as f64 / total as f64) * 100.0;
        format!(
            "Downloading {model_ref} {file} {:>5.1}% ({}/{}){speed}",
            percent,
            format_bytes(downloaded),
            format_bytes(total)
        )
    } else {
        format!(
            "Downloading {model_ref} {file} {}{speed}",
            format_bytes(downloaded)
        )
    }
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];
    for candidate in &UNITS[1..] {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = candidate;
    }
    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.1} {unit}")
    }
}

fn tail_lines(text: &str, max_lines: usize) -> String {
    let lines = text.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(max_lines);
    lines[start..].join("\n")
}

fn median(samples: &[f64]) -> f64 {
    let mid = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    }
}

fn mean(samples: &[f64]) -> Option<f64> {
    (!samples.is_empty()).then(|| samples.iter().sum::<f64>() / samples.len() as f64)
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_value(args, name)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn unix_timestamp_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

fn print_usage() {
    eprintln!(
        "usage: model-fit-validate [--output-json report.json] [--models-file refs.txt] [--scenario steady_decode|steady_decode_primary_context|prefill|first_token|kv_warm_reuse|all] [--dense-probe-depth standard|deep] [--benchmark-all] [--fit-only] [--skip-context-aligned-abi] [--no-progress] [--allow-debug-validation] org/repo:Q4_K_M [org/repo:Q5_K_M ...]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_fit::{
        CapabilityEvidence, DenseGraphFeatures, ModelArchitectureClass, ModelSource,
        RecurrentAttentionProfile, RopeProfile, TensorGroupBytes, TensorMatmulGroupProfile,
        TensorMatmulProfile, TensorTypeBytes, TokenizerProfile, WeightCoverage,
    };

    fn test_args() -> Args {
        Args {
            output_json: PathBuf::from("validation.json"),
            skippy_bench_bin: PathBuf::from("skippy-bench"),
            skippy_server_bin: PathBuf::from("skippy-server"),
            metrics_server_bin: PathBuf::from("metrics-server"),
            gpu_benchmark_json: None,
            model_files: Vec::new(),
            benchmark_scenarios: Vec::new(),
            base_port: 18400,
            benchmark_all: false,
            fit_only: false,
            dense_probe_depth: DenseProbeDepth::Standard,
            show_progress: false,
            skip_context_aligned_abi: false,
            allow_debug_validation: false,
            models: vec![ModelInput::Ref(
                "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M".into(),
            )],
        }
    }

    fn minimal_dense_profile() -> ModelProfile {
        ModelProfile {
            source: ModelSource {
                id: "test/model:Q5_K_M".into(),
                path: None,
                metadata_name: None,
            },
            architecture: Some("qwen2".into()),
            architecture_class: ModelArchitectureClass::DenseTransformer,
            weight_coverage: WeightCoverage::Full,
            file_size_bytes: 1,
            tensor_bytes: Some(1),
            base_resident_bytes: Some(1),
            expert_tensor_bytes: Some(0),
            tensor_group_bytes: TensorGroupBytes::default(),
            tensor_matmul: TensorMatmulProfile::default(),
            dense_graph_features: DenseGraphFeatures::default(),
            recurrent_attention: RecurrentAttentionProfile::default(),
            parameter_count: None,
            quantization: Some("Q5_K_M".into()),
            layer_count: Some(28),
            hidden_size: Some(3584),
            ffn_size: Some(18944),
            attention_heads: Some(28),
            kv_heads: Some(4),
            key_length: Some(128),
            value_length: Some(128),
            context_length: Some(32768),
            expert_count: None,
            expert_used_count: None,
            rope: RopeProfile::default(),
            tokenizer: TokenizerProfile::default(),
            capability_evidence: Vec::new(),
        }
    }

    #[test]
    fn dense_probe_tensor_types_include_q5_k_when_metadata_contains_q5_weights() {
        let mut profile = minimal_dense_profile();
        profile.tensor_matmul.attention = TensorMatmulGroupProfile {
            type_bytes: TensorTypeBytes {
                q5_k_bytes: 500,
                q6_k_bytes: 100,
                ..TensorTypeBytes::default()
            },
            ..TensorMatmulGroupProfile::default()
        };
        profile.tensor_matmul.feed_forward = TensorMatmulGroupProfile {
            type_bytes: TensorTypeBytes {
                q5_k_bytes: 5_000,
                q6_k_bytes: 1_000,
                ..TensorTypeBytes::default()
            },
            ..TensorMatmulGroupProfile::default()
        };

        assert_eq!(dense_probe_tensor_types(&profile), vec!["q5_k", "q6_k"]);
        assert!(supports_dense_depth_probe_tensor_type("q5_k"));
    }

    #[test]
    fn registers_primary_context_steady_decode_scenario() {
        let scenarios = benchmark_scenarios();
        let steady_index = scenarios
            .iter()
            .position(|scenario| scenario.name == "steady_decode")
            .expect("steady_decode scenario");
        let primary_index = scenarios
            .iter()
            .position(|scenario| scenario.name == "steady_decode_primary_context")
            .expect("steady_decode_primary_context scenario");

        assert_eq!(
            scenarios[primary_index].kind,
            BenchmarkScenarioKind::PrimaryContextSteadyDecode
        );
        assert_eq!(primary_index, steady_index + 1);
        assert_eq!(scenarios[primary_index].warmup_tokens, 0);
        assert_eq!(scenarios[primary_index].request_count, 1);
    }

    #[test]
    fn standard_attention_runtime_probes_include_primary_context_scale() {
        let args = Args {
            dense_probe_depth: DenseProbeDepth::Standard,
            ..test_args()
        };

        assert_eq!(
            attention_runtime_context_ladder(&args),
            vec![128, 512, DEFAULT_CTX_SIZE]
        );
    }

    #[test]
    fn validation_full_token_probe_uses_required_context_capacity() {
        let mut config = selection_config(&primary_workload_profile());
        config.workload.interaction.expected_prompt_tokens = Some(4096);
        config.workload.requirements.min_context_tokens = Some(8192);
        let profile = ModelProfile {
            source: ModelSource {
                id: "test/smol:Q8_0".into(),
                path: None,
                metadata_name: None,
            },
            architecture: Some("llama".into()),
            architecture_class: ModelArchitectureClass::DenseTransformer,
            weight_coverage: WeightCoverage::Full,
            file_size_bytes: 1,
            tensor_bytes: Some(1),
            base_resident_bytes: Some(1),
            expert_tensor_bytes: Some(0),
            tensor_group_bytes: TensorGroupBytes::default(),
            tensor_matmul: TensorMatmulProfile::default(),
            dense_graph_features: DenseGraphFeatures::default(),
            recurrent_attention: RecurrentAttentionProfile::default(),
            parameter_count: None,
            quantization: Some("Q8_0".into()),
            layer_count: Some(30),
            hidden_size: Some(576),
            ffn_size: Some(1536),
            attention_heads: Some(9),
            kv_heads: Some(3),
            key_length: Some(64),
            value_length: Some(64),
            context_length: Some(8192),
            expert_count: None,
            expert_used_count: None,
            rope: RopeProfile::default(),
            tokenizer: TokenizerProfile {
                model: None,
                vocab_size: Some(49_152),
                chat_template_available: true,
            },
            capability_evidence: vec![CapabilityEvidence::NativeContextAtLeast(8192)],
        };

        assert_eq!(
            decode_context_tokens_for_validation(&config, &profile),
            8192
        );
        assert_eq!(
            active_decode_context_tokens_for_validation(&config, &profile, 8192),
            4352
        );
    }

    #[test]
    fn graph_probe_layer_parser_handles_full_token_probe_names() {
        assert_eq!(
            graph_probe_layers_from_name(
                "ggml_decode_q8_0_full_token_handoff_l30_gqa_576_kv192_1536_vocab49152_ctx4096_h64_qh9_kvh3",
            ),
            30
        );
        assert_eq!(
            graph_probe_context_from_name(
                "ggml_decode_q8_0_full_token_handoff_l30_gqa_576_kv192_1536_vocab49152_ctx8192_nkv4352_h64_qh9_kvh3",
            ),
            Some(4352)
        );
        assert_eq!(
            graph_probe_layers_from_name(
                "ggml_decode_q8_0_full_token_source_sampled_l30_gqa_576_kv192_1536_vocab49152_ctx8192_nkv4352_h64_qh9_kvh3",
            ),
            30
        );
        assert_eq!(
            graph_probe_context_from_name(
                "ggml_decode_q8_0_full_token_source_sampled_l30_gqa_576_kv192_1536_vocab49152_ctx8192_nkv4352_h64_qh9_kvh3",
            ),
            Some(4352)
        );
        assert_eq!(
            graph_probe_layers_from_name(
                "ggml_decode_q8_0_submission_l30_gqa_576_kv192_1536_vocab49152_ctx4096_h64_qh9_kvh3",
            ),
            30
        );
        assert_eq!(
            graph_probe_context_from_name(
                "ggml_decode_q8_0_submission_l30_gqa_576_kv192_1536_vocab49152_ctx4096_h64_qh9_kvh3",
            ),
            Some(4096)
        );
    }

    #[test]
    fn synthetic_primary_context_prompt_preserves_requested_word_count() {
        let prompt = synthetic_word_prompt(96);
        let body = prompt
            .lines()
            .skip(1)
            .take_while(|line| !line.starts_with("Explain "))
            .collect::<Vec<_>>()
            .join(" ");
        let context_words = body
            .split_whitespace()
            .filter(|word| *word == "context")
            .count();

        assert_eq!(context_words, 96);
        assert!(prompt.contains("tokenizer-reported prompt length"));
    }

    #[test]
    fn classifies_noisy_context_rescore_when_primary_is_more_representative() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(128.2),
            abi: Some(157.8),
            observed: Some(143.5),
            missing_representative_model_probe: false,
            abi_probe_noisy: true,
            request_decode_noisy: true,
            request_spread_pct: Some(41.8),
            observed_over_fit: Some(143.5 / 128.2),
            observed_over_scenario_fit: Some(143.5 / 543.4),
            abi_over_fit: Some(157.8 / 128.2),
            abi_over_scenario_fit: Some(157.8 / 543.4),
            observed_over_abi: Some(143.5 / 157.8),
            abi_sampling_over_selected_fit: None,
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: None,
        });

        assert_eq!(classification, "context_rescore_unstable");
    }

    #[test]
    fn classifies_primary_agreement_when_context_rescore_is_unstable() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(83.0),
            abi: Some(77.3),
            observed: Some(78.2),
            missing_representative_model_probe: false,
            abi_probe_noisy: true,
            request_decode_noisy: true,
            request_spread_pct: Some(60.2),
            observed_over_fit: Some(78.2 / 83.0),
            observed_over_scenario_fit: Some(78.2 / 99.1),
            abi_over_fit: Some(77.3 / 83.0),
            abi_over_scenario_fit: Some(77.3 / 99.1),
            observed_over_abi: Some(78.2 / 77.3),
            abi_sampling_over_selected_fit: None,
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: None,
        });

        assert_eq!(
            classification,
            "primary_estimate_and_probe_agree_context_rescore_unstable"
        );
    }

    #[test]
    fn classifies_noisy_scenario_agreement_separately_from_primary_miss() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(43.6),
            abi: Some(67.6),
            observed: Some(63.2),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(25.3),
            observed_over_fit: Some(63.2 / 43.6),
            observed_over_scenario_fit: Some(63.2 / 64.6),
            abi_over_fit: Some(67.6 / 43.6),
            abi_over_scenario_fit: Some(67.6 / 64.6),
            observed_over_abi: Some(63.2 / 67.6),
            abi_sampling_over_selected_fit: None,
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: None,
        });

        assert_eq!(
            classification,
            "scenario_estimate_and_probe_agree_noisy_requests"
        );
    }

    #[test]
    fn classifies_context_mismatch_when_scenario_and_probe_match() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(150.0),
            abi: Some(173.6),
            observed: Some(171.0),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: false,
            request_spread_pct: Some(0.8),
            observed_over_fit: Some(171.0 / 150.0),
            observed_over_scenario_fit: Some(171.0 / 171.1),
            abi_over_fit: Some(173.6 / 150.0),
            abi_over_scenario_fit: Some(173.6 / 171.1),
            observed_over_abi: Some(171.0 / 173.6),
            abi_sampling_over_selected_fit: None,
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: None,
        });

        assert_eq!(classification, "primary_context_differs_from_benchmark");
    }

    #[test]
    fn classifies_sampler_sync_residual_before_generic_noisy_miss() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(143.6),
            abi: Some(119.8),
            observed: Some(110.2),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(34.0),
            observed_over_fit: Some(110.2 / 143.6),
            observed_over_scenario_fit: Some(110.2 / 142.4),
            abi_over_fit: Some(119.8 / 143.6),
            abi_over_scenario_fit: Some(119.8 / 142.4),
            observed_over_abi: Some(110.2 / 119.8),
            abi_sampling_over_selected_fit: None,
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: Some(0.74),
        });

        assert_eq!(classification, "sampler_sync_residual_noisy");
    }

    #[test]
    fn classifies_source_sampled_probe_gap_as_llama_boundary_residual() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(150.6),
            abi: Some(119.9),
            observed: Some(120.3),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(29.4),
            observed_over_fit: Some(120.3 / 150.6),
            observed_over_scenario_fit: Some(120.3 / 243.9),
            abi_over_fit: Some(119.9 / 150.6),
            abi_over_scenario_fit: Some(119.9 / 243.9),
            observed_over_abi: Some(120.3 / 119.9),
            abi_sampling_over_selected_fit: Some(5.33 / 6.64),
            selected_full_token_handoff_probe: false,
            selected_full_token_source_sampled_probe: true,
            decode_submission_residual_share_of_predicted: None,
            sampler_sync_residual_share_of_predicted: Some(0.80),
        });

        assert_eq!(classification, "llama_source_boundary_residual_noisy");
    }

    #[test]
    fn classifies_noisy_source_boundary_residual_even_when_observed_abi_is_noisy() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(148.2),
            abi: Some(120.3),
            observed: Some(108.0),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(12.1),
            observed_over_fit: Some(108.0 / 148.2),
            observed_over_scenario_fit: Some(108.0 / 146.9),
            abi_over_fit: Some(120.3 / 148.2),
            abi_over_scenario_fit: Some(120.3 / 146.9),
            observed_over_abi: Some(108.0 / 120.3),
            abi_sampling_over_selected_fit: Some(5.3 / 6.7),
            selected_full_token_handoff_probe: true,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: Some(0.38),
            sampler_sync_residual_share_of_predicted: Some(0.78),
        });

        assert_eq!(classification, "decode_and_sampler_residual_noisy");
    }

    #[test]
    fn classifies_full_token_handoff_gap_as_decode_submission_residual() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(188.5),
            abi: Some(120.3),
            observed: Some(109.1),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(10.7),
            observed_over_fit: Some(109.1 / 188.5),
            observed_over_scenario_fit: Some(109.1 / 188.5),
            abi_over_fit: Some(120.3 / 188.5),
            abi_over_scenario_fit: Some(120.3 / 188.5),
            observed_over_abi: Some(109.1 / 120.3),
            abi_sampling_over_selected_fit: Some(5.32 / 5.31),
            selected_full_token_handoff_probe: true,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: Some(0.34),
            sampler_sync_residual_share_of_predicted: None,
        });

        assert_eq!(classification, "decode_submission_residual_noisy");
    }

    #[test]
    fn classifies_combined_decode_and_sampler_residual() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(234.4),
            abi: Some(121.2),
            observed: Some(121.0),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(23.8),
            observed_over_fit: Some(121.0 / 234.4),
            observed_over_scenario_fit: Some(121.0 / 234.4),
            abi_over_fit: Some(121.2 / 234.4),
            abi_over_scenario_fit: Some(121.2 / 234.4),
            observed_over_abi: Some(121.0 / 121.2),
            abi_sampling_over_selected_fit: Some(5.28 / 4.27),
            selected_full_token_handoff_probe: true,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: Some(0.64),
            sampler_sync_residual_share_of_predicted: Some(1.22),
        });

        assert_eq!(classification, "decode_and_sampler_residual_noisy");
    }

    #[test]
    fn source_boundary_residual_outranks_context_rescore_unstable() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(165.5),
            abi: Some(117.9),
            observed: Some(119.3),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(26.3),
            observed_over_fit: Some(119.3 / 165.5),
            observed_over_scenario_fit: Some(119.3 / 248.8),
            abi_over_fit: Some(117.9 / 165.5),
            abi_over_scenario_fit: Some(117.9 / 248.8),
            observed_over_abi: Some(119.3 / 117.9),
            abi_sampling_over_selected_fit: Some(5.41 / 6.04),
            selected_full_token_handoff_probe: true,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: Some(0.44),
            sampler_sync_residual_share_of_predicted: Some(0.89),
        });

        assert_eq!(classification, "decode_and_sampler_residual_noisy");
    }

    #[test]
    fn classifies_abi_only_residual_as_probe_not_representative() {
        let classification = decode_probe_classification(DecodeProbeClassificationInput {
            predicted: Some(66.1),
            abi: Some(55.8),
            observed: Some(61.4),
            missing_representative_model_probe: false,
            abi_probe_noisy: false,
            request_decode_noisy: true,
            request_spread_pct: Some(11.5),
            observed_over_fit: Some(61.4 / 66.1),
            observed_over_scenario_fit: Some(61.4 / 65.7),
            abi_over_fit: Some(55.8 / 66.1),
            abi_over_scenario_fit: Some(55.8 / 65.7),
            observed_over_abi: Some(61.4 / 55.8),
            abi_sampling_over_selected_fit: Some(15.1 / 15.2),
            selected_full_token_handoff_probe: true,
            selected_full_token_source_sampled_probe: false,
            decode_submission_residual_share_of_predicted: Some(0.11),
            sampler_sync_residual_share_of_predicted: Some(0.99),
        });

        assert_eq!(classification, "probe_not_representative");
    }
}
