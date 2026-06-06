use serde::{Deserialize, Serialize};

pub const GRAPH_FEATURE_ATTENTION_Q_NORM: u32 = 1 << 0;
pub const GRAPH_FEATURE_ATTENTION_K_NORM: u32 = 1 << 1;
pub const GRAPH_FEATURE_ATTENTION_POST_NORM: u32 = 1 << 2;
pub const GRAPH_FEATURE_FFN_POST_NORM: u32 = 1 << 3;

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct BenchmarkOutput {
    pub device: String,
    pub buffer_mb: u32,
    pub runs: u32,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    #[serde(default)]
    pub decode_effective_gbps: Option<f64>,
    #[serde(default)]
    pub decode_fixed_overhead_ms: Option<f64>,
    #[serde(default)]
    pub decode_runtime_overhead_ms: Option<f64>,
    #[serde(default)]
    pub post_prefill_decode_overhead_ms: Option<f64>,
    pub compute_tflops_fp32: Option<f64>,
    pub compute_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_ubatch_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub prefill_moe_matmul_tflops_fp16: Option<f64>,
    #[serde(default)]
    pub sampler_history_us_per_token: Option<f64>,
    #[serde(default)]
    pub sampler_vocab_us_per_token: Option<f64>,
    #[serde(default)]
    pub decode_kernel_probes: Vec<DecodeKernelProbe>,
    pub noise_pct: f64,
    pub runtime_s: f64,
    pub rated_gbps: Option<f64>,
    pub rated_estimated: Option<bool>,
    pub efficiency_pct: Option<f64>,
    pub bus_width_bits: Option<u32>,
    pub mem_clock_mhz: Option<u64>,
    pub gcn_arch: Option<String>,
    pub hbm: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DecodeKernelProbe {
    pub name: String,
    pub tensor_type: String,
    pub rows: u32,
    pub cols: u32,
    pub batch_tokens: u32,
    #[serde(default)]
    pub graph_features: u32,
    #[serde(default)]
    pub graph_node_count: Option<u64>,
    pub effective_gbps: f64,
    pub tflops: Option<f64>,
    #[serde(default)]
    pub elapsed_ms: Option<f64>,
    #[serde(default)]
    pub min_elapsed_ms: Option<f64>,
    #[serde(default)]
    pub max_elapsed_ms: Option<f64>,
    #[serde(default)]
    pub spread_pct: Option<f64>,
    /// Optional GGML graph inventory for composite/source-shaped probes.
    ///
    /// This is diagnostic evidence, not a scoring input. It lets validation
    /// compare a synthetic probe's submitted graph against the ABI-observed
    /// llama.cpp graph bucket-by-bucket: family, operation, tensor type, node
    /// count, and byte counters. Keeping this on the probe makes validation
    /// JSON self-contained while preserving the rule that fit scoring must come
    /// from model metadata plus measured hardware/probe facts, not observed
    /// model throughput.
    #[serde(default)]
    pub graph_inventory: Vec<DecodeGraphInventoryBucket>,
    pub runs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DecodeGraphInventoryBucket {
    pub family: Option<String>,
    pub ggml_op: Option<i64>,
    pub ggml_type: Option<u64>,
    pub node_count: Option<u64>,
    pub element_count: Option<u64>,
    pub output_bytes: Option<u64>,
    pub src0_bytes: Option<u64>,
    pub src1_bytes: Option<u64>,
    #[serde(default)]
    pub ne: Vec<i64>,
}
