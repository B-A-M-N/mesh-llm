mod gpu_benchmark;
mod hf_cache;
mod profile;
mod scoring;
mod types;
mod validation_stats;
mod workload;

pub use gpu_benchmark::{
    GpuBenchmarkAcceleratorFacts, GpuBenchmarkHardwareInput, GpuBenchmarkOutput,
    hardware_profile_from_gpu_benchmark,
};
pub use hf_cache::{HfCacheModelProfile, profile_hf_cache};
pub use profile::profile_gguf_path;
pub use scoring::{
    estimate_kv_cache_bytes, estimate_runtime_memory_bytes, rank_models, score_model,
    score_model_for_context_tokens,
};
pub use types::{
    AcceleratorKind, AcceleratorProfile, BackendEfficiencyConfig, BackendKind, CapabilityEvidence,
    CapabilityRequirements, CpuProfile, DecodeCostBreakdown, DecodeCostGroupBreakdown,
    DecodeEstimateRange, DecodeOverheadConfig, DenseGraphFeatures, EstimateConfidence,
    FirstTokenEstimateRange, FitStatus, HardwareProfile, InteractionProfile, KvCacheKind,
    KvCacheType, MatmulShapeProfile, MeasurementSource, MemoryProfile, ModelArchitectureClass,
    ModelProfile, ModelRecommendation, ModelSource, RecurrentAttentionProfile, Requirement,
    RopeProfile, ScoreWeights, SelectionConfig, TensorGroupBytes, TensorMatmulGroupProfile,
    TensorMatmulProfile, TensorTypeBytes, TokenizerProfile, WeightCoverage, WorkloadPreferences,
    WorkloadProfile, WorkloadTask,
};
pub use validation_stats::{ThroughputSampleStats, throughput_sample_stats};

#[cfg(test)]
mod tests;
