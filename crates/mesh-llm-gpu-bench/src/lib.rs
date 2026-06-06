mod output;
mod runner;

#[cfg(any(feature = "cuda", feature = "hip", feature = "intel"))]
mod capture;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "hip")]
mod hip;

#[cfg(feature = "intel")]
mod intel;

#[cfg(all(feature = "ggml-probe", mesh_llm_gpu_bench_has_ggml_probe))]
mod ggml_probe;

#[cfg(target_os = "macos")]
mod metal;

pub use output::{
    BenchmarkOutput, DecodeGraphInventoryBucket, DecodeKernelProbe, GRAPH_FEATURE_ATTENTION_K_NORM,
    GRAPH_FEATURE_ATTENTION_POST_NORM, GRAPH_FEATURE_ATTENTION_Q_NORM, GRAPH_FEATURE_FFN_POST_NORM,
};
pub use runner::{
    AttentionRuntimeProbeShape, BenchmarkBackend, BenchmarkOptions, BenchmarkRunner,
    DenseFullTokenProbeShape, DenseGraphProbeShape, DenseSampledTokenProbeShape,
    LinearAttentionGraphProbeShape, LogitsReadbackProbeShape, MoeBlockGraphProbeShape,
    OutputProjectionProbeShape, ProbeDepth, parse_benchmark_output, run_benchmark,
    run_benchmark_with_options, run_model_attention_runtime_probe,
    run_model_dense_decode_submission_probe, run_model_dense_full_token_handoff_probe,
    run_model_dense_full_token_probe, run_model_dense_graph_probe,
    run_model_dense_sampled_token_probe, run_model_dense_source_sampled_token_probe,
    run_model_linear_attention_graph_probe, run_model_logits_output_handoff_probe,
    run_model_logits_readback_probe, run_model_logits_sync_probe,
    run_model_moe_block_decode_submission_probe, run_model_moe_block_graph_probe,
    run_model_moe_graph_probe, run_model_output_projection_probe, runner_for,
};
