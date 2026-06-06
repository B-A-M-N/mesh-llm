use anyhow::{Context, Result, bail};
use serde_json::json;
use skippy_runtime::{RuntimeConfig, RuntimeLoadMode, StageModel};

use crate::cli::AbiDecodeProbeArgs;

pub fn abi_decode_probe(args: AbiDecodeProbeArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if args.measured_tokens == 0 {
        bail!("measured_tokens must be greater than zero");
    }

    let model = StageModel::open(
        &args.model_path,
        &RuntimeConfig {
            stage_index: 0,
            layer_start: args.layer_start,
            layer_end: args.layer_end,
            ctx_size: args.ctx_size,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_threads: None,
            n_threads_batch: None,
            n_gpu_layers: args.n_gpu_layers,
            selected_backend_device: None,
            cache_type_k: skippy_runtime::GGML_TYPE_F16,
            cache_type_v: skippy_runtime::GGML_TYPE_F16,
            flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
            load_mode: RuntimeLoadMode::RuntimeSlice,
            projector_path: None,
            include_embeddings: true,
            include_output: true,
            filter_tensors_on_load: false,
        },
    )
    .with_context(|| format!("open model {}", args.model_path.display()))?;

    let prompt_tokens = model
        .tokenize(&args.prompt, true)
        .context("tokenize probe prompt")?;
    let seed_token = *prompt_tokens
        .last()
        .context("probe prompt produced no tokens")?;
    let mut session = model.create_session().context("create probe session")?;
    if prompt_tokens.len() > 1 {
        session
            .prefill_chunked(&prompt_tokens[..prompt_tokens.len() - 1])
            .context("prefill probe prompt")?;
    }
    let result = session
        .benchmark_decode(seed_token, args.warmup_tokens, args.measured_tokens)
        .context("run native decode benchmark")?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "abi-decode-probe",
            "model_path": args.model_path,
            "ctx_size": args.ctx_size,
            "n_gpu_layers": args.n_gpu_layers,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "prompt_token_count": prompt_tokens.len(),
            "warmup_tokens": result.warmup_tokens,
            "measured_tokens": result.measured_tokens,
            "elapsed_ms": result.elapsed_ms,
            "tokens_per_second": result.tokens_per_second,
            "final_token": result.final_token,
            "llama_eval_count": result.llama_eval_count,
            "llama_graph_reuse_count": result.llama_graph_reuse_count,
            "llama_eval_ms": result.llama_eval_ms,
            "llama_eval_tokens_per_second": result.llama_eval_tokens_per_second,
            "non_eval_overhead_ms": result.non_eval_overhead_ms,
            "decode_call_ms": result.decode_call_ms,
            "decode_call_tokens_per_second": result.decode_call_tokens_per_second,
            "sampling_ms": result.sampling_ms,
            "sampling_tokens_per_second": result.sampling_tokens_per_second,
            "logits_ready_ms": result.logits_ready_ms,
            "logits_scan_ms": result.logits_scan_ms,
            "graph_node_count": result.graph_node_count,
            "graph_inventory_bucket_overflow_count": result.graph_inventory_bucket_overflow_count,
            "graph_inventory": result.graph_inventory.iter().map(|bucket| json!({
                "family": bucket.family,
                "ggml_op": bucket.ggml_op,
                "ggml_type": bucket.ggml_type,
                "node_count": bucket.node_count,
                "element_count": bucket.element_count,
                "output_bytes": bucket.output_bytes,
                "src0_bytes": bucket.src0_bytes,
                "src1_bytes": bucket.src1_bytes,
                "ne": bucket.ne,
            })).collect::<Vec<_>>(),
        }))?
    );
    Ok(())
}
