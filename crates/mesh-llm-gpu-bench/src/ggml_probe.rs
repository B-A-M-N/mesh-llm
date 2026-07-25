use crate::{
    AttentionRuntimeProbeShape, BenchmarkBackend, DecodeKernelProbe, DenseFullTokenProbeShape,
    DenseGraphProbeShape, DenseSampledTokenProbeShape, LinearAttentionGraphProbeShape,
    LogitsReadbackProbeShape, MoeBlockGraphProbeShape, OutputProjectionProbeShape, ProbeDepth,
};
use anyhow::{Context, Result, anyhow};
use libc::{c_char, c_int, c_void};
use serde::Deserialize;
use std::ffi::CStr;

unsafe extern "C" {
    fn mesh_llm_gpu_bench_ggml_sampler_probe_json(
        vocab_tokens: i64,
        history_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_decode_probe_json(
        backend_kind: c_int,
        probe_depth: c_int,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        expert_count: i64,
        experts_used: i64,
        expert_width: i64,
        hidden: i64,
        repeat_layers: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        expert_count: i64,
        experts_used: i64,
        expert_width: i64,
        hidden: i64,
        kv_width: i64,
        repeat_layers: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_moe_block_decode_submission_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        expert_count: i64,
        experts_used: i64,
        expert_width: i64,
        hidden: i64,
        kv_width: i64,
        repeat_layers: i64,
        context_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_attention_runtime_probe_json(
        backend_kind: c_int,
        head_dim: i64,
        query_heads: i64,
        kv_heads: i64,
        context_tokens: i64,
        repeat_layers: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_logits_readback_probe_json(
        backend_kind: c_int,
        vocab: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_logits_sync_probe_json(
        backend_kind: c_int,
        vocab: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_logits_output_handoff_probe_json(
        backend_kind: c_int,
        vocab: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_sampled_token_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        vocab: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_full_token_probe_json(
        backend_kind: c_int,
        block_tensor_type_kind: c_int,
        output_tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        vocab: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        head_dim: i64,
        query_heads: i64,
        kv_heads: i64,
        context_tokens: i64,
        active_context_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_full_token_handoff_probe_json(
        backend_kind: c_int,
        block_tensor_type_kind: c_int,
        output_tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        vocab: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        head_dim: i64,
        query_heads: i64,
        kv_heads: i64,
        context_tokens: i64,
        active_context_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_decode_submission_probe_json(
        backend_kind: c_int,
        block_tensor_type_kind: c_int,
        output_tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        vocab: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        head_dim: i64,
        query_heads: i64,
        kv_heads: i64,
        context_tokens: i64,
        active_context_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_dense_source_sampled_token_probe_json(
        backend_kind: c_int,
        block_tensor_type_kind: c_int,
        output_tensor_type_kind: c_int,
        hidden: i64,
        kv_width: i64,
        ffn: i64,
        vocab: i64,
        repeat_layers: i64,
        graph_features: c_int,
        norm_head_width: i64,
        head_dim: i64,
        query_heads: i64,
        kv_heads: i64,
        context_tokens: i64,
        active_context_tokens: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        qkv_width: i64,
        gate_width: i64,
        state_width: i64,
        output_input_width: i64,
        ffn: i64,
        recurrent_layers: i64,
        full_attention_layers: i64,
        kv_width: i64,
        graph_features: c_int,
        norm_head_width: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_output_projection_probe_json(
        backend_kind: c_int,
        tensor_type_kind: c_int,
        hidden: i64,
        vocab: i64,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    fn mesh_llm_gpu_bench_ggml_decode_probe_free(ptr: *mut c_void);
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct NativeSamplerProbe {
    pub history_us_per_token: f64,
    pub vocab_us_per_token: f64,
}

pub fn run_sampler_probe(vocab_tokens: usize, history_tokens: usize) -> Result<NativeSamplerProbe> {
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_sampler_probe_json(
            i64::try_from(vocab_tokens).unwrap_or(i64::MAX),
            i64::try_from(history_tokens).unwrap_or(i64::MAX),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML source-shaped sampler probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML source-shaped sampler probe returned invalid output; prefix={preview}")
    })
}

pub fn run(backend: BenchmarkBackend, probe_depth: ProbeDepth) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let probe_depth = match probe_depth {
        ProbeDepth::HardwareOnly => return Ok(Vec::new()),
        ProbeDepth::Standard => 0,
        ProbeDepth::Deep => 1,
    };

    let mut error: *mut c_char = std::ptr::null_mut();
    let json =
        unsafe { mesh_llm_gpu_bench_ggml_decode_probe_json(backend_kind, probe_depth, &mut error) };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML decode probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML decode probe returned invalid output; prefix={preview}")
    })
}

pub fn run_moe_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    expert_count: u32,
    experts_used: u32,
    expert_width: u32,
    hidden: u32,
    repeat_layers: u32,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => 0,
        "q6_k" => 1,
        other => return Err(anyhow!("unsupported MoE graph probe tensor type {other}")),
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_moe_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(expert_count),
            i64::from(experts_used),
            i64::from(expert_width),
            i64::from(hidden),
            i64::from(repeat_layers.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML MoE graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML MoE graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_moe_block_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => 0,
        "q6_k" => 1,
        other => {
            return Err(anyhow!(
                "unsupported MoE block graph probe tensor type {other}"
            ));
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_moe_block_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.expert_count),
            i64::from(shape.experts_used),
            i64::from(shape.expert_width),
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML MoE block graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML MoE block graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_moe_block_decode_submission_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: MoeBlockGraphProbeShape,
    context_tokens: u32,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => 0,
        "q6_k" => 1,
        other => {
            return Err(anyhow!(
                "unsupported MoE block submission probe tensor type {other}"
            ));
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_moe_block_decode_submission_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.expert_count),
            i64::from(shape.experts_used),
            i64::from(shape.expert_width),
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            i64::from(context_tokens.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML MoE block submission probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML MoE block submission probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_attention_runtime_probe(
    backend: BenchmarkBackend,
    shape: AttentionRuntimeProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_attention_runtime_probe_json(
            backend_kind,
            i64::from(shape.head_dim),
            i64::from(shape.query_heads),
            i64::from(shape.kv_heads),
            i64::from(shape.context_tokens.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML attention runtime probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML attention runtime probe returned invalid output; prefix={preview}")
    })
}

pub fn run_logits_readback_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_logits_readback_probe_json(
            backend_kind,
            i64::from(shape.vocab.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML logits readback probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML logits readback probe returned invalid output; prefix={preview}")
    })
}

pub fn run_logits_sync_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_logits_sync_probe_json(
            backend_kind,
            i64::from(shape.vocab),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML logits sync probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML logits sync probe returned invalid output; prefix={preview}")
    })
}

pub fn run_logits_output_handoff_probe(
    backend: BenchmarkBackend,
    shape: LogitsReadbackProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_logits_output_handoff_probe_json(
            backend_kind,
            i64::from(shape.vocab),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML logits output handoff probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML logits output handoff probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_sampled_token_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: DenseSampledTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_sampled_token_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.vocab.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense sampled-token probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense sampled-token probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_full_token_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let block_tensor_type_kind = dense_graph_tensor_type_kind(block_tensor_type)?;
    let output_tensor_type_kind = dense_graph_tensor_type_kind(output_tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_full_token_probe_json(
            backend_kind,
            block_tensor_type_kind,
            output_tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.vocab.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            i64::from(shape.head_dim.max(1)),
            i64::from(shape.query_heads.max(1)),
            i64::from(shape.kv_heads.max(1)),
            i64::from(shape.context_tokens.max(1)),
            i64::from(shape.active_context_tokens.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense full-token probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense full-token probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_full_token_handoff_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let block_tensor_type_kind = dense_graph_tensor_type_kind(block_tensor_type)?;
    let output_tensor_type_kind = dense_graph_tensor_type_kind(output_tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_full_token_handoff_probe_json(
            backend_kind,
            block_tensor_type_kind,
            output_tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.vocab.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            i64::from(shape.head_dim.max(1)),
            i64::from(shape.query_heads.max(1)),
            i64::from(shape.kv_heads.max(1)),
            i64::from(shape.context_tokens.max(1)),
            i64::from(shape.active_context_tokens.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense full-token handoff probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense full-token handoff probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_decode_submission_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let block_tensor_type_kind = dense_graph_tensor_type_kind(block_tensor_type)?;
    let output_tensor_type_kind = dense_graph_tensor_type_kind(output_tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_decode_submission_probe_json(
            backend_kind,
            block_tensor_type_kind,
            output_tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.vocab.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            i64::from(shape.head_dim.max(1)),
            i64::from(shape.query_heads.max(1)),
            i64::from(shape.kv_heads.max(1)),
            i64::from(shape.context_tokens.max(1)),
            i64::from(shape.active_context_tokens.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense decode submission probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense decode submission probe returned invalid output; prefix={preview}")
    })
}

pub fn run_dense_source_sampled_token_probe(
    backend: BenchmarkBackend,
    block_tensor_type: &str,
    output_tensor_type: &str,
    shape: DenseFullTokenProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let block_tensor_type_kind = dense_graph_tensor_type_kind(block_tensor_type)?;
    let output_tensor_type_kind = dense_graph_tensor_type_kind(output_tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_dense_source_sampled_token_probe_json(
            backend_kind,
            block_tensor_type_kind,
            output_tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.kv_width.max(1)),
            i64::from(shape.ffn),
            i64::from(shape.vocab.max(1)),
            i64::from(shape.repeat_layers.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            i64::from(shape.head_dim.max(1)),
            i64::from(shape.query_heads.max(1)),
            i64::from(shape.kv_heads.max(1)),
            i64::from(shape.context_tokens.max(1)),
            i64::from(shape.active_context_tokens.max(1)),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML dense source sampled-token probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML dense source sampled-token probe returned invalid output; prefix={preview}")
    })
}

pub fn run_linear_attention_graph_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: LinearAttentionGraphProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_linear_attention_graph_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.qkv_width),
            i64::from(shape.gate_width),
            i64::from(shape.state_width),
            i64::from(shape.output_input_width),
            i64::from(shape.ffn),
            i64::from(shape.recurrent_layers.max(1)),
            i64::from(shape.full_attention_layers),
            i64::from(shape.kv_width.max(1)),
            c_int::try_from(shape.graph_features).unwrap_or(c_int::MAX),
            i64::from(shape.norm_head_width),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML linear attention graph probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML linear attention graph probe returned invalid output; prefix={preview}")
    })
}

pub fn run_output_projection_probe(
    backend: BenchmarkBackend,
    tensor_type: &str,
    shape: OutputProjectionProbeShape,
) -> Result<Vec<DecodeKernelProbe>> {
    let backend_kind = match backend {
        BenchmarkBackend::Metal => 0,
        BenchmarkBackend::Cuda => 1,
        BenchmarkBackend::Hip => 2,
        BenchmarkBackend::Intel => {
            return Ok(Vec::new());
        }
    };
    let tensor_type_kind = dense_graph_tensor_type_kind(tensor_type)?;
    let mut error: *mut c_char = std::ptr::null_mut();
    let json = unsafe {
        mesh_llm_gpu_bench_ggml_output_projection_probe_json(
            backend_kind,
            tensor_type_kind,
            i64::from(shape.hidden),
            i64::from(shape.vocab),
            &mut error,
        )
    };
    if json.is_null() {
        let message = if error.is_null() {
            "GGML output projection probe failed".to_string()
        } else {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(error.cast()) };
            message
        };
        return Err(anyhow!(message));
    }

    let bytes = unsafe { CStr::from_ptr(json) }.to_bytes().to_vec();
    unsafe { mesh_llm_gpu_bench_ggml_decode_probe_free(json.cast()) };
    serde_json::from_slice(&bytes).with_context(|| {
        let preview = String::from_utf8_lossy(&bytes);
        let preview = preview.chars().take(512).collect::<String>();
        format!("GGML output projection probe returned invalid output; prefix={preview}")
    })
}

fn dense_graph_tensor_type_kind(tensor_type: &str) -> Result<c_int> {
    match tensor_type.to_ascii_lowercase().as_str() {
        "q4_k" => Ok(0),
        "q6_k" => Ok(1),
        "q8_0" => Ok(2),
        "f16" => Ok(3),
        "q5_k" => Ok(4),
        other => Err(anyhow!("unsupported dense graph probe tensor type {other}")),
    }
}
