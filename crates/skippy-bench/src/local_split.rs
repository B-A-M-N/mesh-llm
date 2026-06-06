use std::{
    fs,
    process::{Command, Stdio},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use model_artifact::ModelIdentity;
use serde_json::json;
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    activation_state_flags_from_frame_flags, recv_reply, write_stage_message,
};
use skippy_runtime::{RuntimeConfig, RuntimeLoadMode, StageModel};
use skippy_topology::{
    BoundaryDecision, NodeSpec, PlannerPolicy, TopologyPlanRequest, WireValidation,
    dense_attention_layers, infer_family_capability, plan_contiguous_with_splits,
};

use crate::{
    cli::{
        LocalPrefillCompressionArgs, LocalSplitBinaryArgs, LocalSplitChainBinaryArgs,
        LocalSplitCompareArgs, LocalSplitInprocessArgs,
    },
    model_identity::model_identity_for_path,
    support::{
        ChildGuard, activation_width, connect_ready, generate_run_id, parse_wire_dtype,
        temp_config_path_for,
    },
};

struct BinarySplitResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    boundary_producer_stage_index: i32,
    boundary_layer_start: i32,
    boundary_layer_end: i32,
    boundary_token_count: u32,
    boundary_payload_bytes: u64,
    boundary_wire_payload_bytes: usize,
}

struct BinaryChainResult {
    model_identity: ModelIdentity,
    token_id: i32,
    predicted_token: i32,
    activation_width: i32,
    wire_dtype: String,
    stage0_wire_payload_bytes: usize,
    stage0_payload_bytes: u64,
    split_layer_1: u32,
    split_layer_2: u32,
    layer_end: u32,
}

struct CompressionTiming {
    compressed_bytes: usize,
    compress_ms: f64,
    decompress_ms: f64,
}

struct WireRoundTripTiming {
    wire_bytes: usize,
    encode_ms: f64,
    decode_ms: f64,
    error: ActivationError,
}

struct TransformCompressionTiming {
    compressed_bytes: usize,
    encode_ms: f64,
    decode_ms: f64,
}

#[derive(Clone, Copy)]
struct ActivationError {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
}

#[derive(Clone, Copy)]
enum LosslessTransform {
    ByteShuffle,
    XorWordDelta,
    XorTokenDelta,
    XorWordDeltaByteShuffle,
    XorTokenDeltaByteShuffle,
}

impl LosslessTransform {
    fn name(self) -> &'static str {
        match self {
            Self::ByteShuffle => "byte_shuffle_u32_lz4",
            Self::XorWordDelta => "xor_word_delta_lz4",
            Self::XorTokenDelta => "xor_token_delta_lz4",
            Self::XorWordDeltaByteShuffle => "xor_word_delta_byte_shuffle_lz4",
            Self::XorTokenDeltaByteShuffle => "xor_token_delta_byte_shuffle_lz4",
        }
    }
}

pub fn local_split_binary(args: LocalSplitBinaryArgs) -> Result<()> {
    let result = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        activation_wire_dtype: args.activation_wire_dtype,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "wire_dtype": result.wire_dtype,
            "boundary": {
                "producer_stage_index": result.boundary_producer_stage_index,
                "layer_start": result.boundary_layer_start,
                "layer_end": result.boundary_layer_end,
                "token_count": result.boundary_token_count,
                "payload_bytes": result.boundary_payload_bytes,
                "wire_payload_bytes": result.boundary_wire_payload_bytes,
            }
        }))?
    );

    Ok(())
}

pub fn local_split_compare(args: LocalSplitCompareArgs) -> Result<()> {
    let baseline = run_full_model_decode(
        &args.model_path,
        args.layer_end,
        args.ctx_size,
        args.n_gpu_layers,
        &args.prompt,
    )?;
    let split = run_binary_split(BinarySplitConfig {
        stage_server_bin: args.stage_server_bin,
        model_path: args.model_path,
        model_id: args.model_id,
        split_layer: args.split_layer,
        layer_end: args.layer_end,
        ctx_size: args.ctx_size,
        n_gpu_layers: args.n_gpu_layers,
        prompt: args.prompt,
        stage1_bind_addr: args.stage1_bind_addr,
        activation_wire_dtype: args.activation_wire_dtype,
        child_logs: args.child_logs,
        startup_timeout_secs: args.startup_timeout_secs,
    })?;

    let matches = baseline.predicted_token == split.predicted_token;
    let output = json!({
        "mode": "local-split-compare",
        "model_identity": split.model_identity,
        "matches": matches,
        "baseline": {
            "token_id": baseline.token_id,
            "predicted_token": baseline.predicted_token,
        },
        "split": {
            "token_id": split.token_id,
            "predicted_token": split.predicted_token,
            "activation_width": split.activation_width,
            "wire_dtype": split.wire_dtype,
            "boundary": {
                "producer_stage_index": split.boundary_producer_stage_index,
                "layer_start": split.boundary_layer_start,
                "layer_end": split.boundary_layer_end,
                "token_count": split.boundary_token_count,
                "payload_bytes": split.boundary_payload_bytes,
                "wire_payload_bytes": split.boundary_wire_payload_bytes,
            }
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);

    if !matches && !args.allow_mismatch {
        bail!(
            "split predicted token {} did not match full-model predicted token {}",
            split.predicted_token,
            baseline.predicted_token
        );
    }

    Ok(())
}

pub fn local_prefill_compression(args: LocalPrefillCompressionArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    if args.prefill_tokens == 0 {
        bail!("prefill_tokens must be greater than zero");
    }
    if args.iterations == 0 {
        bail!("iterations must be greater than zero");
    }

    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let mut tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    if tokens.is_empty() {
        bail!("prompt produced no tokens");
    }
    tokens.truncate(args.prefill_tokens);

    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let boundary = session0
        .prefill_chunk_frame(&tokens, None, 0)
        .context("stage 0 failed to produce prefill activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty prefill activation frame");
    }
    let activation_width = activation_width(&boundary)?;
    if let Some(payload_out) = args.payload_out.as_ref() {
        fs::write(payload_out, &boundary.payload)
            .with_context(|| format!("failed to write {}", payload_out.display()))?;
    }

    let timings = measure_lz4_round_trip(&boundary.payload, args.iterations)?;
    let f16_timings = measure_wire_round_trip(
        WireActivationDType::F16,
        &boundary.payload,
        tokens.len(),
        activation_width,
        boundary.desc.flags,
        args.iterations,
    )?;
    let q8_timings = measure_wire_round_trip(
        WireActivationDType::Q8,
        &boundary.payload,
        tokens.len(),
        activation_width,
        boundary.desc.flags,
        args.iterations,
    )?;
    let transform_timings = measure_lossless_transforms(
        &boundary.payload,
        tokens.len(),
        activation_width,
        args.iterations,
    )?;
    let best_compressed_bytes = timings
        .iter()
        .map(|timing| timing.compressed_bytes)
        .min()
        .context("compression timing set is empty")?;
    let raw_activation_bytes = boundary.payload.len();
    let best_ratio = best_compressed_bytes as f64 / raw_activation_bytes as f64;
    let best_bytes_saved = raw_activation_bytes.saturating_sub(best_compressed_bytes);
    let mean_compress_ms = mean_ms(timings.iter().map(|timing| timing.compress_ms));
    let mean_decompress_ms = mean_ms(timings.iter().map(|timing| timing.decompress_ms));

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-prefill-compression",
            "codec": "lz4_flex_block_prepend_size",
            "model_identity": model_identity,
            "layer_range": {
                "start": 0,
                "end": args.split_layer,
                "model_layer_end": args.layer_end,
            },
            "prefill_tokens": tokens.len(),
            "requested_prefill_tokens": args.prefill_tokens,
            "activation_width": activation_width,
            "raw_activation_bytes": raw_activation_bytes,
            "best_compressed_bytes": best_compressed_bytes,
            "best_ratio": best_ratio,
            "best_bytes_saved": best_bytes_saved,
            "mean_compress_ms": mean_compress_ms,
            "mean_decompress_ms": mean_decompress_ms,
            "mean_round_trip_ms": mean_compress_ms + mean_decompress_ms,
            "wire_round_trips": [
                summarize_wire_round_trip("f16", raw_activation_bytes, &f16_timings),
                summarize_wire_round_trip("q8", raw_activation_bytes, &q8_timings),
            ],
            "lossless_transforms": transform_timings,
            "iterations": args.iterations,
        }))?
    );

    Ok(())
}

fn measure_lz4_round_trip(payload: &[u8], iterations: usize) -> Result<Vec<CompressionTiming>> {
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let compress_started = Instant::now();
        let compressed = lz4_flex::compress_prepend_size(payload);
        let compress_ms = compress_started.elapsed().as_secs_f64() * 1000.0;

        let decompress_started = Instant::now();
        let decompressed = lz4_flex::decompress_size_prepended(&compressed)
            .context("failed to decompress lz4 prefill activation payload")?;
        let decompress_ms = decompress_started.elapsed().as_secs_f64() * 1000.0;

        if decompressed != payload {
            bail!("lz4 prefill activation round trip changed payload bytes");
        }

        timings.push(CompressionTiming {
            compressed_bytes: compressed.len(),
            compress_ms,
            decompress_ms,
        });
    }
    Ok(timings)
}

fn measure_lossless_transforms(
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
    iterations: usize,
) -> Result<Vec<serde_json::Value>> {
    let transforms = [
        LosslessTransform::ByteShuffle,
        LosslessTransform::XorWordDelta,
        LosslessTransform::XorTokenDelta,
        LosslessTransform::XorWordDeltaByteShuffle,
        LosslessTransform::XorTokenDeltaByteShuffle,
    ];
    transforms
        .into_iter()
        .map(|transform| {
            let timings = measure_transformed_lz4(
                transform,
                payload,
                token_count,
                activation_width,
                iterations,
            )?;
            Ok(summarize_transform_compression(
                transform,
                payload.len(),
                &timings,
            ))
        })
        .collect()
}

fn measure_transformed_lz4(
    transform: LosslessTransform,
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
    iterations: usize,
) -> Result<Vec<TransformCompressionTiming>> {
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let encode_started = Instant::now();
        let transformed =
            apply_lossless_transform(transform, payload, token_count, activation_width)?;
        let compressed = lz4_flex::compress_prepend_size(&transformed);
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let decode_started = Instant::now();
        let decompressed = lz4_flex::decompress_size_prepended(&compressed)
            .context("failed to decompress transformed activation payload")?;
        let restored =
            invert_lossless_transform(transform, &decompressed, token_count, activation_width)?;
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;

        if restored != payload {
            bail!(
                "{} transformed activation round trip changed payload bytes",
                transform.name()
            );
        }

        timings.push(TransformCompressionTiming {
            compressed_bytes: compressed.len(),
            encode_ms,
            decode_ms,
        });
    }
    Ok(timings)
}

fn summarize_transform_compression(
    transform: LosslessTransform,
    raw_activation_bytes: usize,
    timings: &[TransformCompressionTiming],
) -> serde_json::Value {
    let best_compressed_bytes = timings
        .iter()
        .map(|timing| timing.compressed_bytes)
        .min()
        .unwrap_or_default();
    json!({
        "codec": transform.name(),
        "best_compressed_bytes": best_compressed_bytes,
        "best_ratio": if raw_activation_bytes == 0 {
            0.0
        } else {
            best_compressed_bytes as f64 / raw_activation_bytes as f64
        },
        "best_bytes_saved": raw_activation_bytes.saturating_sub(best_compressed_bytes),
        "mean_encode_ms": mean_ms(timings.iter().map(|timing| timing.encode_ms)),
        "mean_decode_ms": mean_ms(timings.iter().map(|timing| timing.decode_ms)),
        "mean_round_trip_ms": mean_ms(timings.iter().map(|timing| {
            timing.encode_ms + timing.decode_ms
        })),
    })
}

fn apply_lossless_transform(
    transform: LosslessTransform,
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
) -> Result<Vec<u8>> {
    match transform {
        LosslessTransform::ByteShuffle => byte_shuffle_u32(payload),
        LosslessTransform::XorWordDelta => xor_word_delta(payload),
        LosslessTransform::XorTokenDelta => xor_token_delta(payload, token_count, activation_width),
        LosslessTransform::XorWordDeltaByteShuffle => byte_shuffle_u32(&xor_word_delta(payload)?),
        LosslessTransform::XorTokenDeltaByteShuffle => {
            byte_shuffle_u32(&xor_token_delta(payload, token_count, activation_width)?)
        }
    }
}

fn invert_lossless_transform(
    transform: LosslessTransform,
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
) -> Result<Vec<u8>> {
    match transform {
        LosslessTransform::ByteShuffle => byte_unshuffle_u32(payload),
        LosslessTransform::XorWordDelta => inverse_xor_word_delta(payload),
        LosslessTransform::XorTokenDelta => {
            inverse_xor_token_delta(payload, token_count, activation_width)
        }
        LosslessTransform::XorWordDeltaByteShuffle => {
            inverse_xor_word_delta(&byte_unshuffle_u32(payload)?)
        }
        LosslessTransform::XorTokenDeltaByteShuffle => {
            inverse_xor_token_delta(&byte_unshuffle_u32(payload)?, token_count, activation_width)
        }
    }
}

fn byte_shuffle_u32(payload: &[u8]) -> Result<Vec<u8>> {
    ensure_f32_payload(payload)?;
    let word_count = payload.len() / 4;
    let mut out = vec![0_u8; payload.len()];
    for (word_index, word) in payload.chunks_exact(4).enumerate() {
        for byte_index in 0..4 {
            out[byte_index * word_count + word_index] = word[byte_index];
        }
    }
    Ok(out)
}

fn byte_unshuffle_u32(payload: &[u8]) -> Result<Vec<u8>> {
    ensure_f32_payload(payload)?;
    let word_count = payload.len() / 4;
    let mut out = vec![0_u8; payload.len()];
    for word_index in 0..word_count {
        for byte_index in 0..4 {
            out[word_index * 4 + byte_index] = payload[byte_index * word_count + word_index];
        }
    }
    Ok(out)
}

fn xor_word_delta(payload: &[u8]) -> Result<Vec<u8>> {
    ensure_f32_payload(payload)?;
    let mut out = Vec::with_capacity(payload.len());
    let mut previous = 0_u32;
    for word in payload.chunks_exact(4) {
        let value = u32::from_le_bytes(word.try_into().expect("chunks_exact returns 4 bytes"));
        out.extend_from_slice(&(value ^ previous).to_le_bytes());
        previous = value;
    }
    Ok(out)
}

fn inverse_xor_word_delta(payload: &[u8]) -> Result<Vec<u8>> {
    ensure_f32_payload(payload)?;
    let mut out = Vec::with_capacity(payload.len());
    let mut previous = 0_u32;
    for word in payload.chunks_exact(4) {
        let delta = u32::from_le_bytes(word.try_into().expect("chunks_exact returns 4 bytes"));
        let value = delta ^ previous;
        out.extend_from_slice(&value.to_le_bytes());
        previous = value;
    }
    Ok(out)
}

fn xor_token_delta(payload: &[u8], token_count: usize, activation_width: i32) -> Result<Vec<u8>> {
    let layout = activation_layout(payload, token_count, activation_width)?;
    let mut out = Vec::with_capacity(payload.len());
    for row_index in 0..layout.row_count {
        for column_index in 0..layout.width {
            let value = f32_word_at(payload, layout.width, row_index, column_index);
            let previous = if row_index == 0 {
                0
            } else {
                f32_word_at(payload, layout.width, row_index - 1, column_index)
            };
            out.extend_from_slice(&(value ^ previous).to_le_bytes());
        }
    }
    Ok(out)
}

fn inverse_xor_token_delta(
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
) -> Result<Vec<u8>> {
    let layout = activation_layout(payload, token_count, activation_width)?;
    let mut restored = vec![0_u32; layout.row_count * layout.width];
    for row_index in 0..layout.row_count {
        for column_index in 0..layout.width {
            let delta = f32_word_at(payload, layout.width, row_index, column_index);
            let previous = if row_index == 0 {
                0
            } else {
                restored[(row_index - 1) * layout.width + column_index]
            };
            restored[row_index * layout.width + column_index] = delta ^ previous;
        }
    }
    let mut out = Vec::with_capacity(payload.len());
    for word in restored {
        out.extend_from_slice(&word.to_le_bytes());
    }
    Ok(out)
}

struct ActivationLayout {
    row_count: usize,
    width: usize,
}

fn activation_layout(
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
) -> Result<ActivationLayout> {
    ensure_f32_payload(payload)?;
    if activation_width <= 0 {
        bail!("activation_width must be positive");
    }
    let width = activation_width as usize;
    let elements = payload.len() / 4;
    if !elements.is_multiple_of(width) {
        bail!("activation payload is not a whole number of activation rows");
    }
    let row_count = elements / width;
    if row_count < token_count {
        bail!("activation payload row count is smaller than token count");
    }
    Ok(ActivationLayout { row_count, width })
}

fn f32_word_at(payload: &[u8], width: usize, row_index: usize, column_index: usize) -> u32 {
    let offset = (row_index * width + column_index) * 4;
    u32::from_le_bytes(
        payload[offset..offset + 4]
            .try_into()
            .expect("slice has 4 bytes"),
    )
}

fn ensure_f32_payload(payload: &[u8]) -> Result<()> {
    if payload.len() & 3 != 0 {
        bail!("activation payload is not f32 aligned");
    }
    Ok(())
}

fn measure_wire_round_trip(
    dtype: WireActivationDType,
    payload: &[u8],
    token_count: usize,
    activation_width: i32,
    frame_flags: u64,
    iterations: usize,
) -> Result<Vec<WireRoundTripTiming>> {
    let token_count_i32 = i32::try_from(token_count).context("token count exceeds i32")?;
    let state_flags = activation_state_flags_from_frame_flags(frame_flags);
    let mut timings = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let encode_started = Instant::now();
        let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
            dtype,
            token_count_i32,
            activation_width,
            payload,
            state_flags,
        )
        .context("failed to encode activation wire payload")?;
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let mut state = StageStateHeader::new(WireMessageKind::PrefillEmbd, dtype);
        state.flags = state_flags;
        let message = StageWireMessage {
            kind: WireMessageKind::PrefillEmbd,
            pos_start: 0,
            token_count: token_count_i32,
            state,
            request_id: 0,
            session_id: 0,
            sampling: None,
            chat_sampling_metadata: None,
            tokens: Vec::new(),
            positions: Vec::new(),
            activation,
            raw_bytes: Vec::new(),
        };

        let decode_started = Instant::now();
        let decoded = message
            .activation_f32_payload(activation_width)
            .context("failed to decode activation wire payload")?;
        let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
        let error = activation_error(payload, &decoded)?;

        timings.push(WireRoundTripTiming {
            wire_bytes: message.activation.len(),
            encode_ms,
            decode_ms,
            error,
        });
    }
    Ok(timings)
}

fn summarize_wire_round_trip(
    dtype: &str,
    raw_activation_bytes: usize,
    timings: &[WireRoundTripTiming],
) -> serde_json::Value {
    let best_wire_bytes = timings
        .iter()
        .map(|timing| timing.wire_bytes)
        .min()
        .unwrap_or_default();
    let first_error = timings
        .first()
        .map(|timing| timing.error)
        .unwrap_or(ActivationError {
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
        });
    json!({
        "dtype": dtype,
        "best_wire_bytes": best_wire_bytes,
        "best_ratio": if raw_activation_bytes == 0 {
            0.0
        } else {
            best_wire_bytes as f64 / raw_activation_bytes as f64
        },
        "best_bytes_saved": raw_activation_bytes.saturating_sub(best_wire_bytes),
        "mean_encode_ms": mean_ms(timings.iter().map(|timing| timing.encode_ms)),
        "mean_decode_ms": mean_ms(timings.iter().map(|timing| timing.decode_ms)),
        "mean_round_trip_ms": mean_ms(timings.iter().map(|timing| {
            timing.encode_ms + timing.decode_ms
        })),
        "error": {
            "max_abs": first_error.max_abs,
            "mean_abs": first_error.mean_abs,
            "rmse": first_error.rmse,
        }
    })
}

fn activation_error(original: &[u8], decoded: &[u8]) -> Result<ActivationError> {
    if original.len() != decoded.len() || original.len() & 3 != 0 {
        bail!("activation error inputs have incompatible f32 byte lengths");
    }

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_squared = 0.0_f64;
    let mut count = 0usize;

    for (original_chunk, decoded_chunk) in original.chunks_exact(4).zip(decoded.chunks_exact(4)) {
        let original_value = f32::from_le_bytes(
            original_chunk
                .try_into()
                .expect("chunks_exact returns 4 bytes"),
        );
        let decoded_value = f32::from_le_bytes(
            decoded_chunk
                .try_into()
                .expect("chunks_exact returns 4 bytes"),
        );
        let abs_error = (original_value - decoded_value).abs() as f64;
        max_abs = max_abs.max(abs_error);
        sum_abs += abs_error;
        sum_squared += abs_error * abs_error;
        count += 1;
    }

    if count == 0 {
        return Ok(ActivationError {
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
        });
    }

    Ok(ActivationError {
        max_abs,
        mean_abs: sum_abs / count as f64,
        rmse: (sum_squared / count as f64).sqrt(),
    })
}

fn mean_ms(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0usize;
    let mut total = 0.0;
    for value in values {
        count += 1;
        total += value;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

pub fn local_split_chain_binary(args: LocalSplitChainBinaryArgs) -> Result<()> {
    let result = run_binary_chain(args)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-chain-binary",
            "model_identity": result.model_identity,
            "token_id": result.token_id,
            "predicted_token": result.predicted_token,
            "activation_width": result.activation_width,
            "wire_dtype": result.wire_dtype,
            "stages": [
                {
                    "stage_index": 0,
                    "layer_start": 0,
                    "layer_end": result.split_layer_1,
                    "payload_bytes": result.stage0_payload_bytes,
                    "wire_payload_bytes": result.stage0_wire_payload_bytes,
                },
                {
                    "stage_index": 1,
                    "layer_start": result.split_layer_1,
                    "layer_end": result.split_layer_2,
                    "forwarded_over_binary": true,
                },
                {
                    "stage_index": 2,
                    "layer_start": result.split_layer_2,
                    "layer_end": result.layer_end,
                    "returned_predicted_token": true,
                }
            ]
        }))?
    );
    Ok(())
}

struct FullModelResult {
    token_id: i32,
    predicted_token: i32,
}

fn run_full_model_decode(
    model_path: &std::path::Path,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: &str,
) -> Result<FullModelResult> {
    let config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end,
        ctx_size,
        lane_count: 1,
        n_batch: None,
        n_ubatch: None,
        n_threads: None,
        n_threads_batch: None,
        n_gpu_layers,
        selected_backend_device: None,
        cache_type_k: skippy_runtime::GGML_TYPE_F16,
        cache_type_v: skippy_runtime::GGML_TYPE_F16,
        flash_attn_type: skippy_runtime::FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        projector_path: None,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: false,
    };
    let model = StageModel::open(model_path, &config).context("failed to open full model")?;
    let tokens = model
        .tokenize(prompt, true)
        .context("failed to tokenize prompt with full model")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session = model
        .create_session()
        .context("failed to create full-model session")?;
    let predicted_token = session
        .decode_step_frame(token_id, None, 0)
        .context("full model failed to decode")?
        .0;
    Ok(FullModelResult {
        token_id,
        predicted_token,
    })
}

struct BinarySplitConfig {
    stage_server_bin: std::path::PathBuf,
    model_path: std::path::PathBuf,
    model_id: String,
    split_layer: u32,
    layer_end: u32,
    ctx_size: u32,
    n_gpu_layers: i32,
    prompt: String,
    stage1_bind_addr: std::net::SocketAddr,
    activation_wire_dtype: String,
    child_logs: bool,
    startup_timeout_secs: u64,
}

fn run_binary_split(args: BinarySplitConfig) -> Result<BinarySplitResult> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer],
        2,
        &args.activation_wire_dtype,
    )?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let config_path = temp_config_path_for(&run_id, "stage-1");
    let config = json!({
        "run_id": run_id,
        "topology_id": "local-split-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": null
    });
    fs::write(&config_path, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", config_path.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve-binary",
        "--config",
        config_path
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    if args.child_logs {
        stage_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        stage_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _stage1 = ChildGuard::spawn(stage_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 1,
        session_id: 1,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary decode")?;
    let reply = recv_reply(&mut stream).context("receive binary reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary stop")?;

    Ok(BinarySplitResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        boundary_producer_stage_index: boundary.desc.producer_stage_index,
        boundary_layer_start: boundary.desc.layer_start,
        boundary_layer_end: boundary.desc.layer_end,
        boundary_token_count: boundary.desc.token_count,
        boundary_payload_bytes: boundary.desc.payload_bytes,
        boundary_wire_payload_bytes: message.activation.len(),
    })
}

fn run_binary_chain(args: LocalSplitChainBinaryArgs) -> Result<BinaryChainResult> {
    if args.split_layer_1 == 0
        || args.split_layer_1 >= args.split_layer_2
        || args.split_layer_2 >= args.layer_end
    {
        bail!("split_layer_1 and split_layer_2 must partition 0..layer_end in ascending order");
    }
    validate_local_topology_plan(
        &args.model_path,
        args.layer_end,
        &[args.split_layer_1, args.split_layer_2],
        3,
        &args.activation_wire_dtype,
    )?;
    let wire_dtype = parse_wire_dtype(&args.activation_wire_dtype)?;
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;
    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer_1,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;
    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }
    let activation_width = activation_width(&boundary)?;

    let run_id = generate_run_id();
    let stage1_config_path = temp_config_path_for(&run_id, "stage-1");
    let stage2_config_path = temp_config_path_for(&run_id, "stage-2");
    let stage2_config = json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-2",
        "stage_index": 2,
        "layer_start": args.split_layer_2,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage2_bind_addr,
        "upstream": {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": format!("tcp://{}", args.stage1_bind_addr)
        },
        "downstream": null
    });
    let stage1_config = json!({
        "run_id": run_id,
        "topology_id": "local-split-chain-binary",
        "model_id": model_identity.model_id,
        "model_path": args.model_path,
        "stage_id": "stage-1",
        "stage_index": 1,
        "layer_start": args.split_layer_1,
        "layer_end": args.split_layer_2,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "filter_tensors_on_load": true,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage1_bind_addr,
        "upstream": {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": "driver"
        },
        "downstream": {
            "stage_id": "stage-2",
            "stage_index": 2,
            "endpoint": format!("tcp://{}", args.stage2_bind_addr)
        }
    });
    fs::write(
        &stage2_config_path,
        serde_json::to_vec_pretty(&stage2_config)?,
    )
    .with_context(|| format!("failed to write {}", stage2_config_path.display()))?;
    fs::write(
        &stage1_config_path,
        serde_json::to_vec_pretty(&stage1_config)?,
    )
    .with_context(|| format!("failed to write {}", stage1_config_path.display()))?;

    let mut stage2_command = Command::new(&args.stage_server_bin);
    stage2_command.args([
        "serve-binary",
        "--config",
        stage2_config_path
            .to_str()
            .context("stage 2 config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage2_command, args.child_logs);
    let _stage2 = ChildGuard::spawn(stage2_command)?;

    let mut stage1_command = Command::new(&args.stage_server_bin);
    stage1_command.args([
        "serve-binary",
        "--config",
        stage1_config_path
            .to_str()
            .context("stage 1 config path is not valid UTF-8")?,
        "--activation-width",
        &activation_width.to_string(),
        "--activation-wire-dtype",
        &args.activation_wire_dtype,
    ]);
    configure_child_logs(&mut stage1_command, args.child_logs);
    let _stage1 = ChildGuard::spawn(stage1_command)?;

    let mut stream = connect_ready(args.stage1_bind_addr, args.startup_timeout_secs)
        .context("stage 1 binary server did not become ready")?;
    let mut state = StageStateHeader::new(WireMessageKind::DecodeEmbd, wire_dtype);
    state.prompt_token_count = 0;
    state.decode_step = 0;
    state.current_token = token_id;
    state.source_stage_index = 0;
    state.flags |=
        skippy_protocol::binary::activation_state_flags_from_frame_flags(boundary.desc.flags);
    let activation = skippy_protocol::binary::encode_f32_activation_payload_with_state_flags(
        wire_dtype,
        1,
        activation_width,
        &boundary.payload,
        state.flags,
    )
    .context("failed to encode boundary activation for wire")?;
    let message = StageWireMessage {
        kind: WireMessageKind::DecodeEmbd,
        pos_start: 0,
        token_count: 1,
        state,
        request_id: 2,
        session_id: 2,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: vec![token_id],
        positions: vec![0],
        activation,
        raw_bytes: Vec::new(),
    };
    write_stage_message(&mut stream, &message, wire_dtype).context("send binary chain decode")?;
    let reply = recv_reply(&mut stream).context("receive binary chain reply")?;
    if reply.kind != WireReplyKind::PredictedToken {
        bail!("expected predicted-token reply, got {:?}", reply.kind);
    }
    write_stage_message(&mut stream, &StageWireMessage::stop(wire_dtype), wire_dtype)
        .context("send binary chain stop")?;

    Ok(BinaryChainResult {
        model_identity,
        token_id,
        predicted_token: reply.predicted,
        activation_width,
        wire_dtype: args.activation_wire_dtype,
        stage0_wire_payload_bytes: message.activation.len(),
        stage0_payload_bytes: boundary.desc.payload_bytes,
        split_layer_1: args.split_layer_1,
        split_layer_2: args.split_layer_2,
        layer_end: args.layer_end,
    })
}

fn validate_local_topology_plan(
    model_path: &std::path::Path,
    layer_end: u32,
    splits: &[u32],
    stage_count: usize,
    activation_wire_dtype: &str,
) -> Result<()> {
    let identity = model_path.display().to_string();
    let family = infer_family_capability(&identity, layer_end, 0);
    let request = TopologyPlanRequest {
        topology_id: "local-split-binary".to_string(),
        model_id: identity,
        layers: dense_attention_layers(layer_end, 0),
        nodes: (0..stage_count)
            .map(|index| NodeSpec {
                node_id: format!("local-stage-{index}"),
                cached_slice_bytes: 0,
                vram_bytes: 0,
            })
            .collect(),
        family: family.clone(),
        policy: PlannerPolicy::default(),
    };
    let plan = plan_contiguous_with_splits(&request, splits).context("topology planner failed")?;

    if activation_wire_dtype.eq_ignore_ascii_case("q8") {
        match family.as_ref().map(|family| family.q8_wire_validation) {
            Some(WireValidation::Validated) => {}
            Some(WireValidation::Rejected) => {
                bail!(
                    "topology planner rejected q8 activation wire dtype for {}; use f16 or add a passing q8 correctness record",
                    model_path.display()
                );
            }
            Some(WireValidation::Untested) => {
                bail!(
                    "topology planner has no q8 validation for {}; use f16 until this family/split passes correctness",
                    model_path.display()
                );
            }
            None => {}
        }
    }

    let rejected = plan
        .boundaries
        .iter()
        .filter(|boundary| boundary.decision == BoundaryDecision::Rejected)
        .collect::<Vec<_>>();
    if !rejected.is_empty() {
        let reasons = rejected
            .iter()
            .map(|boundary| {
                format!(
                    "layer {}: {:?}: {}",
                    boundary.layer_boundary,
                    boundary.reason_codes,
                    boundary.messages.join("; ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        bail!("topology planner rejected split plan:\n{reasons}");
    }

    Ok(())
}

fn configure_child_logs(command: &mut Command, child_logs: bool) {
    if child_logs {
        command.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
}

pub fn local_split_inprocess(args: LocalSplitInprocessArgs) -> Result<()> {
    if args.split_layer == 0 || args.split_layer >= args.layer_end {
        bail!("split_layer must be greater than zero and less than layer_end");
    }

    let stage0_config = RuntimeConfig {
        stage_index: 0,
        layer_start: 0,
        layer_end: args.split_layer,
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
        include_output: false,
        filter_tensors_on_load: true,
    };
    let stage1_config = RuntimeConfig {
        stage_index: 1,
        layer_start: args.split_layer,
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
        include_embeddings: false,
        include_output: true,
        filter_tensors_on_load: true,
    };

    let stage0 =
        StageModel::open(&args.model_path, &stage0_config).context("failed to open stage 0")?;
    let stage1 =
        StageModel::open(&args.model_path, &stage1_config).context("failed to open stage 1")?;
    let tokens = stage0
        .tokenize(&args.prompt, true)
        .context("failed to tokenize prompt")?;
    let token_id = *tokens.first().context("prompt produced no tokens")?;

    let mut session0 = stage0
        .create_session()
        .context("failed to create stage 0 session")?;
    let mut session1 = stage1
        .create_session()
        .context("failed to create stage 1 session")?;

    let (_boundary_prediction, boundary) = session0
        .decode_step_frame(token_id, None, 0)
        .context("stage 0 failed to produce activation frame")?;
    if boundary.payload.is_empty() {
        bail!("stage 0 produced an empty activation frame");
    }

    let (predicted_token, final_frame) = session1
        .decode_step_frame(token_id, Some(&boundary), 0)
        .context("stage 1 failed to consume activation frame")?;
    if !final_frame.payload.is_empty() {
        bail!("final stage unexpectedly produced an activation payload");
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "mode": "local-split-inprocess",
            "token_id": token_id,
            "predicted_token": predicted_token,
            "boundary": {
                "producer_stage_index": boundary.desc.producer_stage_index,
                "layer_start": boundary.desc.layer_start,
                "layer_end": boundary.desc.layer_end,
                "token_count": boundary.desc.token_count,
                "sequence_count": boundary.desc.sequence_count,
                "payload_bytes": boundary.desc.payload_bytes,
                "actual_payload_bytes": boundary.payload.len(),
            },
            "final": {
                "producer_stage_index": final_frame.desc.producer_stage_index,
                "layer_start": final_frame.desc.layer_start,
                "layer_end": final_frame.desc.layer_end,
                "payload_bytes": final_frame.desc.payload_bytes,
            }
        }))?
    );

    Ok(())
}
