//! Stateless Nemotron-H MoE blocks behind the shared stage-engine contract.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use safemlx::{
    Array, Device, DeviceType, Dtype, Stream,
    memory::{active_memory, cache_memory, peak_memory, reset_peak_memory},
    module::{Module, ModuleParameters, ModuleParametersExt},
};
use safemlx_lm::{
    models::nemotron_h::{BlockInput, LayerBlockType, TransformerBlock, get_nemotron_h_model_args},
    weights::{StrictLoadConfig, StrictLoadReport, load_safetensors_dir_strict},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_engine::{
    StageActivation, StageEngine, StageEngineInfo, StageExecutionKind, StageExecutionOutput,
    StageExecutionRequest,
};

use crate::derived::{nemotron_h_validation_values, validate_nemotron_h_moe_stage_output};

use super::{MlxComputeDtype, MlxStageEngine, MlxStageEngineConfig, array_activation};

/// Tolerance-aware direct-block versus shared-stage-contract evidence.
#[derive(Clone, Debug, Serialize)]
pub struct MlxNemotronHStageValidationReport {
    pub model_dir: PathBuf,
    pub layer: usize,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_is_finite: bool,
    pub direct_output_f32_sha256: String,
    pub stage_output_f32_sha256: String,
    pub output_within_tolerance: bool,
    pub cross_session_stable: bool,
    pub session_reset_stable: bool,
    pub executions_compared: usize,
    pub max_abs_diff: f32,
    pub max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub cross_session_max_abs_diff: f32,
    pub cross_session_max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub reset_max_abs_diff: f32,
    pub reset_max_relative_diff_for_reference_magnitude_above_atol: f32,
    pub comparison_atol: f32,
    pub comparison_rtol: f32,
    pub mlx_active_memory_bytes: usize,
    pub mlx_cache_memory_bytes: usize,
    pub mlx_peak_memory_bytes: usize,
}

/// Compares direct safemlx execution with execution through `StageEngine`.
pub fn validate_nemotron_h_stage_engine(
    model_dir: impl AsRef<Path>,
    layer: usize,
) -> Result<MlxNemotronHStageValidationReport> {
    let model_dir = model_dir.as_ref();
    const ATOL: f32 = 1.0e-4;
    const RTOL: f32 = 1.0e-4;
    let (direct, direct_values) = validate_nemotron_h_moe_stage_output(model_dir, layer)?;
    reset_peak_memory()?;
    let engine = MlxStageEngine::spawn(MlxStageEngineConfig {
        model_dir: model_dir.to_path_buf(),
        model_id: "nemotron-h-stage-validation".to_string(),
        stage_index: 1,
        layer_start: u32::try_from(layer)?,
        layer_end: u32::try_from(layer.checked_add(1).context("layer index overflow")?)?,
        compute_dtype: MlxComputeDtype::Bf16,
        weight_quantization: None,
        ctx_size: Some(1),
    })?;
    let width = usize::try_from(engine.info().activation_width)?;
    let values = nemotron_h_validation_values(i32::try_from(width)?);
    let first = execute_validation_input(&engine, 1, width, &values)?;
    let second_session = execute_validation_input(&engine, 2, width, &values)?;
    engine.reset_session(2)?;
    engine.reset_session(1)?;
    let after_reset = execute_validation_input(&engine, 1, width, &values)?;
    let outputs = [first, second_session, after_reset];
    let output_is_finite = outputs
        .iter()
        .flat_map(StageActivation::values)
        .all(f32::is_finite);
    ensure!(
        output_is_finite,
        "Nemotron-H stage-engine output contains non-finite values"
    );
    let stage_output_f32_sha256 = bytes_sha256(&outputs[0].f32_le_bytes);
    let mut comparison = OutputComparison::accumulator();
    for output in &outputs {
        comparison.include(compare_outputs(
            &direct_values,
            &output.values(),
            ATOL,
            RTOL,
        )?);
    }
    let cross_session_comparison =
        compare_outputs(&outputs[0].values(), &outputs[1].values(), ATOL, RTOL)?;
    let cross_session_stable = cross_session_comparison.all_close;
    ensure!(
        cross_session_stable,
        "Nemotron-H output changed across session IDs: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        cross_session_comparison.max_abs,
        cross_session_comparison.max_relative_for_reference_magnitude_above_atol,
    );
    let reset_comparison = compare_outputs(&outputs[0].values(), &outputs[2].values(), ATOL, RTOL)?;
    let session_reset_stable = reset_comparison.all_close;
    ensure!(
        session_reset_stable,
        "Nemotron-H output changed after session reset: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        reset_comparison.max_abs,
        reset_comparison.max_relative_for_reference_magnitude_above_atol,
    );
    let output_within_tolerance = comparison.all_close;
    ensure!(
        output_within_tolerance,
        "Nemotron-H stage-engine output differs from direct block execution: max_abs={} max_relative_for_reference_magnitude_above_atol={} atol={ATOL} rtol={RTOL}",
        comparison.max_abs,
        comparison.max_relative_for_reference_magnitude_above_atol,
    );
    Ok(MlxNemotronHStageValidationReport {
        model_dir: model_dir.to_path_buf(),
        layer,
        input_shape: vec![1, 1, width],
        output_shape: vec![1, outputs[0].token_count, outputs[0].width],
        output_is_finite,
        direct_output_f32_sha256: direct.output_f32_sha256,
        stage_output_f32_sha256,
        output_within_tolerance,
        cross_session_stable,
        session_reset_stable,
        executions_compared: outputs.len(),
        max_abs_diff: comparison.max_abs,
        max_relative_diff_for_reference_magnitude_above_atol: comparison
            .max_relative_for_reference_magnitude_above_atol,
        cross_session_max_abs_diff: cross_session_comparison.max_abs,
        cross_session_max_relative_diff_for_reference_magnitude_above_atol:
            cross_session_comparison.max_relative_for_reference_magnitude_above_atol,
        reset_max_abs_diff: reset_comparison.max_abs,
        reset_max_relative_diff_for_reference_magnitude_above_atol: reset_comparison
            .max_relative_for_reference_magnitude_above_atol,
        comparison_atol: ATOL,
        comparison_rtol: RTOL,
        mlx_active_memory_bytes: active_memory()?,
        mlx_cache_memory_bytes: cache_memory()?,
        mlx_peak_memory_bytes: peak_memory()?,
    })
}

fn execute_validation_input(
    engine: &MlxStageEngine,
    session_id: u64,
    width: usize,
    values: &[f32],
) -> Result<StageActivation> {
    engine
        .execute(StageExecutionRequest {
            session_id,
            kind: StageExecutionKind::Prefill,
            token_ids: vec![0],
            positions: vec![0],
            input: Some(StageActivation::from_values(1, width, values)?),
            sampling: None,
        })?
        .activation
        .context("Nemotron-H internal stage returned no activation")
}

struct OutputComparison {
    all_close: bool,
    max_abs: f32,
    max_relative_for_reference_magnitude_above_atol: f32,
}

impl OutputComparison {
    const fn accumulator() -> Self {
        Self {
            all_close: true,
            max_abs: 0.0,
            max_relative_for_reference_magnitude_above_atol: 0.0,
        }
    }

    fn include(&mut self, next: Self) {
        self.all_close = self.all_close && next.all_close;
        self.max_abs = self.max_abs.max(next.max_abs);
        self.max_relative_for_reference_magnitude_above_atol = self
            .max_relative_for_reference_magnitude_above_atol
            .max(next.max_relative_for_reference_magnitude_above_atol);
    }
}

fn compare_outputs(
    direct: &[f32],
    staged: &[f32],
    atol: f32,
    rtol: f32,
) -> Result<OutputComparison> {
    ensure!(
        direct.len() == staged.len(),
        "direct and staged output lengths differ"
    );
    let mut comparison = OutputComparison::accumulator();
    for (&expected, &actual) in direct.iter().zip(staged) {
        let abs = (expected - actual).abs();
        let relative = if expected.abs() > atol {
            abs / expected.abs()
        } else {
            0.0
        };
        comparison.max_abs = comparison.max_abs.max(abs);
        comparison.max_relative_for_reference_magnitude_above_atol = comparison
            .max_relative_for_reference_magnitude_above_atol
            .max(relative);
        comparison.all_close = comparison.all_close && abs <= atol + rtol * expected.abs();
    }
    Ok(comparison)
}

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(super) struct NemotronHMoeStage {
    block: TransformerBlock,
    stream: Stream,
    compute_dtype: Dtype,
    ctx_size: Option<usize>,
    info: StageEngineInfo,
}

impl NemotronHMoeStage {
    pub(super) fn load(config: MlxStageEngineConfig) -> Result<Self> {
        ensure!(
            config.compute_dtype == MlxComputeDtype::Bf16,
            "Nemotron-H staged execution is currently validated only with BF16 compute"
        );
        ensure!(
            config.weight_quantization.is_none(),
            "Nemotron-H stages must be loaded from an already-derived checkpoint"
        );
        ensure!(
            config.layer_end == config.layer_start.saturating_add(1),
            "Nemotron-H staged execution currently requires exactly one layer"
        );
        let args = get_nemotron_h_model_args(&config.model_dir)?;
        let layer = usize::try_from(config.layer_start)?;
        ensure!(
            args.layer_block_types()?.get(layer) == Some(&LayerBlockType::Moe),
            "Nemotron-H staged execution currently supports only stateless MoE layers"
        );
        let info = StageEngineInfo {
            engine: "mlx".to_string(),
            model_id: config.model_id,
            stage_index: config.stage_index,
            layer_start: config.layer_start,
            layer_end: config.layer_end,
            total_layers: u32::try_from(args.num_hidden_layers)?,
            activation_width: u32::try_from(args.hidden_size)?,
        };
        info.validate()?;
        ensure!(
            !info.is_first() && !info.is_final(),
            "Nemotron-H MoE proof stages must be internal residual stages"
        );

        let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
        let weights_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut block = TransformerBlock::new(&args, layer, &stream)?;
        let load_config =
            StrictLoadConfig::default().strip_prefix(format!("model.layers.{layer}."));
        let mut load_report = StrictLoadReport::default();
        load_safetensors_dir_strict(
            &mut block,
            &config.model_dir,
            &weights_stream,
            &load_config,
            &mut load_report,
        )?;
        load_report.finish(&block, &load_config)?;
        block.copy_to_stream(&stream)?;
        stream.synchronize()?;
        tracing::info!(
            model = %info.model_id,
            stage = info.stage_index,
            layer = info.layer_start,
            tensors = block.parameters().flatten().len(),
            weight_quantization = "checkpoint",
            "MLX Nemotron-H MoE stage loaded",
        );
        Ok(Self {
            block,
            stream,
            compute_dtype: config.compute_dtype.mlx(),
            ctx_size: config.ctx_size.map(usize::try_from).transpose()?,
            info,
        })
    }

    pub(super) const fn info(&self) -> &StageEngineInfo {
        &self.info
    }

    pub(super) fn execute(
        &mut self,
        request: StageExecutionRequest,
    ) -> Result<StageExecutionOutput> {
        if request.kind == StageExecutionKind::Verify {
            bail!("MLX Nemotron-H stage verification is not implemented yet");
        }
        if request
            .sampling
            .as_ref()
            .is_some_and(|sampling| sampling.enabled())
        {
            bail!("MLX staged execution currently supports greedy sampling only");
        }
        ensure!(!request.token_ids.is_empty(), "stage request has no tokens");
        let input = request
            .input
            .as_ref()
            .context("Nemotron-H MoE stage requires residual input")?;
        ensure!(
            input.token_count == request.token_ids.len(),
            "input activation token count does not match token sideband"
        );
        ensure!(
            input.width == self.info.activation_width as usize,
            "input activation width mismatch"
        );
        if let Some(ctx_size) = self.ctx_size {
            ensure!(
                input.token_count <= ctx_size,
                "MLX stage context limit {ctx_size} exceeded by {} tokens",
                input.token_count
            );
        }

        let hidden = Array::from_slice(
            &input.values(),
            &[
                1,
                i32::try_from(input.token_count)?,
                i32::try_from(input.width)?,
            ],
        )
        .as_dtype(self.compute_dtype, &self.stream)?;
        let output = self.block.forward(
            BlockInput {
                x: &hidden,
                mask: None,
                cache: None,
            },
            &self.stream,
        )?;
        Ok(StageExecutionOutput {
            activation: Some(array_activation(&output, &self.stream)?),
            predicted_tokens: Vec::new(),
        })
    }

    pub(super) fn reset_session(&mut self, _session_id: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_comparison_accepts_roundoff_and_rejects_drift() {
        let close = compare_outputs(
            &[1.0, -0.5, 0.0],
            &[1.000_000_1, -0.5, 1.0e-5],
            1.0e-4,
            1.0e-4,
        )
        .unwrap();
        assert!(close.all_close);
        assert!(close.max_abs <= 1.0e-5);
        assert!(close.max_relative_for_reference_magnitude_above_atol < 1.0e-6);

        let drift = compare_outputs(&[1.0, -0.5], &[1.01, -0.5], 1.0e-4, 1.0e-4).unwrap();
        assert!(!drift.all_close);
        assert!(drift.max_abs > 0.009);
    }
}
