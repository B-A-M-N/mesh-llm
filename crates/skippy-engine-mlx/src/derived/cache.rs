//! Identity-bound reusable cache for derived MLX stage artifacts.

use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use model_hf::safetensors_stage::{SafetensorsStageMaterializer, SafetensorsStageRequest};
use serde::Serialize;

use super::{
    DERIVED_STAGE_SCHEMA_VERSION, MlxDerivedStageConfig, MlxDerivedStageReport, REPORT_FILE,
    artifact_file_bytes, derive_quantized_stage, open_locked, output_content_sha256,
    prepare_derivation_recipe, sha256_file,
};
use crate::stage::MlxWeightQuantization;

/// Configuration for an identity-bound, reusable derived-stage cache entry.
#[derive(Clone, Debug)]
pub struct MlxDerivedStageCacheConfig {
    pub source: SafetensorsStageRequest,
    pub cache_root: PathBuf,
    pub quantization: MlxWeightQuantization,
    /// Soft output bundle target. A single packed tensor may exceed this size.
    pub shard_size_bytes: usize,
}

/// Result of a managed derived-stage lookup or build.
#[derive(Clone, Debug, Serialize)]
pub struct MlxDerivedStageCacheResult {
    pub cache_hit: bool,
    /// Tensor payload range requests made by this invocation.
    pub source_range_request_count: usize,
    pub output_dir: PathBuf,
    pub report: MlxDerivedStageReport,
}

/// Loads a verified derived stage from cache or builds it from exact tensor ranges.
pub fn derive_quantized_stage_cached(
    materializer: &SafetensorsStageMaterializer,
    config: &MlxDerivedStageCacheConfig,
) -> Result<MlxDerivedStageCacheResult> {
    ensure!(
        config.shard_size_bytes > 0,
        "derived shard size must be non-zero"
    );
    let recipe = prepare_derivation_recipe(
        materializer,
        config.source.clone(),
        config.quantization,
        config.shard_size_bytes,
    )?;
    fs::create_dir_all(&config.cache_root).with_context(|| {
        format!(
            "create MLX derived stage cache {}",
            config.cache_root.display()
        )
    })?;
    let output_dir = config.cache_root.join(&recipe);
    let lock_path = config.cache_root.join(format!(".{recipe}.lock"));
    // Keep this pathname stable across invocations. Removing an advisory-lock
    // file after unlock can split waiters between the unlinked and new inodes.
    let _lock = open_locked(&lock_path, false)?.expect("blocking cache lock is acquired");

    if let Some(report) = load_cached(&output_dir, &recipe)? {
        return Ok(MlxDerivedStageCacheResult {
            cache_hit: true,
            source_range_request_count: 0,
            output_dir,
            report,
        });
    }
    remove_invalid_cache_entry(&output_dir)?;
    let report = derive_quantized_stage(
        materializer,
        &MlxDerivedStageConfig {
            source: config.source.clone(),
            output_dir: output_dir.clone(),
            quantization: config.quantization,
            shard_size_bytes: config.shard_size_bytes,
        },
    )?;
    Ok(MlxDerivedStageCacheResult {
        cache_hit: false,
        source_range_request_count: report.source_range_request_count,
        output_dir,
        report,
    })
}

fn load_cached(output_dir: &Path, recipe: &str) -> Result<Option<MlxDerivedStageReport>> {
    if !output_dir.is_dir() {
        return Ok(None);
    }
    if !contains_only_regular_files(output_dir)? {
        return Ok(None);
    }
    let bytes = match fs::read(output_dir.join(REPORT_FILE)) {
        Ok(bytes) => bytes,
        Err(_) => return Ok(None),
    };
    let mut report = match serde_json::from_slice::<MlxDerivedStageReport>(&bytes) {
        Ok(report) => report,
        Err(_) => return Ok(None),
    };
    if report.schema_version != DERIVED_STAGE_SCHEMA_VERSION
        || report.derivation_recipe_sha256 != recipe
        || report.artifact_file_bytes != artifact_file_bytes(output_dir)?
        || report.output_content_sha256 != output_content_sha256(output_dir)?
        || !shards_match(output_dir, &report)?
    {
        return Ok(None);
    }
    // `output_dir` is diagnostic, not artifact identity. Refresh it so a
    // relocated cache remains reusable without exposing its stale build path.
    report.output_dir = output_dir.to_path_buf();
    Ok(Some(report))
}

fn contains_only_regular_files(output_dir: &Path) -> Result<bool> {
    for entry in fs::read_dir(output_dir)? {
        if !entry?.file_type()?.is_file() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn shards_match(output_dir: &Path, report: &MlxDerivedStageReport) -> Result<bool> {
    for shard in &report.shards {
        let relative = Path::new(&shard.file);
        if relative.components().count() != 1
            || !matches!(relative.components().next(), Some(Component::Normal(_)))
        {
            return Ok(false);
        }
        let path = output_dir.join(relative);
        let metadata = match fs::metadata(&path) {
            Ok(metadata) if metadata.is_file() => metadata,
            _ => return Ok(false),
        };
        if metadata.len() != shard.file_bytes || sha256_file(&path)? != shard.sha256 {
            return Ok(false);
        }
    }
    Ok(!report.shards.is_empty())
}

fn remove_invalid_cache_entry(path: &Path) -> Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)
            .with_context(|| format!("remove invalid derived stage cache {}", path.display()))?;
    } else if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("remove invalid derived stage cache {}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::derived::{MlxDerivedStageShard, write_json};

    fn write_test_cache(output_dir: &Path, recipe: &str) -> MlxDerivedStageReport {
        fs::create_dir_all(output_dir).unwrap();
        fs::write(output_dir.join("config.json"), b"{}\n").unwrap();
        fs::write(output_dir.join("model.safetensors"), b"packed").unwrap();
        let shard_path = output_dir.join("model.safetensors");
        let mut report = MlxDerivedStageReport {
            schema_version: DERIVED_STAGE_SCHEMA_VERSION,
            derivation_recipe_sha256: recipe.to_string(),
            output_content_sha256: output_content_sha256(output_dir).unwrap(),
            checkpoint_sha256: "checkpoint".to_string(),
            plan_sha256: "plan".to_string(),
            repo: "owner/model".to_string(),
            revision: "0".repeat(40),
            layer_start: 0,
            layer_end: 1,
            quantization: json!({"mode":"affine","bits":4,"group_size":64}),
            quantization_label: "affine-4bit-g64".to_string(),
            safemlx_revision: "revision".to_string(),
            output_dir: output_dir.to_path_buf(),
            source_tensor_count: 1,
            source_tensor_bytes: 16,
            source_range_request_count: 1,
            source_temporary_file_peak_bytes: 16,
            quantized_tensor_count: 1,
            copied_tensor_count: 0,
            output_tensor_bytes: 6,
            artifact_file_bytes: artifact_file_bytes(output_dir).unwrap(),
            working_disk_peak_bytes: 22,
            mlx_active_memory_bytes: 0,
            mlx_cache_memory_bytes: 0,
            mlx_peak_memory_bytes: 0,
            shards: vec![MlxDerivedStageShard {
                file: "model.safetensors".to_string(),
                file_bytes: 6,
                sha256: sha256_file(&shard_path).unwrap(),
            }],
        };
        write_json(output_dir.join(REPORT_FILE), &report).unwrap();
        report.output_content_sha256 = output_content_sha256(output_dir).unwrap();
        report
    }

    #[test]
    fn validates_cache_content_and_rejects_tampering() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        let expected = write_test_cache(&output, "recipe");

        let cached = load_cached(&output, "recipe").unwrap().unwrap();
        assert_eq!(cached.output_content_sha256, expected.output_content_sha256);

        fs::write(output.join("model.safetensors"), b"broken").unwrap();
        assert!(load_cached(&output, "recipe").unwrap().is_none());
    }

    #[test]
    fn rejects_traversal_in_cached_shard_name() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        let mut report = write_test_cache(&output, "recipe");
        report.shards[0].file = "../model.safetensors".to_string();
        assert!(!shards_match(&output, &report).unwrap());
    }

    #[test]
    fn accepts_relocated_cache_and_refreshes_diagnostic_path() {
        let directory = tempfile::tempdir().unwrap();
        let original_root = directory.path().join("original");
        let original = original_root.join("recipe");
        write_test_cache(&original, "recipe");
        let relocated_root = directory.path().join("relocated");
        fs::rename(&original_root, &relocated_root).unwrap();
        let relocated = relocated_root.join("recipe");

        let cached = load_cached(&relocated, "recipe").unwrap().unwrap();

        assert_eq!(cached.output_dir, relocated);
    }

    #[cfg(unix)]
    #[test]
    fn rejects_non_regular_cache_entries() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("recipe");
        write_test_cache(&output, "recipe");
        symlink("config.json", output.join("unexpected-index.json")).unwrap();

        assert!(load_cached(&output, "recipe").unwrap().is_none());
    }
}
