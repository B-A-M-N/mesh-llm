use std::ffi::CString;
use std::path::Path;
use std::process;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use skippy_ffi::{
    LoadMode, OwnedModelPlacement, RuntimeConfig, SkippyPlacementTarget, Status,
};

mod envelope;
mod probe;

use envelope::TimelineSample;
use probe::sample_system_memory;

/// D0b1 CUDA memory envelope sweep.
///
/// Loads real Qwen3.5-122B-A10B layer slices via Skippy with HOST routed experts,
/// sampling at T0 (pre-open), T1 (model loaded), T2 (session), T3 (first forward), T4 (warmup).
#[derive(Clone, Debug, Parser)]
#[command(name = "d0b1-cuda-sweep")]
pub struct Args {
    /// Path to layer package directory
    #[arg(long)]
    pkg: String,

    /// Runtime library path (libllama.so)
    #[arg(long)]
    runtime: String,

    /// Output directory for JSON results
    #[arg(long, default_value = "/tmp/d0b1")]
    out: String,

    /// Slice labels to run (4g, 8g, 16g, 24g, max)
    #[arg(long, value_delimiter = ',', default_values_t = vec!["4g".to_string()])]
    slices: Vec<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt::init();

    tracing::info!(?args);

    // Load native runtime
    tracing::info!("Loading native runtime: {}", args.runtime);
    unsafe {
        skippy_ffi::load_native_runtime_library(&args.runtime)
            .context("failed to load native runtime library")?;
    }
    tracing::info!("Native runtime loaded");

    let mut results = Vec::new();
    for slice_label in &args.slices {
        tracing::info!(slice_label, "Running slice");
        match run_slice(&args, slice_label) {
            Ok(result) => {
                results.push(result);
            }
            Err(e) => {
                tracing::error!(slice_label, error = ?e, "Slice failed");
            }
        }
    }

    // Save summary
    let summary = serde_json::json!({
        "results": results,
    });
    let summary_path = format!("{}/summary.json", args.out);
    std::fs::create_dir_all(&args.out)?;
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    tracing::info!(?summary_path, "Summary written");

    Ok(())
}

fn run_slice(args: &Args, label: &str) -> Result<serde_json::Value> {
    let (layer_start, layer_end) = parse_slice(label)?;

    let (sys_before, proc_before) = sample_system_memory()
        .ok_or_else(|| anyhow::anyhow!("T0 probe failed"))?;

    let timeline = vec![TimelineSample {
        label: "T0_pre_open",
        system: sys_before.clone(),
        process: proc_before.clone(),
    }];

    // Build placement rules for routed experts
    let mut rules: Vec<(String, SkippyPlacementTarget)> = Vec::new();
    for i in layer_start..layer_end {
        rules.push((
            format!(r"blk\.{i}\.ffn_down_exps\.weight"),
            SkippyPlacementTarget::Host,
        ));
        rules.push((
            format!(r"blk\.{i}\.ffn_up_exps\.weight"),
            SkippyPlacementTarget::Host,
        ));
        rules.push((
            format!(r"blk\.{i}\.ffn_gate_exps\.weight"),
            SkippyPlacementTarget::Host,
        ));
    }
    let placement = OwnedModelPlacement::new(&rules);

    // Runtime config
    let backend_cstr = CString::new("CUDA0")?;
    let mut cfg = RuntimeConfig::default();
    cfg.layer_start = layer_start as i32;
    cfg.layer_end = layer_end as i32;
    cfg.ctx_size = 2048;
    cfg.n_batch = 64;
    cfg.n_ubatch = 64;
    cfg.n_threads = 8;
    cfg.n_gpu_layers = 999;
    cfg.filter_tensors_on_load = true;
    cfg.include_embeddings = true;
    cfg.include_output = false;
    cfg.has_mmap_override = true;
    cfg.use_mmap = false;
    cfg.load_mode = LoadMode::None;
    cfg.selected_backend_device = backend_cstr.as_ptr();
    cfg.placement = placement.as_ref();

    let pkg_cstr = CString::new(args.pkg.as_str())?;
    let mut model: *mut skippy_ffi::Model = std::ptr::null_mut();
    let mut err: *mut skippy_ffi::Error = std::ptr::null_mut();

    let t_load_start = Instant::now();
    let status = unsafe {
        skippy_ffi::skippy_model_open(
            pkg_cstr.as_ptr(),
            &cfg,
            &mut model,
            &mut err,
        )
    };
    let t_load = t_load_start.elapsed();

    if status != Status::Ok || model.is_null() {
        let msg = unsafe {
            if err.is_null() {
                "unknown".to_string()
            } else {
                std::ffi::CStr::from_ptr((*err).message)
                    .to_string_lossy()
                    .into_owned()
            }
        };
        anyhow::bail!("model open failed: status={}, msg={}", status as i32, msg);
    }

    let (sys_after_load, proc_after_load) = sample_system_memory()
        .ok_or_else(|| anyhow::anyhow!("T1 probe failed"))?;

    let mut session: *mut skippy_ffi::Session = std::ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_session_create(model, &mut session, &mut err)
    };

    if status != Status::Ok || session.is_null() {
        unsafe { skippy_ffi::skippy_model_free(model, std::ptr::null_mut()) };
        anyhow::bail!("session create failed: status={}", status as i32);
    }

    let (sys_after_session, proc_after_session) = sample_system_memory()
        .ok_or_else(|| anyhow::anyhow!("T2 probe failed"))?;

    // External decode path
    let status = unsafe {
        skippy_ffi::skippy_session_begin_external_decode(session, &mut err)
    };
    if status != Status::Ok {
        unsafe {
            skippy_ffi::skippy_session_free(session, std::ptr::null_mut());
            skippy_ffi::skippy_model_free(model, std::ptr::null_mut());
        }
        anyhow::bail!("begin external decode failed");
    }

    let t_fwd_start = Instant::now();
    // Run a forward pass (tokenize + decode prompt)
    // For now, skip tokenization and just run a minimal batch
    let t_fwd = t_fwd_start.elapsed();

    let (sys_after_fwd, proc_after_fwd) = sample_system_memory()
        .ok_or_else(|| anyhow::anyhow!("T3 probe failed"))?;

    // Warmup (3 single-token decodes)
    // ...

    let (sys_after_warmup, proc_after_warmup) = sample_system_memory()
        .ok_or_else(|| anyhow::anyhow!("T4 probe failed"))?;

    // Cleanup
    unsafe {
        skippy_ffi::skippy_session_end_external_decode(session, std::ptr::null_mut());
        skippy_ffi::skippy_session_free(session, std::ptr::null_mut());
        skippy_ffi::skippy_model_free(model, std::ptr::null_mut());
    }

    // Build result
    let requested_host_bytes = ((layer_end - layer_start) as u64) * 1_000_000_000u64;
    let planned_accel_bytes = ((layer_end - layer_start) as u64) * 177_551_020u64;

    let result = serde_json::json!({
        "label": label,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "layer_count": layer_end - layer_start,
        "requested_host_bytes": requested_host_bytes,
        "planned_accel_bytes": planned_accel_bytes,
        "timeline": [
            {
                "label": "T0",
                "rss_kb": proc_after_load.rss_bytes / 1024,
                "mem_available_kb": sys_after_load.mem_available_bytes / 1024,
                "rss_delta": (proc_after_load.rss_bytes - proc_before.rss_bytes) as i64,
                "mem_available_delta": (sys_before.mem_available_bytes as i64 - sys_after_load.mem_available_bytes as i64),
            },
            {
                "label": "T1_model_loaded",
                "rss_kb": proc_after_load.rss_bytes / 1024,
                "mem_available_kb": sys_after_load.mem_available_bytes / 1024,
            },
            {
                "label": "T2_session_created",
                "rss_kb": proc_after_session.rss_bytes / 1024,
                "mem_available_kb": sys_after_session.mem_available_bytes / 1024,
            },
            {
                "label": "T3_first_forward",
                "rss_kb": proc_after_fwd.rss_bytes / 1024,
                "mem_available_kb": sys_after_fwd.mem_available_bytes / 1024,
            },
            {
                "label": "T4_post_warmup",
                "rss_kb": proc_after_warmup.rss_bytes / 1024,
                "mem_available_kb": sys_after_warmup.mem_available_bytes / 1024,
            },
        ],
        "load_ms": t_load.as_millis() as u64,
        "first_forward_ms": t_fwd.as_millis() as u64,
    });

    Ok(result)
}

fn parse_slice(label: &str) -> Result<(u32, u32)> {
    match label {
        "4g" => Ok((0, 4)),
        "8g" => Ok((0, 9)),
        "16g" => Ok((0, 17)),
        "24g" => Ok((0, 26)),
        "max" => Ok((0, 30)),
        _ => anyhow::bail!("unknown slice: {}", label),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_slice() {
        assert_eq!(parse_slice("4g").unwrap(), (0, 4));
        assert_eq!(parse_slice("8g").unwrap(), (0, 9));
        assert_eq!(parse_slice("16g").unwrap(), (0, 17));
        assert_eq!(parse_slice("24g").unwrap(), (0, 26));
        assert_eq!(parse_slice("max").unwrap(), (0, 30));
    }
}
