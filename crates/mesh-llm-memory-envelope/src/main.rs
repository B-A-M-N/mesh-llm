use clap::Parser;

mod probe;

use probe::sample;

/// Memory envelope measurement configuration.
#[derive(Clone, Debug, Parser, serde::Serialize, serde::Deserialize)]
#[command(name = "mesh-memory-envelope", about = "Measure runtime memory envelope for Mesh-LLM")]
pub struct Config {
    /// Backend device: "cuda" or "vulkan"
    #[arg(long, default_value = "cuda")]
    pub backend: String,

    /// Path to model package or stage GGUF
    #[arg(long)]
    pub model_path: Option<String>,

    /// Target HOST placement in GiB (0 = all accelerator)
    #[arg(long, default_value = "0")]
    pub host_gib: u64,

    /// Prompt token count for warm forward
    #[arg(long, default_value = "1")]
    pub prompt_tokens: u32,

    /// Number of warmup iterations before measurement
    #[arg(long, default_value = "1")]
    pub warmup: u32,

    /// Number of measurement repetitions
    #[arg(long, default_value = "3")]
    pub repetitions: u32,

    /// Output file path (stdout if omitted)
    #[arg(long)]
    pub output: Option<String>,
}

/// A single memory envelope measurement.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemoryEnvelopeSample {
    pub requested_host_weight_bytes: u64,
    pub planned_accelerator_weight_bytes: u64,

    pub rss_before_bytes: u64,
    pub rss_after_load_bytes: u64,
    pub rss_after_forward_bytes: u64,

    pub mem_available_before_bytes: u64,
    pub mem_available_after_load_bytes: u64,
    pub mem_available_after_forward_bytes: u64,

    pub swap_before_bytes: u64,
    pub swap_after_bytes: u64,

    pub minor_faults_delta: u64,
    pub major_faults_delta: u64,

    pub accelerator_memory_before_bytes: Option<u64>,
    pub accelerator_memory_after_load_bytes: Option<u64>,
    pub accelerator_memory_after_forward_bytes: Option<u64>,

    pub host_backend_buffer_bytes: Option<u64>,
    pub accelerator_backend_buffer_bytes: Option<u64>,

    pub load_duration_ms: u64,
    pub warm_forward_duration_ms: u64,

    pub forward_ok: bool,
}

/// Result of a memory envelope run.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MemoryEnvelopeResult {
    pub config: Config,
    pub samples: Vec<MemoryEnvelopeSample>,
    pub failed: bool,
    pub error: Option<String>,
}

fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();
    let config = Config::parse();

    tracing::info!(?config, "starting memory envelope measurement");

    let result = run_measurement(&config);

    let json = serde_json::to_string_pretty(&result)?;
    match &config.output {
        Some(path) => std::fs::write(path, &json)?,
        None => println!("{}", json),
    }

    if result.failed {
        std::process::exit(1);
    }
    Ok(())
}

fn run_measurement(config: &Config) -> MemoryEnvelopeResult {
    // Baseline sample.
    let baseline = sample();

    if baseline.is_none() {
        return MemoryEnvelopeResult {
            config: config.clone(),
            samples: vec![],
            failed: true,
            error: Some("baseline memory probe failed".to_string()),
        };
    }

    // If no model path provided, report probe-only success.
    if config.model_path.is_none() {
        return MemoryEnvelopeResult {
            config: config.clone(),
            samples: vec![],
            failed: false,
            error: None,
        };
    }

    // TODO: Load model with specified HOST placement profile, measure at each boundary.
    tracing::warn!("model loading not yet implemented in D0a harness");

    MemoryEnvelopeResult {
        config: config.clone(),
        samples: vec![],
        failed: false,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baseline_probe() {
        let result = run_measurement(&Config {
            backend: "cuda".to_string(),
            model_path: None,
            host_gib: 0,
            prompt_tokens: 1,
            warmup: 1,
            repetitions: 1,
            output: None,
        });
        assert!(!result.failed, "probe-only run should not fail");
    }
}
