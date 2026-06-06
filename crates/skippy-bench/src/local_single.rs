use std::{
    fs,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    cli::LocalSingleArgs,
    model_identity::model_identity_for_path,
    support::{ChildGuard, generate_run_id, retry, temp_config_path, temp_db_path},
};

#[derive(Deserialize)]
struct CreateRunResponse {
    run_id: String,
}

#[derive(Deserialize)]
struct StageStatus {
    ready: bool,
    runtime_loaded: bool,
}

#[derive(Serialize)]
struct TextRequest<'a> {
    request_id: &'a str,
    session_id: &'a str,
    prompt: &'a str,
    max_new_tokens: usize,
}

#[derive(Serialize)]
struct TextRequestResult {
    request_id: String,
    session_id: String,
    elapsed_ms: f64,
    tokenize_elapsed_ms: Option<f64>,
    prefill_elapsed_ms: Option<f64>,
    decode_elapsed_ms: Option<f64>,
    prompt_token_count: usize,
    generated_token_count: usize,
    generated_tokens_per_sec: Option<f64>,
    decode_tokens_per_sec: Option<f64>,
}

pub fn local_single(args: LocalSingleArgs) -> Result<()> {
    if args.layer_start >= args.layer_end {
        bail!("layer_start must be less than layer_end");
    }
    if args.request_count == 0 {
        bail!("request_count must be greater than zero");
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(args.startup_timeout_secs))
        .build()
        .context("failed to build HTTP client")?;
    let run_id = args.run_id.unwrap_or_else(generate_run_id);
    let metrics_http = format!("http://{}", args.metrics_http_addr);
    let metrics_otlp = format!("http://{}", args.metrics_otlp_grpc_addr);
    let stage_http = format!("http://{}", args.stage_bind_addr);
    let db = args.db.unwrap_or_else(|| temp_db_path(&run_id));
    let stage_config = temp_config_path(&run_id);
    let model_identity = model_identity_for_path(&args.model_id, Some(&args.model_path))?;

    let mut metrics_command = Command::new(&args.metrics_server_bin);
    metrics_command.args([
        "serve",
        "--db",
        db.to_str().context("db path is not valid UTF-8")?,
        "--http-addr",
        &args.metrics_http_addr.to_string(),
        "--otlp-grpc-addr",
        &args.metrics_otlp_grpc_addr.to_string(),
    ]);
    if args.child_logs {
        metrics_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        metrics_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _metrics = ChildGuard::spawn(metrics_command)?;

    let run_config = json!({
        "run_id": run_id,
        "topology_id": args.topology_id,
        "model_id": model_identity.model_id,
        "model_identity": model_identity,
        "mode": "local-single",
    });
    retry(args.startup_timeout_secs, || {
        let response = client
            .post(format!("{metrics_http}/v1/runs"))
            .json(&run_config)
            .send()
            .and_then(|response| response.error_for_status())?
            .json::<CreateRunResponse>()?;
        if response.run_id == run_id {
            Ok(())
        } else {
            Err(anyhow!(
                "metrics-server returned unexpected run_id {}",
                response.run_id
            ))
        }
    })
    .context("metrics-server did not become ready")?;

    let config = json!({
        "run_id": run_id,
        "topology_id": run_config["topology_id"],
        "model_id": run_config["model_id"],
        "model_path": args.model_path,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "n_gpu_layers": args.n_gpu_layers,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "filter_tensors_on_load": false,
        "load_mode": "runtime-slice",
        "bind_addr": args.stage_bind_addr,
        "upstream": null,
        "downstream": null
    });
    fs::write(&stage_config, serde_json::to_vec_pretty(&config)?)
        .with_context(|| format!("failed to write {}", stage_config.display()))?;

    let mut stage_command = Command::new(&args.stage_server_bin);
    stage_command.args([
        "serve",
        "--config",
        stage_config
            .to_str()
            .context("stage config path is not valid UTF-8")?,
        "--metrics-otlp-grpc",
        &metrics_otlp,
    ]);
    if args.child_logs {
        stage_command
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit());
    } else {
        stage_command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let _stage = ChildGuard::spawn(stage_command)?;

    retry(args.startup_timeout_secs, || {
        let status = client
            .get(format!("{stage_http}/v1/status"))
            .send()
            .and_then(|response| response.error_for_status())
            .map_err(anyhow::Error::new)?
            .json::<StageStatus>()?;
        if status.ready && status.runtime_loaded {
            Ok(())
        } else {
            Err(anyhow!("stage is not ready yet"))
        }
    })
    .context("stage server did not become ready")?;

    if args.warmup_new_tokens > 0 {
        let warmup_request = TextRequest {
            request_id: "local-single-warmup-1",
            session_id: "local-single-warmup-session-1",
            prompt: &args.prompt,
            max_new_tokens: args.warmup_new_tokens,
        };
        client
            .post(format!("{stage_http}/v1/text"))
            .json(&warmup_request)
            .send()
            .context("failed to send warmup text request")?
            .error_for_status_with_body("warmup text request failed")?;
    }

    let mut request_results = Vec::new();
    let mut text_response = Value::Null;
    for request_index in 0..args.request_count {
        let request_id = format!("local-single-request-{}", request_index + 1);
        let session_id = if args.reuse_session {
            "local-single-session-1".to_string()
        } else {
            format!("local-single-session-{}", request_index + 1)
        };
        let request = TextRequest {
            request_id: &request_id,
            session_id: &session_id,
            prompt: &args.prompt,
            max_new_tokens: args.max_new_tokens,
        };
        let text_request_start = Instant::now();
        text_response = client
            .post(format!("{stage_http}/v1/text"))
            .json(&request)
            .send()
            .context("failed to send text request")?
            .error_for_status_with_body("text request failed")?
            .json()
            .context("failed to parse text response")?;
        let elapsed = text_request_start.elapsed();
        let generated_token_count = text_response
            .get("generated_token_ids")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let prompt_token_count = text_response
            .get("prompt_token_ids")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        let tokenize_elapsed_ms = text_response
            .get("tokenize_elapsed_ms")
            .and_then(Value::as_f64);
        let prefill_elapsed_ms = text_response
            .get("prefill_elapsed_ms")
            .and_then(Value::as_f64);
        let decode_elapsed_ms = text_response
            .get("decode_elapsed_ms")
            .and_then(Value::as_f64);
        let generated_tokens_per_sec = if elapsed.as_secs_f64() > 0.0 {
            Some(generated_token_count as f64 / elapsed.as_secs_f64())
        } else {
            None
        };
        let decode_tokens_per_sec = decode_elapsed_ms
            .filter(|elapsed_ms| *elapsed_ms > 0.0)
            .map(|elapsed_ms| generated_token_count as f64 / (elapsed_ms / 1000.0));
        request_results.push(TextRequestResult {
            request_id,
            session_id,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            tokenize_elapsed_ms,
            prefill_elapsed_ms,
            decode_elapsed_ms,
            prompt_token_count,
            generated_token_count,
            generated_tokens_per_sec,
            decode_tokens_per_sec,
        });
    }
    let generated_token_count = request_results
        .iter()
        .map(|request| request.generated_token_count)
        .sum::<usize>();
    let text_request_elapsed_ms = request_results
        .iter()
        .map(|request| request.elapsed_ms)
        .sum::<f64>();
    let generated_tokens_per_sec = if text_request_elapsed_ms > 0.0 {
        Some(generated_token_count as f64 / (text_request_elapsed_ms / 1000.0))
    } else {
        None
    };

    thread::sleep(Duration::from_secs(1));
    client
        .post(format!("{metrics_http}/v1/runs/{run_id}/finalize"))
        .send()
        .context("failed to finalize run")?
        .error_for_status()
        .context("metrics-server rejected finalize")?;
    let report: Value = client
        .get(format!("{metrics_http}/v1/runs/{run_id}/report.json"))
        .send()
        .context("failed to fetch report")?
        .error_for_status()
        .context("metrics-server rejected report fetch")?
        .json()
        .context("failed to parse report")?;

    if let Some(output) = args.output {
        fs::write(&output, serde_json::to_vec_pretty(&report)?)
            .with_context(|| format!("failed to write {}", output.display()))?;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "run_id": run_id,
            "model_identity": run_config["model_identity"],
            "text_response": text_response,
            "text_request_elapsed_ms": text_request_elapsed_ms,
            "generated_token_count": generated_token_count,
            "generated_tokens_per_sec": generated_tokens_per_sec,
            "warmup_new_tokens": args.warmup_new_tokens,
            "request_count": args.request_count,
            "reuse_session": args.reuse_session,
            "request_results": request_results,
            "report_counts": report["counts"],
        }))?
    );

    Ok(())
}

trait ResponseStatusBodyExt {
    fn error_for_status_with_body(self, context: &str) -> Result<Response>;
}

impl ResponseStatusBodyExt for Response {
    fn error_for_status_with_body(self, context: &str) -> Result<Response> {
        let status = self.status();
        if status.is_success() {
            return Ok(self);
        }
        let body = self
            .text()
            .unwrap_or_else(|error| format!("failed to read error body: {error}"));
        bail!("{context}: HTTP status {status}: {body}");
    }
}
