use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{cmp::Ordering, env, fs, path::PathBuf};

const DEFAULT_SCENARIO: &str = "steady_decode";
const DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR: f64 = 0.10;
const DEFAULT_MAX_INDIVIDUAL_ERROR: f64 = 0.10;
const DEFAULT_MAX_NOISY: usize = 0;
const DEFAULT_MAX_RUNTIME_ERRORS: usize = 0;
const DEFAULT_MIN_ACCURACY_MODELS: usize = 1;

#[derive(Debug)]
struct Args {
    report_json: PathBuf,
    scenario: String,
    max_median_absolute_error: f64,
    max_individual_error: f64,
    max_noisy: usize,
    max_runtime_errors: usize,
    min_models: Option<usize>,
    min_accuracy_models: Option<usize>,
    markdown_out: Option<PathBuf>,
    require_graph_inventory_match: bool,
    allow_classified_individual_misses: bool,
    report_only: bool,
}

#[derive(Debug, Deserialize)]
struct ValidationReport {
    models: Vec<ModelReport>,
    summary: Option<ValidationSummary>,
}

#[derive(Debug, Deserialize)]
struct ValidationSummary {
    error_count: usize,
    #[serde(default)]
    runtime_error_count: usize,
}

#[derive(Debug, Deserialize)]
struct ModelReport {
    input_ref: String,
    recommendation: Option<RecommendationReport>,
    decode_probe_diagnostic: Option<DecodeProbeDiagnostic>,
    graph_inventory_diagnostic: Option<GraphInventoryDiagnostic>,
    runtime_diagnostic: Option<RuntimeDiagnostic>,
    benchmarks: Vec<ScenarioReport>,
    errors: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RecommendationReport {
    #[serde(default)]
    estimated_decode_context_tokens: Option<u32>,
    decode_cost_breakdown: Option<DecodeCostBreakdown>,
}

#[derive(Debug, Deserialize)]
struct DecodeCostBreakdown {
    #[serde(default)]
    context_tokens: Option<u32>,
    bandwidth_ms: f64,
    compute_ms: f64,
    fixed_overhead_ms: f64,
    #[serde(default)]
    runtime_overhead_ms: f64,
    measured_graph_overhead_ms: f64,
    architecture_overhead_ms: f64,
    #[serde(default)]
    sampled_decode_sampler_ms: f64,
    selected_time_ms: f64,
    estimated_tokens_per_sec: f64,
    probed_bytes: u64,
    fallback_bytes: u64,
    groups: Vec<DecodeCostGroupBreakdown>,
}

#[derive(Debug, Deserialize)]
struct DecodeCostGroupBreakdown {
    group: String,
    tensor_type: String,
    traffic_bytes: u64,
    source: String,
    bandwidth_bytes_per_sec: u64,
    bandwidth_ms: f64,
    probe_name: Option<String>,
    probe_shape_distance: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct DecodeProbeDiagnostic {
    predicted_tokens_per_second: Option<f64>,
    #[serde(default)]
    scenario_predicted_tokens_per_second: Option<f64>,
    abi_tokens_per_second: Option<f64>,
    observed_tokens_per_second: Option<f64>,
    observed_over_fit: Option<f64>,
    #[serde(default)]
    observed_over_scenario_fit: Option<f64>,
    abi_over_fit: Option<f64>,
    #[serde(default)]
    abi_over_scenario_fit: Option<f64>,
    observed_over_abi: Option<f64>,
    #[serde(default)]
    scenario_prediction_source: Option<String>,
    observed_vs_fit: String,
    #[serde(default)]
    observed_vs_scenario_fit: Option<String>,
    abi_vs_fit: String,
    #[serde(default)]
    abi_vs_scenario_fit: Option<String>,
    observed_vs_abi: String,
    #[serde(default)]
    predicted_decode_submission_ms_per_token: Option<f64>,
    #[serde(default)]
    abi_decode_call_ms_per_token: Option<f64>,
    #[serde(default)]
    decode_submission_residual_ms_per_token: Option<f64>,
    #[serde(default)]
    decode_submission_residual_share_of_predicted: Option<f64>,
    #[serde(default)]
    predicted_sampler_sync_ms_per_token: Option<f64>,
    #[serde(default)]
    abi_sampling_ms_per_token: Option<f64>,
    #[serde(default)]
    sampler_sync_residual_ms_per_token: Option<f64>,
    #[serde(default)]
    sampler_sync_residual_share_of_predicted: Option<f64>,
    #[serde(default)]
    selected_fit_probe_max_spread_pct: Option<f64>,
    classification: String,
}

#[derive(Debug, Deserialize)]
struct GraphInventoryDiagnostic {
    status: String,
    graph_unclassified_matmul_src0_bytes: u64,
    comparisons: Vec<GraphInventoryComparison>,
}

#[derive(Debug, Deserialize)]
struct GraphInventoryComparison {
    name: String,
    src0_over_metadata: Option<f64>,
    node_count_delta: i64,
}

#[derive(Debug, Deserialize)]
struct RuntimeDiagnostic {
    validation_shape: String,
    selected_backend: String,
    selected_accelerator: Option<String>,
    layer_start: u32,
    layer_end: Option<u32>,
    ctx_size: u32,
    n_gpu_layers: i32,
    cache_type_k: String,
    cache_type_v: String,
    flash_attn_type: String,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    load_mode: String,
}

#[derive(Debug, Deserialize)]
struct ScenarioReport {
    scenario: String,
    predicted: Option<f64>,
    #[serde(default)]
    prediction_source: Option<String>,
    #[serde(default)]
    primary_predicted: Option<f64>,
    #[serde(default)]
    primary_observed_over_fit: Option<f64>,
    prediction_context_tokens: Option<u32>,
    prediction_decode_cost_breakdown: Option<DecodeCostBreakdown>,
    observed: Option<f64>,
    observed_over_fit: Option<f64>,
    first_token_breakdown: Option<FirstTokenBreakdown>,
    verdict: String,
    benchmark: BenchmarkSummary,
}

#[derive(Debug, Deserialize)]
struct FirstTokenBreakdown {
    prompt_token_count: Option<u64>,
    tokenizer_vocab_size: Option<u32>,
    predicted_prefill_ms: Option<f64>,
    predicted_sampler_ms: Option<f64>,
    observed_prefill_ms: Option<f64>,
    observed_decode_ms: Option<f64>,
    observed_sampled_decode_residual_ms: Option<f64>,
    observed_sampled_decode_residual_us_per_prompt_token: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkSummary {
    spread_pct: Option<f64>,
    raw_spread_pct: Option<f64>,
    sample_count: usize,
    #[serde(default)]
    raw_sample_count: usize,
    #[serde(default)]
    request_sample_count: usize,
    #[serde(default)]
    request_spread_pct: Option<f64>,
    #[serde(default)]
    denoised_outlier_count: usize,
}

#[derive(Debug)]
struct ScenarioRow<'a> {
    model: &'a ModelReport,
    scenario: &'a ScenarioReport,
    absolute_error: f64,
    primary_absolute_error: f64,
}

struct DecodeProbeDisplay {
    primary_predicted: Option<f64>,
    scenario_predicted: Option<f64>,
    observed_over_primary: Option<f64>,
    observed_over_scenario: Option<f64>,
    abi_over_primary: Option<f64>,
    abi_over_scenario: Option<f64>,
    observed_vs_primary: String,
    observed_vs_scenario: String,
    abi_vs_primary: String,
    abi_vs_scenario: String,
    scenario_source: String,
}

fn main() -> Result<()> {
    let args = Args::parse()?;
    let bytes = fs::read(&args.report_json)
        .with_context(|| format!("read validation report {}", args.report_json.display()))?;
    let report: ValidationReport =
        serde_json::from_slice(&bytes).context("parse validation report JSON")?;
    let markdown = render_markdown(&args, &report);
    print!("{markdown}");
    if let Some(path) = &args.markdown_out {
        fs::write(path, &markdown)
            .with_context(|| format!("write markdown report {}", path.display()))?;
    }
    if args.report_only {
        return Ok(());
    }
    enforce_thresholds(&args, &report)
}

impl Args {
    fn parse() -> Result<Self> {
        let mut values = env::args().skip(1);
        let mut parsed = Self {
            report_json: PathBuf::new(),
            scenario: DEFAULT_SCENARIO.into(),
            max_median_absolute_error: DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR,
            max_individual_error: DEFAULT_MAX_INDIVIDUAL_ERROR,
            max_noisy: DEFAULT_MAX_NOISY,
            max_runtime_errors: DEFAULT_MAX_RUNTIME_ERRORS,
            min_models: None,
            min_accuracy_models: None,
            markdown_out: None,
            require_graph_inventory_match: false,
            allow_classified_individual_misses: false,
            report_only: false,
        };

        while let Some(arg) = values.next() {
            match arg.as_str() {
                "--scenario" => parsed.scenario = next_value(&mut values, "--scenario")?,
                "--max-median-absolute-error" => {
                    parsed.max_median_absolute_error =
                        parse_next(&mut values, "--max-median-absolute-error")?;
                }
                "--max-individual-error" => {
                    parsed.max_individual_error =
                        parse_next(&mut values, "--max-individual-error")?;
                }
                "--max-noisy" => parsed.max_noisy = parse_next(&mut values, "--max-noisy")?,
                "--max-runtime-errors" => {
                    parsed.max_runtime_errors = parse_next(&mut values, "--max-runtime-errors")?;
                }
                "--min-models" => {
                    parsed.min_models = Some(parse_next(&mut values, "--min-models")?)
                }
                "--min-accuracy-models" => {
                    parsed.min_accuracy_models =
                        Some(parse_next(&mut values, "--min-accuracy-models")?);
                }
                "--markdown-out" => {
                    parsed.markdown_out =
                        Some(PathBuf::from(next_value(&mut values, "--markdown-out")?));
                }
                "--require-graph-inventory-match" => parsed.require_graph_inventory_match = true,
                "--allow-classified-individual-misses" => {
                    parsed.allow_classified_individual_misses = true;
                }
                "--report-only" => parsed.report_only = true,
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other if other.starts_with('-') => bail!("unknown argument {other}"),
                path => {
                    if !parsed.report_json.as_os_str().is_empty() {
                        bail!("only one report JSON path may be provided");
                    }
                    parsed.report_json = PathBuf::from(path);
                }
            }
        }

        if parsed.report_json.as_os_str().is_empty() {
            bail!("provide a validation report JSON path");
        }
        Ok(parsed)
    }
}

fn scenario_rows<'a>(report: &'a ValidationReport, scenario: &str) -> Vec<ScenarioRow<'a>> {
    report
        .models
        .iter()
        .filter_map(|model| {
            let scenario = model
                .benchmarks
                .iter()
                .find(|benchmark| benchmark.scenario == scenario)?;
            let ratio = scenario.observed_over_fit?;
            let primary_ratio = scenario.primary_observed_over_fit.unwrap_or(ratio);
            Some(ScenarioRow {
                model,
                scenario,
                absolute_error: (ratio - 1.0).abs(),
                primary_absolute_error: (primary_ratio - 1.0).abs(),
            })
        })
        .collect()
}

fn render_markdown(args: &Args, report: &ValidationReport) -> String {
    let scenarios = selected_scenarios(args, report);
    let mut markdown = String::new();
    markdown.push_str("# Model Fit Validation\n\n");
    render_decode_probe_diagnostics(report, &mut markdown);
    render_graph_inventory_diagnostics(report, &mut markdown);
    render_decode_cost_breakdowns(report, &scenarios, &mut markdown);
    render_runtime_diagnostics(report, &mut markdown);
    for scenario in scenarios {
        let rows = scenario_rows(report, &scenario);
        markdown.push_str(&render_scenario_markdown(report, &scenario, &rows));
    }
    markdown
}

fn render_graph_inventory_diagnostics(report: &ValidationReport, markdown: &mut String) {
    let rows = report
        .models
        .iter()
        .filter_map(|model| {
            model
                .graph_inventory_diagnostic
                .as_ref()
                .map(|diagnostic| (model, diagnostic))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return;
    }

    markdown.push_str("## Graph Inventory Diagnostics\n\n");
    markdown.push_str("| model | status | unclassified matmul bytes | mismatch comparisons |\n");
    markdown.push_str("|---|---|---:|---|\n");
    for (model, diagnostic) in rows {
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {} |\n",
            model.input_ref,
            diagnostic.status,
            diagnostic.graph_unclassified_matmul_src0_bytes,
            graph_inventory_mismatch_label(diagnostic),
        ));
    }
    markdown.push('\n');
}

fn render_decode_probe_diagnostics(report: &ValidationReport, markdown: &mut String) {
    let rows = report
        .models
        .iter()
        .filter_map(|model| {
            model
                .decode_probe_diagnostic
                .as_ref()
                .map(|diagnostic| (model, diagnostic))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return;
    }

    markdown.push_str("## Decode Probe Diagnostics\n\n");
    markdown.push_str(
        "| model | primary tok/s | scenario tok/s | abi tok/s | observed tok/s | observed/primary | observed/scenario | abi/primary | abi/scenario | observed/abi | fit probe max spread | pred submit ms | abi decode-call ms | submit residual ms | submit residual share | pred sampler ms | abi sampler ms | sampler residual ms | sampler residual share | primary vs observed | scenario vs observed | primary vs abi | scenario vs abi | abi vs observed | source | classification |\n",
    );
    markdown.push_str(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|\n",
    );
    for (model, diagnostic) in rows {
        let display = decode_probe_display(model, diagnostic);
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            model.input_ref,
            number_option(display.primary_predicted),
            number_option(display.scenario_predicted),
            number_option(diagnostic.abi_tokens_per_second),
            number_option(diagnostic.observed_tokens_per_second),
            ratio_option(display.observed_over_primary),
            ratio_option(display.observed_over_scenario),
            ratio_option(display.abi_over_primary),
            ratio_option(display.abi_over_scenario),
            ratio_option(diagnostic.observed_over_abi),
            percent_option(diagnostic.selected_fit_probe_max_spread_pct.map(|value| value / 100.0)),
            number_option(diagnostic.predicted_decode_submission_ms_per_token),
            number_option(diagnostic.abi_decode_call_ms_per_token),
            number_option(diagnostic.decode_submission_residual_ms_per_token),
            ratio_option(diagnostic.decode_submission_residual_share_of_predicted),
            number_option(diagnostic.predicted_sampler_sync_ms_per_token),
            number_option(diagnostic.abi_sampling_ms_per_token),
            number_option(diagnostic.sampler_sync_residual_ms_per_token),
            ratio_option(diagnostic.sampler_sync_residual_share_of_predicted),
            display.observed_vs_primary,
            display.observed_vs_scenario,
            display.abi_vs_primary,
            display.abi_vs_scenario,
            diagnostic.observed_vs_abi,
            display.scenario_source,
            diagnostic.classification,
        ));
    }
    markdown.push('\n');
}

fn decode_probe_display(
    model: &ModelReport,
    diagnostic: &DecodeProbeDiagnostic,
) -> DecodeProbeDisplay {
    let steady = scenario_report_by_name(model, "steady_decode");
    let report_has_primary_diagnostic = diagnostic.scenario_predicted_tokens_per_second.is_some()
        || diagnostic.observed_over_scenario_fit.is_some()
        || diagnostic.abi_over_scenario_fit.is_some();
    let primary_predicted = if report_has_primary_diagnostic {
        diagnostic.predicted_tokens_per_second
    } else {
        steady
            .and_then(|scenario| scenario.primary_predicted)
            .or(diagnostic.predicted_tokens_per_second)
    };
    let scenario_predicted = diagnostic
        .scenario_predicted_tokens_per_second
        .or_else(|| steady.and_then(|scenario| scenario.predicted));
    let observed_over_primary = if report_has_primary_diagnostic {
        diagnostic.observed_over_fit
    } else {
        steady
            .and_then(|scenario| scenario.primary_observed_over_fit)
            .or_else(|| ratio(diagnostic.observed_tokens_per_second, primary_predicted))
    };
    let observed_over_scenario = diagnostic
        .observed_over_scenario_fit
        .or_else(|| steady.and_then(|scenario| scenario.observed_over_fit))
        .or_else(|| ratio(diagnostic.observed_tokens_per_second, scenario_predicted));
    let abi_over_primary = if report_has_primary_diagnostic {
        diagnostic.abi_over_fit
    } else {
        ratio(diagnostic.abi_tokens_per_second, primary_predicted)
    };
    let abi_over_scenario = diagnostic
        .abi_over_scenario_fit
        .or_else(|| ratio(diagnostic.abi_tokens_per_second, scenario_predicted));
    DecodeProbeDisplay {
        primary_predicted,
        scenario_predicted,
        observed_over_primary,
        observed_over_scenario,
        abi_over_primary,
        abi_over_scenario,
        observed_vs_primary: if report_has_primary_diagnostic {
            diagnostic.observed_vs_fit.clone()
        } else {
            throughput_ratio_verdict(observed_over_primary)
        },
        observed_vs_scenario: diagnostic
            .observed_vs_scenario_fit
            .clone()
            .unwrap_or_else(|| throughput_ratio_verdict(observed_over_scenario)),
        abi_vs_primary: if report_has_primary_diagnostic {
            diagnostic.abi_vs_fit.clone()
        } else {
            throughput_ratio_verdict(abi_over_primary)
        },
        abi_vs_scenario: diagnostic
            .abi_vs_scenario_fit
            .clone()
            .unwrap_or_else(|| throughput_ratio_verdict(abi_over_scenario)),
        scenario_source: diagnostic
            .scenario_prediction_source
            .clone()
            .or_else(|| steady.and_then(|scenario| scenario.prediction_source.clone()))
            .unwrap_or_else(|| "-".into()),
    }
}

fn scenario_report_by_name<'a>(model: &'a ModelReport, name: &str) -> Option<&'a ScenarioReport> {
    model
        .benchmarks
        .iter()
        .find(|scenario| scenario.scenario == name)
}

fn render_decode_cost_breakdowns(
    report: &ValidationReport,
    scenarios: &[String],
    markdown: &mut String,
) {
    let rows = report
        .models
        .iter()
        .filter_map(|model| decode_cost_breakdown_for_report(model, scenarios))
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return;
    }

    markdown.push_str("## Decode Cost Breakdown\n\n");
    markdown.push_str("| model | scope | ctx | est tok/s | selected ms | bandwidth ms | compute ms | runtime overhead ms | sampler ms | probed MiB | fallback MiB |\n");
    markdown.push_str("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in &rows {
        let overhead_ms = row.breakdown.fixed_overhead_ms
            + row.breakdown.runtime_overhead_ms
            + row.breakdown.measured_graph_overhead_ms
            + row.breakdown.architecture_overhead_ms;
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {:.1} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} |\n",
            row.model.input_ref,
            row.scope,
            row.context_tokens
                .map_or_else(|| "-".into(), |tokens| tokens.to_string()),
            row.breakdown.estimated_tokens_per_sec,
            row.breakdown.selected_time_ms,
            row.breakdown.bandwidth_ms,
            row.breakdown.compute_ms,
            overhead_ms,
            row.breakdown.sampled_decode_sampler_ms,
            mib(row.breakdown.probed_bytes),
            mib(row.breakdown.fallback_bytes),
        ));
    }
    markdown.push('\n');

    markdown.push_str("| model | scope | group | type | source | traffic MiB | ms | bandwidth GB/s | probe | distance |\n");
    markdown.push_str("|---|---|---|---|---|---:|---:|---:|---|---:|\n");
    for row in rows {
        for group in &row.breakdown.groups {
            markdown.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {:.1} | {:.3} | {:.1} | {} | {} |\n",
                row.model.input_ref,
                row.scope,
                group.group,
                group.tensor_type,
                group.source,
                mib(group.traffic_bytes),
                group.bandwidth_ms,
                group.bandwidth_bytes_per_sec as f64 / 1_000_000_000.0,
                group.probe_name.as_deref().unwrap_or("-"),
                group
                    .probe_shape_distance
                    .map_or_else(|| "-".into(), |distance| format!("{distance:.3}")),
            ));
        }
    }
    markdown.push('\n');
}

struct DecodeCostBreakdownRow<'a> {
    model: &'a ModelReport,
    scope: String,
    context_tokens: Option<u32>,
    breakdown: &'a DecodeCostBreakdown,
}

fn decode_cost_breakdown_for_report<'a>(
    model: &'a ModelReport,
    scenarios: &[String],
) -> Option<DecodeCostBreakdownRow<'a>> {
    for scenario_name in scenarios {
        let Some(scenario) = model
            .benchmarks
            .iter()
            .find(|benchmark| &benchmark.scenario == scenario_name)
        else {
            continue;
        };
        if let Some(breakdown) = scenario.prediction_decode_cost_breakdown.as_ref() {
            return Some(DecodeCostBreakdownRow {
                model,
                scope: scenario.scenario.clone(),
                context_tokens: scenario
                    .prediction_context_tokens
                    .or(breakdown.context_tokens),
                breakdown,
            });
        }
    }
    model
        .recommendation
        .as_ref()
        .and_then(|recommendation| recommendation.decode_cost_breakdown.as_ref())
        .map(|breakdown| DecodeCostBreakdownRow {
            model,
            scope: "default".into(),
            context_tokens: model
                .recommendation
                .as_ref()
                .and_then(|recommendation| recommendation.estimated_decode_context_tokens)
                .or(breakdown.context_tokens),
            breakdown,
        })
}

fn render_runtime_diagnostics(report: &ValidationReport, markdown: &mut String) {
    let rows = report
        .models
        .iter()
        .filter_map(|model| {
            model
                .runtime_diagnostic
                .as_ref()
                .map(|diagnostic| (model, diagnostic))
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return;
    }

    markdown.push_str("## Runtime Diagnostics\n\n");
    markdown.push_str(
        "| model | backend | accelerator | shape | layers | ctx | n_gpu_layers | kv | flash | batch | ubatch | load |\n",
    );
    markdown.push_str("|---|---|---|---|---:|---:|---:|---|---|---:|---:|---|\n");
    for (model, diagnostic) in rows {
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {} | {}-{} | {} | {} | {}/{} | {} | {} | {} | {} |\n",
            model.input_ref,
            diagnostic.selected_backend,
            diagnostic.selected_accelerator.as_deref().unwrap_or("-"),
            diagnostic.validation_shape,
            diagnostic.layer_start,
            diagnostic
                .layer_end
                .map_or_else(|| "-".into(), |value| value.to_string()),
            diagnostic.ctx_size,
            diagnostic.n_gpu_layers,
            diagnostic.cache_type_k,
            diagnostic.cache_type_v,
            diagnostic.flash_attn_type,
            integer_option(diagnostic.n_batch.map(u64::from)),
            integer_option(diagnostic.n_ubatch.map(u64::from)),
            diagnostic.load_mode,
        ));
    }
    markdown.push('\n');
}

fn render_scenario_markdown(
    report: &ValidationReport,
    scenario: &str,
    rows: &[ScenarioRow<'_>],
) -> String {
    let accuracy_rows = accuracy_rows(rows);
    let accuracy_excluded = accuracy_excluded_count(rows);
    let median_error = median_absolute_error(&accuracy_rows);
    let primary_accuracy_rows = primary_fit_accuracy_rows(&accuracy_rows);
    let primary_median_error = primary_median_absolute_error(&primary_accuracy_rows);
    let noisy = noisy_count(rows);
    let runtime_errors = runtime_error_count(report, scenario);
    let mut markdown = String::new();
    markdown.push_str(&format!("Scenario: `{scenario}`\n\n"));
    markdown.push_str("| metric | value |\n|---|---:|\n");
    markdown.push_str(&format!("| models in report | {} |\n", report.models.len()));
    markdown.push_str(&format!("| scenario samples | {} |\n", rows.len()));
    markdown.push_str(&format!(
        "| accuracy-gated samples | {} |\n",
        accuracy_rows.len()
    ));
    markdown.push_str(&format!(
        "| accuracy-excluded diagnostic samples | {accuracy_excluded} |\n"
    ));
    if let Some(summary) = &report.summary {
        markdown.push_str(&format!("| report errors | {} |\n", summary.error_count));
        markdown.push_str(&format!(
            "| report runtime errors | {} |\n",
            summary.runtime_error_count
        ));
    }
    markdown.push_str(&format!("| noisy samples | {noisy} |\n"));
    markdown.push_str(&format!("| runtime-error samples | {runtime_errors} |\n"));
    if scenario == "steady_decode" {
        markdown.push_str(&format!(
            "| primary median absolute error | {} |\n",
            percent_option(primary_median_error)
        ));
    }
    markdown.push_str(&format!(
        "| median absolute error | {} |\n\n",
        percent_option(median_error)
    ));
    if scenario == "first_token" {
        render_first_token_rows(rows, &mut markdown);
    } else {
        render_standard_rows(rows, &mut markdown);
    }
    markdown
}

fn render_standard_rows(rows: &[ScenarioRow<'_>], markdown: &mut String) {
    markdown.push_str("| model | predicted | pred ctx | primary predicted | primary ctx | observed | observed/fit | observed/primary | source | spread | raw spread | request spread | samples | requests | outliers | verdict |\n");
    markdown.push_str(
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for row in rows {
        markdown.push_str(&standard_row_markdown(row));
    }
}

fn standard_row_markdown(row: &ScenarioRow<'_>) -> String {
    format!(
        "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
        row.model.input_ref,
        number_option(row.scenario.predicted),
        integer_option(row.scenario.prediction_context_tokens.map(u64::from)),
        number_option(row.scenario.primary_predicted),
        integer_option(
            row.model
                .recommendation
                .as_ref()
                .and_then(|recommendation| recommendation.estimated_decode_context_tokens)
                .map(u64::from)
        ),
        number_option(row.scenario.observed),
        ratio_option(row.scenario.observed_over_fit),
        ratio_option(row.scenario.primary_observed_over_fit),
        row.scenario.prediction_source.as_deref().unwrap_or("-"),
        percent_option(row.scenario.benchmark.spread_pct.map(|value| value / 100.0)),
        percent_option(
            row.scenario
                .benchmark
                .raw_spread_pct
                .map(|value| value / 100.0)
        ),
        percent_option(
            row.scenario
                .benchmark
                .request_spread_pct
                .map(|value| value / 100.0)
        ),
        sample_count_label(&row.scenario.benchmark),
        request_sample_count_label(&row.scenario.benchmark),
        row.scenario.benchmark.denoised_outlier_count,
        row.scenario.verdict
    )
}

fn render_first_token_rows(rows: &[ScenarioRow<'_>], markdown: &mut String) {
    markdown.push_str("| model | predicted ms | observed ms | observed/fit | prompt toks | vocab | pred prefill | pred sampler | obs prefill | obs decode | sampled residual | residual us/tok | spread | samples | verdict |\n");
    markdown
        .push_str("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n");
    for row in rows {
        let breakdown = row.scenario.first_token_breakdown.as_ref();
        markdown.push_str(&format!(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.model.input_ref,
            number_option(row.scenario.predicted),
            number_option(row.scenario.observed),
            ratio_option(row.scenario.observed_over_fit),
            integer_option(breakdown.and_then(|value| value.prompt_token_count)),
            integer_option(breakdown.and_then(|value| value.tokenizer_vocab_size.map(u64::from))),
            number_option(breakdown.and_then(|value| value.predicted_prefill_ms)),
            number_option(breakdown.and_then(|value| value.predicted_sampler_ms)),
            number_option(breakdown.and_then(|value| value.observed_prefill_ms)),
            number_option(breakdown.and_then(|value| value.observed_decode_ms)),
            number_option(breakdown.and_then(|value| value.observed_sampled_decode_residual_ms)),
            number_option(
                breakdown
                    .and_then(|value| value.observed_sampled_decode_residual_us_per_prompt_token)
            ),
            percent_option(row.scenario.benchmark.spread_pct.map(|value| value / 100.0)),
            sample_count_label(&row.scenario.benchmark),
            row.scenario.verdict
        ));
    }
}

fn enforce_thresholds(args: &Args, report: &ValidationReport) -> Result<()> {
    let mut failures = Vec::new();
    for model in &report.models {
        if !model.errors.is_empty() {
            failures.push(format!(
                "{} has model errors: {}",
                model.input_ref,
                model.errors.join("; ")
            ));
        }
        if args.require_graph_inventory_match {
            enforce_graph_inventory_match(model, &mut failures);
        }
    }

    for scenario in selected_scenarios(args, report) {
        let rows = scenario_rows(report, &scenario);
        enforce_scenario_thresholds(args, &scenario, report, &rows, &mut failures);
    }

    if failures.is_empty() {
        return Ok(());
    }
    bail!("model-fit validation failed:\n{}", failures.join("\n"))
}

fn enforce_scenario_thresholds(
    args: &Args,
    scenario: &str,
    report: &ValidationReport,
    rows: &[ScenarioRow<'_>],
    failures: &mut Vec<String>,
) {
    let accuracy_rows = accuracy_rows(rows);
    match args.min_models {
        Some(min_models) if rows.len() < min_models => {
            failures.push(format!(
                "scenario {scenario} produced {} samples, expected at least {min_models}",
                rows.len()
            ));
        }
        _ => {}
    }
    let noisy = noisy_count(rows);
    if noisy > args.max_noisy {
        failures.push(format!(
            "scenario {scenario} had {noisy} noisy samples, max allowed is {}",
            args.max_noisy
        ));
    }
    let runtime_errors = runtime_error_count(report, scenario);
    if runtime_errors > args.max_runtime_errors {
        failures.push(format!(
            "scenario {scenario} had {runtime_errors} runtime-error samples, max allowed is {}",
            args.max_runtime_errors
        ));
    }
    let gate_uses_primary_fit = scenario == "steady_decode";
    let primary_accuracy_rows = primary_fit_accuracy_rows(&accuracy_rows);
    let gate_rows = if gate_uses_primary_fit {
        primary_accuracy_rows.as_slice()
    } else {
        accuracy_rows.as_slice()
    };
    enforce_accuracy_row_floor(
        scenario,
        gate_rows.len(),
        args.min_accuracy_models
            .unwrap_or(DEFAULT_MIN_ACCURACY_MODELS),
        failures,
    );
    match gate_median_absolute_error(scenario, gate_rows) {
        Some(median_error) if median_error > args.max_median_absolute_error => {
            let label = if gate_uses_primary_fit {
                "primary-fit"
            } else {
                "scenario-fit"
            };
            failures.push(format!(
                "scenario {scenario} {label} median absolute error {:.2}% exceeded {:.2}%",
                median_error * 100.0,
                args.max_median_absolute_error * 100.0
            ));
        }
        _ => {}
    }
    for row in gate_rows {
        let row_error = gate_absolute_error(scenario, row);
        if row_error > args.max_individual_error {
            if args.allow_classified_individual_misses && row_has_classified_miss(row) {
                continue;
            }
            let label = if gate_uses_primary_fit {
                "primary-fit"
            } else {
                "scenario-fit"
            };
            failures.push(format!(
                "{} scenario {scenario} {label} error {:.2}% exceeded {:.2}%",
                row.model.input_ref,
                row_error * 100.0,
                args.max_individual_error * 100.0
            ));
        }
    }
}

fn enforce_accuracy_row_floor(
    scenario: &str,
    actual: usize,
    required: usize,
    failures: &mut Vec<String>,
) {
    if actual < required {
        failures.push(format!(
            "scenario {scenario} produced {actual} accuracy-gated samples, expected at least {required}; diagnostic-only rows do not prove tok/s accuracy"
        ));
    }
}

fn enforce_graph_inventory_match(model: &ModelReport, failures: &mut Vec<String>) {
    if !model_has_observed_decode_validation(model) {
        return;
    }
    let Some(diagnostic) = &model.graph_inventory_diagnostic else {
        failures.push(format!(
            "{} is missing graph inventory diagnostics",
            model.input_ref
        ));
        return;
    };
    if !graph_inventory_status_is_match(&diagnostic.status) {
        failures.push(format!(
            "{} graph inventory status is {}, expected metadata inventory match",
            model.input_ref, diagnostic.status
        ));
    }
    if diagnostic.graph_unclassified_matmul_src0_bytes > 0 {
        failures.push(format!(
            "{} graph inventory has {} unclassified matmul src0 bytes",
            model.input_ref, diagnostic.graph_unclassified_matmul_src0_bytes
        ));
    }
    for comparison in &diagnostic.comparisons {
        let mismatched_bytes = comparison
            .src0_over_metadata
            .is_some_and(|ratio| (ratio - 1.0).abs() > DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR);
        if mismatched_bytes || comparison.node_count_delta != 0 {
            failures.push(format!(
                "{} graph inventory comparison {} mismatched: src0/meta={} node_delta={}",
                model.input_ref,
                comparison.name,
                ratio_option(comparison.src0_over_metadata),
                comparison.node_count_delta
            ));
        }
    }
}

fn model_has_observed_decode_validation(model: &ModelReport) -> bool {
    model
        .benchmarks
        .iter()
        .any(|scenario| scenario.observed.is_some())
        || model
            .decode_probe_diagnostic
            .as_ref()
            .and_then(|diagnostic| diagnostic.observed_tokens_per_second)
            .is_some()
}

fn graph_inventory_status_is_match(status: &str) -> bool {
    matches!(
        status,
        "metadata_inventory_matches" | "metadata_inventory_matches_probe_depth_risk"
    )
}

fn graph_inventory_mismatch_label(diagnostic: &GraphInventoryDiagnostic) -> String {
    let mismatches = diagnostic
        .comparisons
        .iter()
        .filter(|comparison| {
            comparison
                .src0_over_metadata
                .is_some_and(|ratio| (ratio - 1.0).abs() > DEFAULT_MAX_MEDIAN_ABSOLUTE_ERROR)
                || comparison.node_count_delta != 0
        })
        .map(|comparison| comparison.name.as_str())
        .collect::<Vec<_>>();
    if mismatches.is_empty() {
        "-".into()
    } else {
        mismatches.join(", ")
    }
}

fn row_has_classified_miss(row: &ScenarioRow<'_>) -> bool {
    row.model
        .decode_probe_diagnostic
        .as_ref()
        .is_some_and(|diagnostic| classified_miss(&diagnostic.classification))
}

fn row_is_accuracy_excluded(row: &ScenarioRow<'_>) -> bool {
    row.model
        .decode_probe_diagnostic
        .as_ref()
        .is_some_and(|diagnostic| accuracy_excluded_classification(&diagnostic.classification))
}

fn accuracy_excluded_classification(classification: &str) -> bool {
    matches!(
        classification,
        "steady_path_overhead_mismatch"
            | "steady_path_overhead_mismatch_noisy"
            | "sampler_sync_residual"
            | "sampler_sync_residual_noisy"
            | "llama_source_boundary_residual"
            | "llama_source_boundary_residual_noisy"
            | "decode_submission_residual"
            | "decode_submission_residual_noisy"
            | "decode_and_sampler_residual"
            | "decode_and_sampler_residual_noisy"
            | "estimate_and_probe_agree_noisy_requests"
            | "primary_estimate_and_probe_agree_context_rescore_unstable"
            | "scenario_estimate_and_probe_agree_noisy_requests"
            | "runtime_path_mismatch"
            | "mixed_estimate_and_runtime_mismatch"
            | "context_rescore_unstable"
            | "abi_probe_noisy"
            | "validation_request_noise"
            | "missing_representative_decode_probe"
            | "probe_not_representative"
            | "unstable_probe_geometry"
    )
}

fn classified_miss(classification: &str) -> bool {
    matches!(
        classification,
        "metadata_estimate_miss"
            | "runtime_path_mismatch"
            | "steady_path_overhead_mismatch_noisy"
            | "sampler_sync_residual"
            | "sampler_sync_residual_noisy"
            | "llama_source_boundary_residual"
            | "llama_source_boundary_residual_noisy"
            | "decode_submission_residual"
            | "decode_submission_residual_noisy"
            | "decode_and_sampler_residual"
            | "decode_and_sampler_residual_noisy"
            | "estimate_and_probe_agree_noisy_requests"
            | "scenario_estimate_and_probe_agree_noisy_requests"
            | "context_rescore_unstable"
            | "probe_not_representative"
            | "probe_differs_but_observed_matches_fit"
            | "mixed_estimate_and_runtime_mismatch"
            | "abi_probe_noisy"
            | "validation_request_noise"
            | "missing_representative_decode_probe"
            | "no_abi_probe"
    )
}

fn row_has_primary_context_mismatch(row: &ScenarioRow<'_>) -> bool {
    row.model
        .decode_probe_diagnostic
        .as_ref()
        .is_some_and(|diagnostic| {
            diagnostic.classification == "primary_context_differs_from_benchmark"
        })
}

fn selected_scenarios(args: &Args, report: &ValidationReport) -> Vec<String> {
    if args.scenario != "all" {
        return vec![args.scenario.clone()];
    }
    let mut scenarios = report
        .models
        .iter()
        .flat_map(|model| {
            model
                .benchmarks
                .iter()
                .map(|benchmark| benchmark.scenario.clone())
        })
        .collect::<Vec<_>>();
    scenarios.sort();
    scenarios.dedup();
    scenarios
}

fn accuracy_rows<'a>(rows: &'a [ScenarioRow<'a>]) -> Vec<&'a ScenarioRow<'a>> {
    rows.iter()
        .filter(|row| accuracy_gated_verdict(&row.scenario.verdict))
        .filter(|row| !row_is_accuracy_excluded(row))
        .collect()
}

fn primary_fit_accuracy_rows<'a>(rows: &[&'a ScenarioRow<'a>]) -> Vec<&'a ScenarioRow<'a>> {
    rows.iter()
        .copied()
        .filter(|row| !row_has_primary_context_mismatch(row))
        .collect()
}

fn accuracy_excluded_count(rows: &[ScenarioRow<'_>]) -> usize {
    rows.iter()
        .filter(|row| accuracy_gated_verdict(&row.scenario.verdict))
        .filter(|row| row_is_accuracy_excluded(row))
        .count()
}

fn accuracy_gated_verdict(verdict: &str) -> bool {
    matches!(verdict, "match" | "slower-than-fit" | "faster-than-fit")
}

fn runtime_error_count(report: &ValidationReport, scenario: &str) -> usize {
    report
        .models
        .iter()
        .filter_map(|model| {
            model
                .benchmarks
                .iter()
                .find(|benchmark| benchmark.scenario == scenario)
        })
        .filter(|scenario| scenario.verdict == "runtime-error" || scenario.verdict == "error")
        .count()
}

fn noisy_count(rows: &[ScenarioRow<'_>]) -> usize {
    rows.iter()
        .filter(|row| row.scenario.verdict == "inconclusive-noisy")
        .count()
}

fn median_absolute_error(rows: &[&ScenarioRow<'_>]) -> Option<f64> {
    median_error_by(rows, |row| row.absolute_error)
}

fn primary_median_absolute_error(rows: &[&ScenarioRow<'_>]) -> Option<f64> {
    median_error_by(rows, |row| row.primary_absolute_error)
}

fn gate_median_absolute_error(scenario: &str, rows: &[&ScenarioRow<'_>]) -> Option<f64> {
    median_error_by(rows, |row| gate_absolute_error(scenario, row))
}

fn gate_absolute_error(scenario: &str, row: &ScenarioRow<'_>) -> f64 {
    if scenario == "steady_decode" {
        row.primary_absolute_error
    } else {
        row.absolute_error
    }
}

fn median_error_by(
    rows: &[&ScenarioRow<'_>],
    error: impl Fn(&ScenarioRow<'_>) -> f64,
) -> Option<f64> {
    let mut samples = rows.iter().map(|row| error(row)).collect::<Vec<_>>();
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mid = samples.len() / 2;
    Some(if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    })
}

fn sample_count_label(summary: &BenchmarkSummary) -> String {
    if summary.raw_sample_count > 0 && summary.raw_sample_count != summary.sample_count {
        format!("{}/{}", summary.sample_count, summary.raw_sample_count)
    } else {
        summary.sample_count.to_string()
    }
}

fn request_sample_count_label(summary: &BenchmarkSummary) -> String {
    if summary.request_sample_count > 0 {
        summary.request_sample_count.to_string()
    } else {
        "-".into()
    }
}

fn number_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.1}"))
}

fn integer_option(value: Option<u64>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

fn ratio_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.3}"))
}

fn ratio(numerator: Option<f64>, denominator: Option<f64>) -> Option<f64> {
    match (numerator, denominator) {
        (Some(numerator), Some(denominator)) if denominator > 0.0 => Some(numerator / denominator),
        _ => None,
    }
}

fn throughput_ratio_verdict(ratio: Option<f64>) -> String {
    let Some(ratio) = ratio else {
        return "missing".into();
    };
    if (ratio - 1.0).abs() <= DEFAULT_MAX_INDIVIDUAL_ERROR {
        "match".into()
    } else if ratio < 1.0 {
        "slower-than-reference".into()
    } else {
        "faster-than-reference".into()
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn percent_option(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{:.1}%", value * 100.0))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_value(args, name)?
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid {name}: {err}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{name} requires a value"))
}

fn print_usage() {
    eprintln!(
        "usage: model-fit-check-validation [--scenario steady_decode|prefill|first_token|kv_warm_reuse|all] [--max-median-absolute-error 0.10] [--max-individual-error 0.10] [--max-noisy 0] [--max-runtime-errors 0] [--min-models N] [--min-accuracy-models N] [--markdown-out report.md] [--require-graph-inventory-match] [--allow-classified-individual-misses] [--report-only] report.json"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accuracy_row_floor_rejects_zero_evidence_gate() {
        let mut failures = Vec::new();

        enforce_accuracy_row_floor("steady_decode", 0, 1, &mut failures);

        assert_eq!(
            failures,
            vec![
                "scenario steady_decode produced 0 accuracy-gated samples, expected at least 1; diagnostic-only rows do not prove tok/s accuracy"
            ]
        );
    }

    #[test]
    fn accuracy_row_floor_can_be_disabled_for_diagnostic_runs() {
        let mut failures = Vec::new();

        enforce_accuracy_row_floor("steady_decode", 0, 0, &mut failures);

        assert!(failures.is_empty());
    }
}
