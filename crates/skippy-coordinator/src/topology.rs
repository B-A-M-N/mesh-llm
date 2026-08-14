use std::cmp::Ordering;
use std::path::PathBuf;

mod locked;

pub use locked::{LockedTopologyStage, plan_locked_topology};

/// Default auto lane cap.  Matches llama-server's default of `--parallel 4`.
/// Users can override via `gpu.parallel` in config.toml or the per-model
/// `parallel` setting.
const MAX_AUTO_PARALLEL_LANES: usize = 4;
const MINIMUM_AUTO_CONTEXT_LENGTH: u32 = 65_536;
const CONTEXT_STEPS: &[u32] = &[512, 1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072];

/// Compute-buffer reserve applied to the KV term of each layer's placement
/// cost. Charging KV at 100/85 holds back 15% of a node's post-weight space for
/// llama.cpp compute-graph buffers and scratch — algebraically identical to the
/// single-node context planner's `usable_kv_cache_budget`, which grants KV 85%
/// of post-weight space (`context_planning.rs`). Without this, placement packed
/// a node with `weights + KV` alone and left the decode's transient buffers
/// nowhere to go, OOM-ing the stage or swapping the host. Because the reserve
/// rides on the KV term it scales with context length, matching how compute
/// buffers grow with `n_ctx`. A fixed per-node floor (see the coordinator's
/// node headroom) covers the context-independent minimum on top of this.
const KV_COMPUTE_RESERVE_NUMERATOR: u128 = 100;
const KV_COMPUTE_RESERVE_DENOMINATOR: u128 = 85;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlanningInput {
    pub native_context_length: u32,
    pub layer_count: u32,
    pub model_weight_bytes: u64,
    pub layer_weight_bytes: Vec<u64>,
    pub kv_bytes_per_token: u64,
    pub minimum_nodes: usize,
    pub nodes: Vec<TopologyNode>,
    pub context_length_override: Option<u32>,
    pub parallel_lanes_override: Option<usize>,
    pub target_decode_tpot_ms: Option<u32>,
    pub layer_class_bytes: Vec<TopologyLayerClassBytes>,
}

/// Per-layer class byte breakdown (mirrors skippy_runtime::policy::TensorClassBytes
/// but defined locally to avoid a circular dependency with skippy-runtime).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TopologyLayerClassBytes {
    pub routed_expert: u64,
    pub shared_expert: u64,
    pub attention: u64,
    pub recurrent_ssm: u64,
    pub routing_gate: u64,
    pub normalization: u64,
    pub other: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyNode {
    pub node_id: String,
    pub detected_vram_bytes: u64,
    pub detected_host_available_bytes: u64,
    pub max_vram_bytes: Option<u64>,
    pub runtime_headroom_bytes: u64,
    pub stage_transfer_latency_ms: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyPlan {
    pub context_length: u32,
    pub parallel_lanes: usize,
    pub stages: Vec<TopologyStagePlan>,
    pub estimated_decode_network_ms_per_token: Option<u32>,
    pub decode_tpot_target_met: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologyStagePlan {
    pub stage_id: String,
    pub stage_index: u32,
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub parameter_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum TopologyPlanError {
    #[error("topology planning requires native GGUF context length")]
    MissingNativeContext,
    #[error("topology planning requires at least one model layer")]
    MissingLayers,
    #[error("topology planning requires model weight bytes")]
    MissingModelWeights,
    #[error("topology planning requires KV bytes per token")]
    MissingKvBytesPerToken,
    #[error("topology planning requires at least one node")]
    MissingNodes,
    #[error("requested context {requested} is below minimum valid context {minimum}")]
    ContextBelowMinimum { requested: u32, minimum: u32 },
    #[error("requested context {requested} exceeds native context {native}")]
    ContextExceedsNative { requested: u32, native: u32 },
    #[error("requested parallel lanes must be greater than zero")]
    ZeroParallelLanes,
    #[error("no topology can distribute all layers and keep context >= {minimum_context}")]
    NoValidTopology { minimum_context: u32 },
    #[error("locked topology must contain at least {minimum} stages; found {actual}")]
    LockedStageCount { minimum: usize, actual: usize },
    #[error("locked topology references unknown node {node_id}")]
    LockedUnknownNode { node_id: String },
    #[error("locked topology assigns node {node_id} more than once")]
    LockedDuplicateNode { node_id: String },
    #[error(
        "locked topology stage {stage_index} must start at layer {expected_start}; found {actual_start}"
    )]
    LockedNonContiguousRange {
        stage_index: usize,
        expected_start: u32,
        actual_start: u32,
    },
    #[error("locked topology stage {stage_index} has empty or reversed range {start}..{end}")]
    LockedInvalidRange {
        stage_index: usize,
        start: u32,
        end: u32,
    },
    #[error("locked topology ends at layer {actual_end}; model has {layer_count} layers")]
    LockedIncompleteCoverage { actual_end: u32, layer_count: u32 },
    #[error("locked topology cannot fit context >= {minimum_context}")]
    LockedTopologyDoesNotFit { minimum_context: u32 },
}

pub fn plan_topology(input: &TopologyPlanningInput) -> Result<TopologyPlan, TopologyPlanError> {
    plan_topology_with_required_stage0(input, None)
}

pub fn plan_topology_with_stage0(
    input: &TopologyPlanningInput,
    stage0_node_id: &str,
) -> Result<TopologyPlan, TopologyPlanError> {
    plan_topology_with_required_stage0(input, Some(stage0_node_id))
}

fn plan_topology_with_required_stage0(
    input: &TopologyPlanningInput,
    required_stage0_node_id: Option<&str>,
) -> Result<TopologyPlan, TopologyPlanError> {
    validate_input(input)?;

    let minimum_context = minimum_valid_context(input.native_context_length);
    let context_candidates = context_candidates(
        input.native_context_length,
        minimum_context,
        input.context_length_override,
    )?;
    let lane_candidates = parallel_lane_candidates(input.parallel_lanes_override)?;
    let nodes = usable_nodes(&input.nodes);
    let latency_aware = latency_aware_planning(input, &nodes);

    let minimum_nodes = input.minimum_nodes.max(1);
    let mut best_latency_candidate: Option<CandidatePlan> = None;
    for context_length in context_candidates {
        for node_count in minimum_nodes..=nodes.len().min(input.layer_count as usize) {
            for parallel_lanes in lane_candidates.iter().copied() {
                let mut best_for_count: Option<CandidatePlan> = None;
                for_each_node_subset(&nodes, node_count, |subset| {
                    let Some(candidate) =
                        fit_candidate(input, subset, context_length, parallel_lanes)
                    else {
                        return;
                    };
                    if !candidate_has_required_stage0(&candidate, required_stage0_node_id) {
                        return;
                    }
                    if best_for_count
                        .as_ref()
                        .is_none_or(|current| candidate_better_for_same_shape(&candidate, current))
                    {
                        best_for_count = Some(candidate);
                    }
                });
                if let Some(candidate) = best_for_count {
                    if latency_aware {
                        if best_latency_candidate.as_ref().is_none_or(|current| {
                            latency_candidate_better(&candidate, current, input)
                        }) {
                            best_latency_candidate = Some(candidate);
                        }
                        continue;
                    }
                    return Ok(candidate.plan);
                }
            }
        }
    }

    if let Some(candidate) = best_latency_candidate {
        return Ok(candidate.plan);
    }

    Err(TopologyPlanError::NoValidTopology { minimum_context })
}

fn validate_input(input: &TopologyPlanningInput) -> Result<(), TopologyPlanError> {
    if input.native_context_length == 0 {
        return Err(TopologyPlanError::MissingNativeContext);
    }
    if input.layer_count == 0 {
        return Err(TopologyPlanError::MissingLayers);
    }
    if input.model_weight_bytes == 0 {
        return Err(TopologyPlanError::MissingModelWeights);
    }
    if input.kv_bytes_per_token == 0 {
        return Err(TopologyPlanError::MissingKvBytesPerToken);
    }
    if input.nodes.is_empty() {
        return Err(TopologyPlanError::MissingNodes);
    }
    Ok(())
}

fn context_candidates(
    native_context: u32,
    minimum_context: u32,
    override_context: Option<u32>,
) -> Result<Vec<u32>, TopologyPlanError> {
    if let Some(requested) = override_context {
        if requested > native_context {
            return Err(TopologyPlanError::ContextExceedsNative {
                requested,
                native: native_context,
            });
        }
        return Ok(vec![requested]);
    }

    let mut candidates = CONTEXT_STEPS
        .iter()
        .copied()
        .filter(|context| *context >= minimum_context && *context <= native_context)
        .collect::<Vec<_>>();
    candidates.push(native_context);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.reverse();
    Ok(candidates)
}

fn parallel_lane_candidates(
    override_lanes: Option<usize>,
) -> Result<Vec<usize>, TopologyPlanError> {
    if let Some(lanes) = override_lanes {
        if lanes == 0 {
            return Err(TopologyPlanError::ZeroParallelLanes);
        }
        return Ok(vec![lanes]);
    }
    Ok((1..=MAX_AUTO_PARALLEL_LANES).rev().collect())
}

pub fn minimum_valid_context(native_context: u32) -> u32 {
    native_context.clamp(1, MINIMUM_AUTO_CONTEXT_LENGTH)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct UsableNode {
    node_id: String,
    usable_vram_bytes: u64,
    usable_ram_bytes: u64,
    stage_transfer_latency_ms: Option<u32>,
}

fn usable_nodes(nodes: &[TopologyNode]) -> Vec<UsableNode> {
    let mut nodes = nodes
        .iter()
        .map(|node| {
            let capped = node
                .max_vram_bytes
                .map(|max| node.detected_vram_bytes.min(max))
                .unwrap_or(node.detected_vram_bytes);
            UsableNode {
                node_id: node.node_id.clone(),
                usable_vram_bytes: capped.saturating_sub(node.runtime_headroom_bytes),
                usable_ram_bytes: node.detected_host_available_bytes.saturating_sub(node.runtime_headroom_bytes),
                stage_transfer_latency_ms: node.stage_transfer_latency_ms,
            }
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| {
        right
            .usable_vram_bytes
            .cmp(&left.usable_vram_bytes)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    nodes
}

fn for_each_node_subset(nodes: &[UsableNode], count: usize, mut visit: impl FnMut(&[UsableNode])) {
    let mut current = Vec::with_capacity(count);
    visit_node_subsets(nodes, count, 0, &mut current, &mut visit);
}

fn visit_node_subsets(
    nodes: &[UsableNode],
    count: usize,
    start: usize,
    current: &mut Vec<UsableNode>,
    visit: &mut impl FnMut(&[UsableNode]),
) {
    if current.len() == count {
        visit(current);
        return;
    }
    let needed = count - current.len();
    if nodes.len().saturating_sub(start) < needed {
        return;
    }
    for index in start..=nodes.len() - needed {
        current.push(nodes[index].clone());
        visit_node_subsets(nodes, count, index + 1, current, visit);
        current.pop();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CandidatePlan {
    plan: TopologyPlan,
    minimum_remaining_vram: u64,
    total_remaining_vram: u128,
    minimum_remaining_ram: u64,
    total_remaining_ram: u128,
}

impl Ord for CandidatePlan {
    fn cmp(&self, other: &Self) -> Ordering {
        self.minimum_remaining_vram
            .cmp(&other.minimum_remaining_vram)
            .then_with(|| self.total_remaining_vram.cmp(&other.total_remaining_vram))
            .then_with(|| self.minimum_remaining_ram.cmp(&other.minimum_remaining_ram))
            .then_with(|| self.total_remaining_ram.cmp(&other.total_remaining_ram))
            .then_with(|| {
                let left = self
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                let right = other
                    .plan
                    .stages
                    .iter()
                    .map(|stage| stage.node_id.as_str())
                    .collect::<Vec<_>>();
                right.cmp(&left)
            })
    }
}

impl PartialOrd for CandidatePlan {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn fit_candidate(
    input: &TopologyPlanningInput,
    nodes: &[UsableNode],
    context_length: u32,
    parallel_lanes: usize,
) -> Option<CandidatePlan> {
    let layer_count = input.layer_count as usize;
    if nodes.len() > layer_count {
        return None;
    }

    let layer_weights = layer_weight_bytes(input);
    let kv_per_layer = input
        .kv_bytes_per_token
        .div_ceil(u64::from(input.layer_count));
    let layer_required_bytes =
        layer_required_bytes(&layer_weights, kv_per_layer, context_length, parallel_lanes)?;

    let mut capacities = nodes.to_vec();
    capacities.sort_by(|left, right| {
        right
            .usable_vram_bytes
            .cmp(&left.usable_vram_bytes)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });

    let mut next_layer = 0u32;
    let mut stages = Vec::with_capacity(capacities.len());
    let mut minimum_remaining_vram = u64::MAX;
    let mut total_remaining_vram = 0u128;
    let mut minimum_remaining_ram = u64::MAX;
    let mut total_remaining_ram = 0u128;

    for (stage_index, node) in capacities.iter().enumerate() {
        let remaining_layers = input.layer_count - next_layer;
        let remaining_nodes = capacities.len() - stage_index;
        let min_for_later = remaining_nodes.saturating_sub(1) as u32;
        let assignable = remaining_layers.saturating_sub(min_for_later);
        let layer_span = if input.layer_class_bytes.is_empty() {
            assignable.min(max_contiguous_layers_from(
                &layer_required_bytes,
                next_layer as usize,
                assignable as usize,
                node.usable_vram_bytes,
            ) as u32)
        } else {
            assignable.min(max_contiguous_hybrid_layers_from(
                &layer_required_bytes,
                next_layer as usize,
                assignable as usize,
                node.usable_vram_bytes,
                node.usable_ram_bytes,
                &input.layer_class_bytes,
            ) as u32)
        };
        if layer_span == 0 {
            return None;
        }

        let layer_start = next_layer;
        let layer_end = layer_start + layer_span;
        let range = layer_start as usize..layer_end as usize;
        let parameter_bytes = sum_u64(&layer_weights[range.clone()]);
        let required_bytes = sum_u64(&layer_required_bytes[range.clone()]);

        // AutoHybrid: check if we need to offload routed experts to host.
        let routed_expert_bytes: u64 = if input.layer_class_bytes.is_empty() {
            0
        } else {
            input.layer_class_bytes[range.clone()]
                .iter()
                .map(|lc| lc.routed_expert)
                .sum()
        };
        let accelerator_bytes = required_bytes.saturating_sub(routed_expert_bytes);

        // If everything fits in VRAM, no offload needed.
        // Otherwise, offload routed experts to host and check host capacity.
        let (host_bytes, final_accel_bytes) = if required_bytes <= node.usable_vram_bytes {
            (0, required_bytes)
        } else {
            if routed_expert_bytes > node.usable_ram_bytes {
                return None;
            }
            if accelerator_bytes > node.usable_vram_bytes {
                return None;
            }
            (routed_expert_bytes, accelerator_bytes)
        };

        if final_accel_bytes > node.usable_vram_bytes {
            return None;
        }
        let remaining_vram = node.usable_vram_bytes - final_accel_bytes;
        let remaining_ram = node.usable_ram_bytes.saturating_sub(host_bytes);
        minimum_remaining_vram = minimum_remaining_vram.min(remaining_vram);
        total_remaining_vram += u128::from(remaining_vram);
        minimum_remaining_ram = minimum_remaining_ram.min(remaining_ram);
        total_remaining_ram += u128::from(remaining_ram);
        stages.push(TopologyStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id: node.node_id.clone(),
            layer_start,
            layer_end,
            parameter_bytes,
        });
        next_layer = layer_end;
    }

    if next_layer != input.layer_count {
        return None;
    }

    let estimated_decode_network_ms_per_token = estimate_decode_network_ms_per_token(nodes);
    Some(CandidatePlan {
        plan: TopologyPlan {
            context_length,
            parallel_lanes,
            stages,
            estimated_decode_network_ms_per_token,
            decode_tpot_target_met: decode_tpot_target_met(
                estimated_decode_network_ms_per_token,
                input.target_decode_tpot_ms,
            ),
        },
        minimum_remaining_vram,
        total_remaining_vram,
        minimum_remaining_ram,
        total_remaining_ram,
    })
}

fn latency_aware_planning(_input: &TopologyPlanningInput, nodes: &[UsableNode]) -> bool {
    nodes
        .iter()
        .any(|node| node.stage_transfer_latency_ms.is_some())
}

fn candidate_has_required_stage0(
    candidate: &CandidatePlan,
    required_stage0_node_id: Option<&str>,
) -> bool {
    required_stage0_node_id.is_none_or(|required| {
        candidate
            .plan
            .stages
            .first()
            .is_some_and(|stage| stage.node_id == required)
    })
}

fn candidate_better_for_same_shape(candidate: &CandidatePlan, current: &CandidatePlan) -> bool {
    let candidate_estimate = candidate
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let current_estimate = current
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    candidate_estimate < current_estimate
        || (candidate_estimate == current_estimate && candidate.cmp(current) == Ordering::Greater)
}

fn latency_candidate_better(
    candidate: &CandidatePlan,
    current: &CandidatePlan,
    input: &TopologyPlanningInput,
) -> bool {
    latency_candidate_ordering(candidate, current, input) == Ordering::Greater
}

fn latency_candidate_ordering(
    left: &CandidatePlan,
    right: &CandidatePlan,
    input: &TopologyPlanningInput,
) -> Ordering {
    let left_estimate = left
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let right_estimate = right
        .plan
        .estimated_decode_network_ms_per_token
        .unwrap_or_default();
    let left_target_met = decode_tpot_target_met(
        left.plan.estimated_decode_network_ms_per_token,
        input.target_decode_tpot_ms,
    )
    .unwrap_or(true);
    let right_target_met = decode_tpot_target_met(
        right.plan.estimated_decode_network_ms_per_token,
        input.target_decode_tpot_ms,
    )
    .unwrap_or(true);

    left_target_met
        .cmp(&right_target_met)
        .then_with(|| right_estimate.cmp(&left_estimate))
        .then_with(|| left.plan.context_length.cmp(&right.plan.context_length))
        .then_with(|| left.plan.parallel_lanes.cmp(&right.plan.parallel_lanes))
        .then_with(|| left.cmp(right))
}

fn estimate_decode_network_ms_per_token(nodes: &[UsableNode]) -> Option<u32> {
    let hop_latency = nodes
        .iter()
        .filter_map(|node| node.stage_transfer_latency_ms)
        .max()?;
    Some(hop_latency.saturating_mul(nodes.len() as u32))
}

fn decode_tpot_target_met(estimate: Option<u32>, target: Option<u32>) -> Option<bool> {
    Some(estimate? <= target?)
}

fn layer_weight_bytes(input: &TopologyPlanningInput) -> Vec<u64> {
    if input.layer_weight_bytes.len() == input.layer_count as usize {
        return input.layer_weight_bytes.clone();
    }
    let weight_per_layer = input
        .model_weight_bytes
        .div_ceil(u64::from(input.layer_count));
    vec![weight_per_layer; input.layer_count as usize]
}

fn candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    _parallel_lanes: usize,
) -> Option<u64> {
    // KV cache is a single shared allocation of size `n_ctx` — all lanes
    // share one unified cache via sequence IDs (kv_unified=true in
    // llama.cpp when lane_count > 1).  Do not multiply by lanes.
    let kv_bytes = u128::from(kv_per_layer).checked_mul(u128::from(context_length))?;
    // Charge KV at 100/85 so 15% of the node's post-weight space is held back
    // for llama.cpp compute-graph buffers/scratch (mirrors the single-node
    // context planner's `usable_kv_cache_budget`). This scales the reserve with
    // context length, matching how compute buffers grow with `n_ctx`.
    let kv_with_compute_reserve = kv_bytes
        .checked_mul(KV_COMPUTE_RESERVE_NUMERATOR)?
        .div_ceil(KV_COMPUTE_RESERVE_DENOMINATOR);
    let total = u128::from(weight_per_layer).checked_add(kv_with_compute_reserve)?;
    total.try_into().ok()
}

fn layer_required_bytes(
    layer_weights: &[u64],
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> Option<Vec<u64>> {
    layer_weights
        .iter()
        .map(|weight| {
            candidate_bytes_per_layer(*weight, kv_per_layer, context_length, parallel_lanes)
        })
        .collect()
}

fn max_contiguous_layers_from(
    layer_required_bytes: &[u64],
    start: usize,
    limit: usize,
    capacity: u64,
) -> u64 {
    let mut total = 0u64;
    let mut count = 0u64;
    for bytes in layer_required_bytes.iter().skip(start).take(limit) {
        let next = total.saturating_add(*bytes);
        if next > capacity {
            break;
        }
        total = next;
        count += 1;
    }
    count
}

fn max_contiguous_hybrid_layers_from(
    layer_required_bytes: &[u64],
    start: usize,
    limit: usize,
    vram_capacity: u64,
    ram_capacity: u64,
    layer_class_bytes: &[TopologyLayerClassBytes],
) -> u64 {
    let mut vram_total = 0u64;
    let mut ram_total = 0u64;
    let mut count = 0u64;
    for (i, bytes) in layer_required_bytes.iter().enumerate().skip(start).take(limit) {
        let routed = layer_class_bytes
            .get(i)
            .map(|lc| lc.routed_expert)
            .unwrap_or(0);
        let non_routed = bytes.saturating_sub(routed);
        let next_vram = vram_total.saturating_add(non_routed);
        let next_ram = ram_total.saturating_add(routed);
        if next_vram > vram_capacity || next_ram > ram_capacity {
            break;
        }
        vram_total = next_vram;
        ram_total = next_ram;
        count += 1;
    }
    count
}

fn sum_u64(values: &[u64]) -> u64 {
    values
        .iter()
        .fold(0u64, |total, value| total.saturating_add(*value))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const QWEN_CODER_480B_NATIVE_CONTEXT: u32 = 262_144;
    const QWEN_CODER_480B_LAYERS: u32 = 62;
    const QWEN_CODER_480B_WEIGHT_BYTES: u64 = 315_680_000_000;
    const QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN: u64 = 128 * 1024;
    const LOCAL_M1_ULTRA_METAL_BYTES: u64 = 115_448_725_504;
    const STUDIO_METAL_BYTES: u64 = 239_143_780_352;
    const STUDIO_RAM_BYTES: u64 = 274_877_906_944;

    fn node(id: &str, gib: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: gib * GIB,
            detected_host_available_bytes: 0,
            max_vram_bytes: None,
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    fn node_with_ram(id: &str, gib: u64, host_gib: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: gib * GIB,
            detected_host_available_bytes: host_gib * GIB,
            max_vram_bytes: None,
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    fn hybrid_qwen3_122b_layer_bytes() -> Vec<TopologyLayerClassBytes> {
        // Load from fixture file for consistency with real package dimensions.
        // The fixture is generated from the actual Qwen3.5-122B-A10B Q3_K_XL model.
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3.5-122b-a10b-q3_k_xl-layer-inventory.json");
        std::fs::read_to_string(&fixture_path)
            .ok()
            .and_then(|content| serde_json::from_str::<serde_json::Value>(&content).ok())
            .and_then(|value| {
                let layers = value.get("layers")?.as_array()?;
                let mut result = Vec::new();
                for layer in layers {
                    let class_bytes = layer.get("class_bytes")?;
                    result.push(TopologyLayerClassBytes {
                        routed_expert: class_bytes.get("routed_expert")?.as_u64()?,
                        shared_expert: class_bytes.get("shared_expert")?.as_u64()?,
                        attention: class_bytes.get("attention")?.as_u64()?,
                        recurrent_ssm: class_bytes.get("recurrent_ssm")?.as_u64()?,
                        routing_gate: class_bytes.get("routing_gate")?.as_u64()?,
                        normalization: class_bytes.get("normalization")?.as_u64()?,
                        other: class_bytes.get("other")?.as_u64()?,
                    });
                }
                Some(result)
            })
            .unwrap_or_default()
    }

    #[test]
    fn qwen35_122b_auto_hybrid_offloads_routed_experts_to_host() {
        // Load the real fixture file for Qwen3.5-122B-A10B Q3_K_XL.
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3.5-122b-a10b-q3_k_xl-layer-inventory.json");
        let fixture = std::fs::read_to_string(&fixture_path)
            .expect("fixture file should exist");
        let fixture: serde_json::Value =
            serde_json::from_str(&fixture).expect("fixture should be valid JSON");

        let layers = fixture["layers"]
            .as_array()
            .expect("fixture should have layers array");
        let layer_count = layers.len() as u32;

        // Build layer_class_bytes and layer_weight_bytes from fixture.
        let mut layer_class_bytes = Vec::new();
        let mut layer_weight_bytes = Vec::new();
        let mut total_weight_bytes = 0u64;
        let mut total_routed_expert_bytes = 0u64;

        for layer in layers {
            let tensor_bytes = layer["tensor_bytes"]
                .as_u64()
                .expect("tensor_bytes should be u64");
            let class_bytes = &layer["class_bytes"];

            let routed_expert = class_bytes["routed_expert"]
                .as_u64()
                .expect("routed_expert should be u64");
            let shared_expert = class_bytes["shared_expert"]
                .as_u64()
                .expect("shared_expert should be u64");
            let attention = class_bytes["attention"].as_u64().expect("attention should be u64");
            let recurrent_ssm = class_bytes["recurrent_ssm"]
                .as_u64()
                .expect("recurrent_ssm should be u64");
            let routing_gate = class_bytes["routing_gate"]
                .as_u64()
                .expect("routing_gate should be u64");
            let normalization = class_bytes["normalization"]
                .as_u64()
                .expect("normalization should be u64");
            let other = class_bytes["other"].as_u64().expect("other should be u64");

            // Verify class sum == tensor_bytes.
            let class_sum =
                routed_expert + shared_expert + attention + recurrent_ssm + routing_gate + normalization + other;
            assert_eq!(
                class_sum, tensor_bytes,
                "class sum {} != tensor_bytes {} for layer {}",
                class_sum,
                tensor_bytes,
                layer["layer_index"].as_u64().unwrap_or(0)
            );

            layer_class_bytes.push(TopologyLayerClassBytes {
                routed_expert,
                shared_expert,
                attention,
                recurrent_ssm,
                routing_gate,
                normalization,
                other,
            });
            layer_weight_bytes.push(tensor_bytes);
            total_weight_bytes += tensor_bytes;
            total_routed_expert_bytes += routed_expert;
        }

        // A4000: 16 GB VRAM, 32 GB RAM.
        // RX 6600 XT: 8 GB VRAM, 32 GB RAM.
        let mut request = input(vec![
            node_with_ram("a4000", 16, 32),
            node_with_ram("rx6600", 8, 32),
        ]);
        request.layer_count = layer_count;
        request.model_weight_bytes = total_weight_bytes;
        request.layer_class_bytes = layer_class_bytes.clone();
        request.layer_weight_bytes = layer_weight_bytes.clone();
        request.kv_bytes_per_token = 64 * 1024;
        request.minimum_nodes = 2;

        let plan = plan_topology(&request).expect("plan_topology should succeed");

        // Assert 49/49 layers assigned with no gaps/overlaps.
        assert_eq!(plan.stages.len(), 2);
        let mut all_layers = Vec::new();
        for stage in &plan.stages {
            all_layers.push((stage.layer_start, stage.layer_end, stage.node_id.clone()));
        }
        all_layers.sort_by_key(|(start, _, _)| *start);

        // Verify contiguous coverage.
        assert_eq!(all_layers[0].0, 0, "first stage should start at layer 0");
        for i in 1..all_layers.len() {
            assert_eq!(
                all_layers[i].0, all_layers[i - 1].1,
                "stages should be contiguous without gaps"
            );
        }
        assert_eq!(
            all_layers.last().unwrap().1,
            layer_count,
            "last stage should end at layer_count"
        );

        // Verify accel_bytes <= usable_vram AND host_bytes <= usable_ram for each stage.
        let a4000_vram = 16 * GIB;
        let rx6600_vram = 8 * GIB;
        let a4000_ram = 32 * GIB;
        let rx6600_ram = 32 * GIB;

        let mut total_host_bytes = 0u64;
        for stage in &plan.stages {
            let vram_capacity = if stage.node_id == "a4000" {
                a4000_vram
            } else {
                rx6600_vram
            };
            let ram_capacity = if stage.node_id == "a4000" {
                a4000_ram
            } else {
                rx6600_ram
            };

            let stage_routed_expert_bytes: u64 = layer_class_bytes
                [stage.layer_start as usize..stage.layer_end as usize]
                .iter()
                .map(|lc| lc.routed_expert)
                .sum();
            let stage_total_bytes: u64 =
                layer_weight_bytes[stage.layer_start as usize..stage.layer_end as usize]
                    .iter()
                    .sum();

            // AutoHybrid: accel_bytes = total - routed_expert_bytes.
            let accel_bytes = stage_total_bytes - stage_routed_expert_bytes;

            assert!(
                accel_bytes <= vram_capacity,
                "accel_bytes {} exceeds VRAM capacity {} for stage {}",
                accel_bytes,
                vram_capacity,
                stage.node_id
            );
            assert!(
                stage_routed_expert_bytes <= ram_capacity,
                "routed_expert_bytes {} exceeds RAM capacity {} for stage {}",
                stage_routed_expert_bytes,
                ram_capacity,
                stage.node_id
            );

            total_host_bytes += stage_routed_expert_bytes;
        }

        // Assert at least some routed expert bytes were offloaded to host
        // (proving AutoHybrid worked).
        assert!(
            total_host_bytes > 0,
            "expected at least some routed expert bytes to be offloaded to host"
        );

        // Print the resulting boundary.
        println!("Qwen3.5-122B-A10B Q3_K_XL topology plan:");
        for stage in &plan.stages {
            println!(
                "  {}: layers {}..{} ({} layers, {} routed expert bytes)",
                stage.node_id,
                stage.layer_start,
                stage.layer_end,
                stage.layer_end - stage.layer_start,
                layer_class_bytes[stage.layer_start as usize..stage.layer_end as usize]
                    .iter()
                    .map(|lc| lc.routed_expert)
                    .sum::<u64>()
            );
        }
    }

    #[test]
    fn qwen35_122b_vram_only_rejection_fails_without_host_assist() {
        // Load the real fixture file.
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3.5-122b-a10b-q3_k_xl-layer-inventory.json");
        let fixture = std::fs::read_to_string(&fixture_path)
            .expect("fixture file should exist");
        let fixture: serde_json::Value =
            serde_json::from_str(&fixture).expect("fixture should be valid JSON");

        let layers = fixture["layers"]
            .as_array()
            .expect("fixture should have layers array");
        let layer_count = layers.len() as u32;

        let mut layer_class_bytes = Vec::new();
        let mut layer_weight_bytes = Vec::new();
        let mut total_weight_bytes = 0u64;

        for layer in layers {
            let tensor_bytes = layer["tensor_bytes"]
                .as_u64()
                .expect("tensor_bytes should be u64");
            let class_bytes = &layer["class_bytes"];

            layer_class_bytes.push(TopologyLayerClassBytes {
                routed_expert: class_bytes["routed_expert"]
                    .as_u64()
                    .expect("routed_expert should be u64"),
                shared_expert: class_bytes["shared_expert"]
                    .as_u64()
                    .expect("shared_expert should be u64"),
                attention: class_bytes["attention"].as_u64().expect("attention should be u64"),
                recurrent_ssm: class_bytes["recurrent_ssm"]
                    .as_u64()
                    .expect("recurrent_ssm should be u64"),
                routing_gate: class_bytes["routing_gate"]
                    .as_u64()
                    .expect("routing_gate should be u64"),
                normalization: class_bytes["normalization"]
                    .as_u64()
                    .expect("normalization should be u64"),
                other: class_bytes["other"].as_u64().expect("other should be u64"),
            });
            layer_weight_bytes.push(tensor_bytes);
            total_weight_bytes += tensor_bytes;
        }

        // A4000: 16 GB VRAM, 0 GB RAM (VRAM-only mode).
        // RX 6600 XT: 8 GB VRAM, 0 GB RAM (VRAM-only mode).
        let mut request = input(vec![
            node("a4000", 16),
            node("rx6600", 8),
        ]);
        request.layer_count = layer_count;
        request.model_weight_bytes = total_weight_bytes;
        request.layer_class_bytes = layer_class_bytes;
        request.layer_weight_bytes = layer_weight_bytes;
        request.kv_bytes_per_token = 64 * 1024;
        request.minimum_nodes = 2;

        // Plan should fail because without host RAM, the routed experts cannot
        // be offloaded, and the layers won't fit in VRAM alone.
        let plan = plan_topology(&request);
        assert!(
            plan.is_err(),
            "plan_topology should fail when host_ram_available_bytes = 0 for all nodes"
        );
    }

    #[test]
    fn qwen_122b_assigns_more_layers_to_higher_vram_node_with_host_assist() {
        let mut request = input(vec![
            node_with_ram("a4000", 16, 32),
            node_with_ram("rx6600", 8, 32),
        ]);
        request.layer_count = 49;
        request.model_weight_bytes = 57_030_728_384;
        request.layer_class_bytes = hybrid_qwen3_122b_layer_bytes();
        request.layer_weight_bytes = vec![1_177_551_020; 49];
        request.kv_bytes_per_token = 64 * 1024;
        request.minimum_nodes = 2;
        let plan = plan_topology(&request).unwrap();
        assert_eq!(plan.stages.len(), 2);
        let a4000 = plan.stages.iter().find(|s| s.node_id == "a4000").unwrap();
        let rx6600 = plan.stages.iter().find(|s| s.node_id == "rx6600").unwrap();
        assert!(a4000.layer_end - a4000.layer_start >= rx6600.layer_end - rx6600.layer_start);
    }

    #[test]
    fn vram_only_feasibility_rejects_49_layers_without_host_assist() {
        let mut request = input(vec![node("a4000", 16), node("rx6600", 8)]);
        request.layer_count = 49;
        request.model_weight_bytes = 57_030_728_384;
        request.layer_weight_bytes = vec![1_177_551_020; 49];
        request.kv_bytes_per_token = 64 * 1024;
        request.minimum_nodes = 2;
        let plan = plan_topology(&request);
        assert!(plan.is_err());
    }

    fn latency_node(id: &str, gib: u64, stage_transfer_latency_ms: u32) -> TopologyNode {
        TopologyNode {
            stage_transfer_latency_ms: Some(stage_transfer_latency_ms),
            ..node(id, gib)
        }
    }

    fn input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: 65_536,
            layer_count: 40,
            model_weight_bytes: 40 * GIB,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: 64 * 1024,
            minimum_nodes: 1,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
            layer_class_bytes: Vec::new(),
        }
    }

    fn qwen_coder_480b_input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: QWEN_CODER_480B_NATIVE_CONTEXT,
            layer_count: QWEN_CODER_480B_LAYERS,
            model_weight_bytes: QWEN_CODER_480B_WEIGHT_BYTES,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: QWEN_CODER_480B_Q8_KV_BYTES_PER_TOKEN,
            minimum_nodes: 2,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
            layer_class_bytes: Vec::new(),
        }
    }

    fn qwen_node(index: usize, gib: u64) -> TopologyNode {
        node(&format!("qwen-node-{index:02}"), gib)
    }

    fn qwen_nodes(count: usize, gib: u64) -> Vec<TopologyNode> {
        (0..count).map(|index| qwen_node(index, gib)).collect()
    }

    fn metal_node(id: &str, metal_recommended_bytes: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: metal_recommended_bytes,
            detected_host_available_bytes: 0,
            max_vram_bytes: Some(metal_recommended_bytes),
            // Metal recommendedMaxWorkingSetSize is already the usable budget
            // reported by the local runtime.
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    #[test]
    fn chooses_highest_context_then_parallelism() {
        let plan = plan_topology(&input(vec![node("a", 23), node("b", 23)])).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 4);
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn prefers_fewest_nodes_before_more_lanes() {
        let plan = plan_topology(&input(vec![
            node("a", 80),
            node("b", 80),
            node("c", 80),
            node("d", 80),
            node("e", 80),
            node("f", 80),
        ]))
        .unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 1);
        assert_eq!(plan.parallel_lanes, 4);
    }

    #[test]
    fn assigns_fewer_layers_to_lower_vram_node() {
        let mut request = input(vec![node("small", 16), node("large", 48)]);
        request.minimum_nodes = 2;
        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        let small = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "small")
            .unwrap();
        let large = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "large")
            .unwrap();
        assert!(small.layer_end - small.layer_start < large.layer_end - large.layer_start);
    }

    #[test]
    fn exact_layer_weights_allow_uneven_package_fit() {
        let mut request = input(vec![node("large", 12), node("small", 9)]);
        request.layer_count = 4;
        request.model_weight_bytes = 18 * GIB;
        request.layer_weight_bytes = vec![GIB / 8, GIB / 8, 9 * GIB, 8 * GIB];
        request.kv_bytes_per_token = 1;
        request.minimum_nodes = 2;

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![("large", 0, 3), ("small", 3, 4)]
        );
        assert_eq!(plan.stages[0].parameter_bytes, 9 * GIB + GIB / 4);
        assert_eq!(plan.stages[1].parameter_bytes, 8 * GIB);
    }

    #[test]
    fn exact_layer_capacity_is_evaluated_at_each_stage_boundary() {
        let mut request = input(vec![node("large", 11), node("small", 3)]);
        request.layer_count = 4;
        request.model_weight_bytes = 12 * GIB;
        request.layer_weight_bytes = vec![9 * GIB, GIB, GIB, GIB];
        request.kv_bytes_per_token = 1;
        request.minimum_nodes = 2;

        let plan = plan_topology(&request).unwrap();

        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 2), (2, 4)]
        );
    }

    #[test]
    fn applies_max_vram_and_runtime_headroom_per_node() {
        let mut capped = node("capped", 80);
        capped.max_vram_bytes = Some(24 * GIB);
        capped.runtime_headroom_bytes = 8 * GIB;
        let mut request = input(vec![capped, node("peer", 48)]);
        request.minimum_nodes = 2;
        let plan = plan_topology(&request).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "capped")
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < 20);
    }

    #[test]
    fn latency_aware_planner_prefers_lower_tpot_over_native_context() {
        let mut request = input(vec![
            latency_node("a", 23, 10),
            latency_node("b", 23, 10),
            latency_node("c", 23, 10),
            latency_node("d", 23, 10),
        ]);
        request.native_context_length = 262_144;
        request.minimum_nodes = 2;
        request.target_decode_tpot_ms = Some(33);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(20));
        assert_eq!(plan.decode_tpot_target_met, Some(true));
    }

    #[test]
    fn latency_aware_planner_reports_target_miss_when_memory_requires_more_stages() {
        let mut request = qwen_coder_480b_input(qwen_nodes(4, 80));
        request
            .nodes
            .iter_mut()
            .for_each(|node| node.stage_transfer_latency_ms = Some(10));
        request.target_decode_tpot_ms = Some(33);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 4);
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(40));
        assert_eq!(plan.decode_tpot_target_met, Some(false));
    }

    #[test]
    fn rejects_below_minimum_context_floor() {
        let err = plan_topology(&input(vec![node("tiny-a", 8), node("tiny-b", 8)]))
            .expect_err("context below the 64k floor should be rejected");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 65_536
            }
        );
    }

    #[test]
    fn minimum_context_floor_caps_at_native_context() {
        assert_eq!(minimum_valid_context(16_384), 16_384);
        assert_eq!(minimum_valid_context(65_536), 65_536);
        assert_eq!(minimum_valid_context(262_144), 65_536);
    }

    #[test]
    fn accepts_explicit_context_override_below_auto_floor() {
        let mut request = input(vec![node("a", 80), node("b", 80)]);
        request.native_context_length = 262_144;
        request.context_length_override = Some(32_768);

        let plan = plan_topology(&request).unwrap();

        assert_eq!(plan.context_length, 32_768);
    }

    #[test]
    fn rejects_context_override_above_native() {
        let mut request = input(vec![node("a", 80)]);
        request.context_length_override = Some(131_072);

        assert_eq!(
            plan_topology(&request),
            Err(TopologyPlanError::ContextExceedsNative {
                requested: 131_072,
                native: 65_536,
            })
        );
    }

    #[test]
    fn qwen_coder_480b_rejects_when_layers_cannot_fit_above_context_floor() {
        // Simulation: 4 x 70 GiB nodes.
        //
        // Expected topology: none.
        //
        // Why: the planner may degrade context only to the shared 64k floor
        // (65_536). At this machine size the full 62-layer package plus
        // 64k KV cannot be distributed, so launching would silently produce
        // an under-resourced split.
        let err = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 70)))
            .expect_err("four 70 GiB nodes cannot hold this layer package above the context floor");

        assert_eq!(
            err,
            TopologyPlanError::NoValidTopology {
                minimum_context: 65_536
            }
        );
    }

    #[test]
    fn qwen_coder_480b_studio_james_and_studio_mic_form_native_topology() {
        // Simulation: meshllm/Qwen3-Coder-480B-A35B-Instruct-UD-Q4_K_XL-layers
        // split across studio-james and studio-mic.
        //
        // studio-james:
        //   Metal recommendedMaxWorkingSetSize = 115_448_725_504 bytes.
        //
        // studio-mic:
        //   Metal recommendedMaxWorkingSetSize = 239_143_780_352 bytes.
        //   RAM = 274_877_906_944 bytes. RAM is documented here because it is
        //   part of the fixture, but the planner must use Metal working set
        //   size, not total RAM.
        //
        // Expected topology: possible, 131_072 context, 4 lanes.
        //
        // Why: this is a fixture-driven simulation. The model package metadata
        // and each machine's Metal working-set budget are passed into the same
        // planner used by runtime orchestration, and the planner reports
        // whether a topology can be formed plus its context and lane count.
        //
        // Context is 131_072 rather than the model's 262_144 native maximum
        // because the planner reserves compute-buffer headroom (KV billed at
        // 100/85). The ~316 GB of weights plus full-native KV would pack the
        // combined ~354.6 GB working-set budget to within a few GB, leaving no
        // room for llama.cpp compute graphs; halving the context restores ~18 GB
        // of headroom across the two stages. This is the fix for stages that
        // previously loaded at native context and then OOM'd on the first token.
        assert_eq!(STUDIO_RAM_BYTES, 274_877_906_944);

        let planned = plan_topology(&qwen_coder_480b_input(vec![
            metal_node("studio-james", LOCAL_M1_ULTRA_METAL_BYTES),
            metal_node("studio-mic", STUDIO_METAL_BYTES),
        ]));
        let (split_possible, context_length, parallel_lanes) = match &planned {
            Ok(plan) => (true, Some(plan.context_length), Some(plan.parallel_lanes)),
            Err(_) => (false, None, None),
        };

        assert!(split_possible, "{planned:?}");
        assert_eq!(context_length, Some(131_072));
        assert_eq!(parallel_lanes, Some(4));

        let plan = planned.expect("studio-james and studio-mic should form a split topology");
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(
            plan.stages.last().unwrap().layer_end,
            QWEN_CODER_480B_LAYERS
        );
    }

    #[test]
    fn qwen_coder_480b_uses_context_floor_when_larger_contexts_do_not_fit() {
        // Simulation: 4 x 80 GiB nodes.
        //
        // Expected topology: 4 stages, 65_536 context, 4 lanes.
        //
        // Why: native 262_144 and 131_072 contexts do not fit across these
        // nodes, but the shared 64k floor does.  Lanes use a shared unified
        // KV cache and do not multiply memory cost, so the auto cap of 4
        // applies.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(4, 80))).unwrap();

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.parallel_lanes, 4);
        assert_eq!(plan.stages.len(), 4);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(
            plan.stages.last().unwrap().layer_end,
            QWEN_CODER_480B_LAYERS
        );
    }

    #[test]
    fn qwen_coder_480b_prefers_native_context_then_parallelism() {
        // Simulation: 5 x 80 GiB nodes.
        //
        // Expected topology: 5 stages, native 262_144 context, 4 lanes.
        //
        // Why: adding the fifth node makes native context fit.  Lanes use a
        // shared unified KV cache, so the auto cap of 4 applies.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(5, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 4);
        assert_eq!(plan.stages.len(), 5);
    }

    #[test]
    fn qwen_coder_480b_prefers_fewest_nodes_then_maximizes_lanes() {
        // Simulation: 10 x 80 GiB nodes.
        //
        // Expected topology: 5 stages, native 262_144 context, 4 lanes.
        //
        // Why: the planner prefers fewest nodes before more lanes. Five nodes
        // is the minimum that can hold the full layer package at native
        // context.  Lanes use a shared unified KV cache, so the auto cap of
        // 4 applies regardless of extra VRAM headroom.
        let plan = plan_topology(&qwen_coder_480b_input(qwen_nodes(10, 80))).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 4);
        assert_eq!(plan.stages.len(), 5);
    }

    #[test]
    fn qwen_coder_480b_excludes_bystander_nodes() {
        // Simulation: 7 x 80 GiB nodes plus 3 x 1 GiB bystanders.
        //
        // Expected topology: 5 stages, native 262_144 context, 4 lanes.
        //
        // Why: the planner prefers fewest nodes first. Five 80 GiB nodes
        // achieve native context. Bystander nodes (1 GiB) cannot carry even
        // one layer at this shape and are excluded entirely.
        let mut nodes = qwen_nodes(7, 80);
        nodes.extend((7..10).map(|index| qwen_node(index, 1)));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        assert_eq!(plan.context_length, QWEN_CODER_480B_NATIVE_CONTEXT);
        assert_eq!(plan.parallel_lanes, 4);
        assert_eq!(plan.stages.len(), 5);
        assert!(
            plan.stages
                .iter()
                .all(|stage| !stage.node_id.ends_with("07")
                    && !stage.node_id.ends_with("08")
                    && !stage.node_id.ends_with("09"))
        );
    }

    #[test]
    fn qwen_coder_480b_assigns_less_work_to_smaller_nodes() {
        // Simulation: 1 x 64 GiB node and 5 x 80 GiB nodes.
        //
        // Expected topology: native context with the 64 GiB node assigned
        // fewer layers than the largest stage.
        //
        // Why: KV and weights are layer-local. Assigning fewer layers to the
        // smaller node prevents it from forcing down the cluster-wide context.
        let mut nodes = vec![qwen_node(0, 64)];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let smallest_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(smallest_stage.layer_end - smallest_stage.layer_start < max_layers);
    }

    #[test]
    fn qwen_coder_480b_applies_max_vram_and_headroom_in_simulation() {
        // Simulation: one physically larger 120 GiB node capped to 80 GiB
        // with 16 GiB runtime headroom, plus 5 x 80 GiB nodes.
        //
        // Expected topology: the capped node receives fewer layers than the
        // largest stage, despite having 120 GiB physically detected.
        //
        // Why: planning must apply max-vram and local runtime headroom per
        // node before assigning layers. The capped node's usable budget is
        // 64 GiB, so it should be treated as smaller than the uncapped peers.
        let mut capped = qwen_node(0, 120);
        capped.max_vram_bytes = Some(80 * GIB);
        capped.runtime_headroom_bytes = 16 * GIB;
        let mut nodes = vec![capped];
        nodes.extend(qwen_nodes(5, 80).into_iter().skip(1));
        let plan = plan_topology(&qwen_coder_480b_input(nodes)).unwrap();

        let capped_stage = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "qwen-node-00")
            .unwrap();
        let max_layers = plan
            .stages
            .iter()
            .map(|stage| stage.layer_end - stage.layer_start)
            .max()
            .unwrap();
        assert!(capped_stage.layer_end - capped_stage.layer_start < max_layers);
    }
}
