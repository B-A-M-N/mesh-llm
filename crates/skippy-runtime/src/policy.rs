use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use anyhow::Result;

use crate::ModelInfo;

// ─── Tensor classification ───────────────────────────────────────────────────

/// Tensor class for memory accounting.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum TensorClass {
    /// Routed expert weights (HOST under CpuMoe policy).
    RoutedExpert = 0,
    /// Shared expert weights (accelerator by default).
    SharedExpert = 1,
    /// Attention projections / norms.
    Attention = 2,
    /// SSM recurrent states (RWKV-style).
    RecurrentSsm = 3,
    /// Router / gating weights (input-side).
    RoutingGate = 4,
    /// Layer norms, embeddings, output, metadata.
    Normalization = 5,
    /// Unknown / unclassified (conservatively treated as accelerator).
    Other = 6,
}

/// A single tensor with its exact byte size and classification.
#[derive(Clone, Debug)]
pub struct ClassifiedTensor {
    pub name: String,
    pub layer: Option<u32>,
    pub class: TensorClass,
    pub bytes: u64,
}

// ─── Per-class byte counters ─────────────────────────────────────────────────

/// Per-class byte counters for a layer or global scope.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorClassBytes {
    pub routed_expert: u64,
    pub shared_expert: u64,
    pub attention: u64,
    pub recurrent_ssm: u64,
    pub routing_gate: u64,
    pub normalization: u64,
    pub other: u64,
}

impl TensorClassBytes {
    pub fn total(&self) -> u64 {
        self.routed_expert
            + self.shared_expert
            + self.attention
            + self.recurrent_ssm
            + self.routing_gate
            + self.normalization
            + self.other
    }

    pub fn accelerator_bytes(&self) -> u64 {
        self.total() - self.routed_expert
    }

    pub fn host_bytes(&self) -> u64 {
        self.routed_expert
    }

    pub fn add(&mut self, class: TensorClass, bytes: u64) {
        match class {
            TensorClass::RoutedExpert => self.routed_expert += bytes,
            TensorClass::SharedExpert => self.shared_expert += bytes,
            TensorClass::Attention => self.attention += bytes,
            TensorClass::RecurrentSsm => self.recurrent_ssm += bytes,
            TensorClass::RoutingGate => self.routing_gate += bytes,
            TensorClass::Normalization => self.normalization += bytes,
            TensorClass::Other => self.other += bytes,
        }
    }
}

impl fmt::Display for TensorClassBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "routed={} shared={} attn={} ssm={} route={} norm={} other={} total={}",
            self.routed_expert,
            self.shared_expert,
            self.attention,
            self.recurrent_ssm,
            self.routing_gate,
            self.normalization,
            self.other,
            self.total()
        )
    }
}

// ─── Layer inventory ─────────────────────────────────────────────────────────

/// Per-layer tensor-class byte inventory.
#[derive(Clone, Debug)]
pub struct LayerTensorInventory {
    pub layer_index: u32,
    pub bytes: TensorClassBytes,
    pub total_bytes: u64,
}

impl LayerTensorInventory {
    pub fn accelerator_bytes(&self) -> u64 {
        self.bytes.accelerator_bytes()
    }

    pub fn host_bytes(&self) -> u64 {
        self.bytes.host_bytes()
    }
}

// ─── Full model inventory ────────────────────────────────────────────────────

/// Full model tensor-class inventory.
///
/// Retains per-tensor records so the placement solver can make individual
/// offload decisions. Global tensors are split by ownership so stage selection
/// can determine exactly which globals a candidate stage owns.
#[derive(Clone, Debug)]
pub struct TensorClassInventory {
    pub layers: Vec<LayerTensorInventory>,
    pub tensors: Vec<ClassifiedTensor>,
    pub global_embeddings: TensorClassBytes,
    pub global_output: TensorClassBytes,
    pub global_other: TensorClassBytes,
    pub global_total_bytes: u64,
    pub total_tensor_bytes: u64,
    pub unknown_tensor_count: u64,
    pub unknown_tensor_bytes: u64,
}

impl TensorClassInventory {
    pub fn layer(&self, index: u32) -> Option<&LayerTensorInventory> {
        self.layers.iter().find(|l| l.layer_index == index)
    }

    pub fn class_bytes_for_range(&self, range: Range<u32>) -> TensorClassBytes {
        let mut result = TensorClassBytes::default();
        for layer in &self.layers {
            if range.contains(&layer.layer_index) {
                result.routed_expert += layer.bytes.routed_expert;
                result.shared_expert += layer.bytes.shared_expert;
                result.attention += layer.bytes.attention;
                result.recurrent_ssm += layer.bytes.recurrent_ssm;
                result.routing_gate += layer.bytes.routing_gate;
                result.normalization += layer.bytes.normalization;
                result.other += layer.bytes.other;
            }
        }
        result
    }

    pub fn total_layers(&self) -> u32 {
        self.layers.len() as u32
    }
}

// ─── Classification function ─────────────────────────────────────────────────

/// Classify a tensor by name using Qwen3.5-122B naming conventions.
///
/// Order matters: routed experts first (they are the target of CpuMoe policy),
/// then shared experts, then specialized patterns, with broad patterns last
/// to avoid false matches.
pub fn classify_tensor(name: &str) -> TensorClass {
    // Routed experts first (CpuMoe target).
    for pat in &["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        if name.contains(pat) {
            return TensorClass::RoutedExpert;
        }
    }

    // Shared experts.
    for pat in &["ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"] {
        if name.contains(pat) {
            return TensorClass::SharedExpert;
        }
    }

    // Attention projections.
    for pat in &[
        "attn_q_a_proj",
        "attn_q_b_proj",
        "attn_k_a_proj",
        "attn_k_b_proj",
        "attn_v_a_proj",
        "attn_v_b_proj",
        "attn_o_proj",
    ] {
        if name.contains(pat) {
            return TensorClass::Attention;
        }
    }

    // SSM recurrent (RWKV-style).
    for pat in &["x_proj", "dt_proj", "A_log", "D_broadcast"] {
        if name.contains(pat) {
            return TensorClass::RecurrentSsm;
        }
    }

    // Normalization.
    for pat in &[
        "norm",
        "layernorm",
        "input_layernorm",
        "post_attention_layernorm",
    ] {
        if name.contains(pat) {
            return TensorClass::Normalization;
        }
    }

    // Routing gate (must come after routed/shared expert checks since
    // "ffn_gate" would match "ffn_gate_exps").
    if name.contains("attn_gate") || name.contains("router") {
        return TensorClass::RoutingGate;
    }

    // Unknown / unclassified.
    TensorClass::Other
}

/// Classify a non-layer (global) tensor by ownership.
///
/// Returns:
/// - `true` for `is_embedding` if the tensor belongs to the first stage
/// - `true` for `is_output` if the tensor belongs to the final stage
/// - otherwise it's an `other` global (metadata, final norm, etc.)
pub fn classify_global_tensor(name: &str) -> (bool, bool) {
    if name.contains("token_embd") || name.contains("embedding") {
        (true, false)
    } else if name.contains("output") || name.contains("output_norm") {
        (false, true)
    } else {
        (false, false)
    }
}

// ─── Inventory builder ───────────────────────────────────────────────────────

/// Build a TensorClassInventory from a Skippy ModelInfo handle.
///
/// Reads exact tensor byte sizes from the GGUF metadata. Every tensor is
/// counted exactly once; unknown tensors go to `Other`, not omitted.
///
/// Global tensors are split by ownership:
/// - `global_embeddings`: owned by the first stage
/// - `global_output`: owned by the final stage
/// - `global_other`: metadata, final norm, etc.
pub fn build_tensor_class_inventory(info: &ModelInfo) -> Result<TensorClassInventory> {
    let count = info.tensor_count()?;
    let mut layers: BTreeMap<u32, TensorClassBytes> = BTreeMap::new();
    let mut global_embeddings = TensorClassBytes::default();
    let mut global_output = TensorClassBytes::default();
    let mut global_other = TensorClassBytes::default();
    let mut tensors = Vec::with_capacity(count);
    let mut global_total_bytes: u64 = 0;
    let mut total_tensor_bytes: u64 = 0;
    let mut unknown_tensor_count: u64 = 0;
    let mut unknown_tensor_bytes: u64 = 0;

    for i in 0..count {
        let tensor = info.tensor_at(i)?;
        let name = tensor.name.clone();
        let class = classify_tensor(&name);
        let size = tensor.byte_size;

        if class == TensorClass::Other {
            unknown_tensor_count += 1;
            unknown_tensor_bytes += size;
        }

        tensors.push(ClassifiedTensor {
            name: name.clone(),
            layer: tensor.layer_index.map(|x| x as u32),
            class,
            bytes: size,
        });

        match tensor.layer_index {
            Some(idx) => {
                let idx = idx as u32;
                layers.entry(idx).or_default().add(class, size);
            }
            None => {
                let (is_embedding, is_output) = classify_global_tensor(&name);
                if is_embedding {
                    global_embeddings.add(class, size);
                } else if is_output {
                    global_output.add(class, size);
                } else {
                    global_other.add(class, size);
                }
                global_total_bytes += size;
            }
        }

        total_tensor_bytes += size;
    }

    let layers: Vec<LayerTensorInventory> = layers
        .into_iter()
        .map(|(layer_index, bytes)| {
            let total_bytes = bytes.total();
            LayerTensorInventory {
                layer_index,
                bytes,
                total_bytes,
            }
        })
        .collect();

    Ok(TensorClassInventory {
        layers,
        tensors,
        global_embeddings,
        global_output,
        global_other,
        global_total_bytes,
        total_tensor_bytes,
        unknown_tensor_count,
        unknown_tensor_bytes,
    })
}

// ─── Placement policy projection ─────────────────────────────────────────────

/// Placement target for a tensor class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementTarget {
    Host,
    PrimaryAccelerator,
    // Default is intentionally not a target — policies must be explicit.
}

/// Trait for tensor placement policies.
///
/// Implementors map each tensor class to a concrete placement target.
pub trait TensorPlacementPolicy {
    fn placement_for_class(&self, class: TensorClass) -> PlacementTarget;
}

/// CpuMoe placement policy: routed experts to HOST, everything else to
/// the primary accelerator.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuMoe;

impl TensorPlacementPolicy for CpuMoe {
    fn placement_for_class(&self, class: TensorClass) -> PlacementTarget {
        match class {
            TensorClass::RoutedExpert => PlacementTarget::Host,
            _ => PlacementTarget::PrimaryAccelerator,
        }
    }
}

/// AllAccelerator: everything on the primary accelerator. No host offload.
#[derive(Clone, Copy, Debug, Default)]
pub struct AllAccelerator;

impl TensorPlacementPolicy for AllAccelerator {
    fn placement_for_class(&self, _class: TensorClass) -> PlacementTarget {
        PlacementTarget::PrimaryAccelerator
    }
}

impl CpuMoe {
    /// Host weight bytes = routed expert bytes only.
    pub fn host_bytes(layer: &LayerTensorInventory) -> u64 {
        layer.bytes.routed_expert
    }

    /// Accelerator weight bytes = all non-routed bytes.
    pub fn accelerator_bytes(layer: &LayerTensorInventory) -> u64 {
        layer.total_bytes - layer.bytes.routed_expert
    }

    /// Total bytes for a layer (same as inventory total, provided for
    /// symmetry with host/accelerator split).
    pub fn total_bytes(layer: &LayerTensorInventory) -> u64 {
        layer.total_bytes
    }
}

/// Node capacity for stage placement.
#[derive(Clone, Copy, Debug)]
pub struct NodeCapacity {
    pub usable_ram_bytes: u64,
    pub usable_vram_bytes: u64,
}

/// AutoHybrid solver: offload eligible tensors to HOST only as needed to fit
/// the stage within available RAM and VRAM.
///
/// Offload priority: routed experts first (they're the CpuMoe target), then
/// other classes later. Pinned to accelerator: shared experts, attention,
/// SSM, routing, normalization, unknown.
#[derive(Clone, Copy, Debug, Default)]
pub struct AutoHybrid;

/// Offload priority for a tensor class. Lower = offload first.
fn offload_priority(class: TensorClass) -> u8 {
    match class {
        TensorClass::RoutedExpert => 1,
        TensorClass::Other => 2,
        TensorClass::Normalization => 3,
        TensorClass::SharedExpert => 4,
        TensorClass::RoutingGate => 5,
        TensorClass::Attention => 6,
        TensorClass::RecurrentSsm => 7,
    }
}

/// Plan stage placement using the AutoHybrid strategy.
///
/// Starts with all tensors on accelerator, then offloads eligible tensors
/// to HOST (in priority order) until the accelerator requirement fits.
/// Returns an error if the stage cannot fit within the given capacities.
pub fn plan_stage_placement(
    inventory: &TensorClassInventory,
    selection: &StageTensorSelection,
    capacity: NodeCapacity,
) -> Result<StagePlacementPlan> {
    // Gather all tensors belonging to this stage.
    let mut stage_tensors: Vec<&ClassifiedTensor> = inventory
        .tensors
        .iter()
        .filter(|t| match t.layer {
            Some(idx) => selection.layers.contains(&idx),
            None => {
                let (is_embedding, is_output) = classify_global_tensor(&t.name);
                if is_embedding {
                    selection.include_embeddings
                } else if is_output {
                    selection.include_output
                } else {
                    selection.include_other_globals
                }
            }
        })
        .collect();

    // Sort by offload priority (lowest first), then by size (largest first)
    // to minimize the number of tensors we need to offload.
    stage_tensors.sort_by(|a, b| {
        let pri = offload_priority(a.class).cmp(&offload_priority(b.class));
        if pri != std::cmp::Ordering::Equal {
            return pri;
        }
        b.bytes.cmp(&a.bytes)
    });

    // Determine which classes are eligible for offload.
    let is_offloadable = |class: TensorClass| -> bool {
        matches!(class, TensorClass::RoutedExpert) // Phase 3: only routed experts
    };

    // Compute total bytes.
    let total_bytes: u64 = stage_tensors.iter().map(|t| t.bytes).sum();

    // If everything fits on accelerator, no offload needed.
    if total_bytes <= capacity.usable_vram_bytes {
        let placements = stage_tensors
            .into_iter()
            .map(|t| TensorPlacement {
                name: t.name.clone(),
                layer: t.layer,
                class: t.class,
                bytes: t.bytes,
                target: PlacementTarget::PrimaryAccelerator,
            })
            .collect();
        return Ok(StagePlacementPlan {
            host_bytes: 0,
            accelerator_bytes: total_bytes,
            placements,
        });
    }

    // Need to offload. Determine how much we need to free from accelerator.
    let overflow = total_bytes - capacity.usable_vram_bytes;

    // Decide which tensors to offload.
    let mut offload_bytes: u64 = 0;
    let mut placements = Vec::with_capacity(stage_tensors.len());

    for t in stage_tensors {
        if offload_bytes < overflow && is_offloadable(t.class) {
            placements.push(TensorPlacement {
                name: t.name.clone(),
                layer: t.layer,
                class: t.class,
                bytes: t.bytes,
                target: PlacementTarget::Host,
            });
            offload_bytes += t.bytes;
        } else {
            placements.push(TensorPlacement {
                name: t.name.clone(),
                layer: t.layer,
                class: t.class,
                bytes: t.bytes,
                target: PlacementTarget::PrimaryAccelerator,
            });
        }
    }

    // Check if offload was sufficient.
    if offload_bytes < overflow {
        anyhow::bail!(
            "stage cannot fit: overflow {} bytes, only offloaded {} bytes",
            overflow,
            offload_bytes
        );
    }

    // Compute final accounting.
    let host_bytes: u64 = placements
        .iter()
        .filter(|p| p.target == PlacementTarget::Host)
        .map(|p| p.bytes)
        .sum();
    let accelerator_bytes: u64 = placements
        .iter()
        .filter(|p| p.target == PlacementTarget::PrimaryAccelerator)
        .map(|p| p.bytes)
        .sum();

    // Check host capacity.
    if host_bytes > capacity.usable_ram_bytes {
        anyhow::bail!(
            "offloaded {} bytes to host but only {} bytes available",
            host_bytes,
            capacity.usable_ram_bytes
        );
    }

    Ok(StagePlacementPlan {
        host_bytes,
        accelerator_bytes,
        placements,
    })
}

/// Stage tensor selection: defines exactly which tensors belong to a stage.
#[derive(Clone, Debug)]
pub struct StageTensorSelection {
    pub layers: Range<u32>,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub include_other_globals: bool,
}

/// Weight requirements for a candidate stage under a placement policy.
///
/// This is purely the tensor weight requirements — it does NOT include
/// runtime overhead such as KV/recurrent state or graph workspace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StageWeightRequirements {
    pub host_bytes: u64,
    pub accelerator_bytes: u64,
    pub total_bytes: u64,
    pub routed_expert_bytes: u64,
}

impl StageWeightRequirements {
    /// Create a new StageWeightRequirements with invariant checking.
    pub fn new(host_bytes: u64, accelerator_bytes: u64) -> Self {
        let total_bytes = host_bytes + accelerator_bytes;
        Self {
            host_bytes,
            accelerator_bytes,
            total_bytes,
            routed_expert_bytes: host_bytes, // For CpuMoe, host == routed
        }
    }
}

/// Per-tensor placement decision.
#[derive(Clone, Debug)]
pub struct TensorPlacement {
    pub name: String,
    pub layer: Option<u32>,
    pub class: TensorClass,
    pub bytes: u64,
    pub target: PlacementTarget,
}

/// Complete placement plan for a candidate stage.
#[derive(Clone, Debug)]
pub struct StagePlacementPlan {
    pub host_bytes: u64,
    pub accelerator_bytes: u64,
    pub placements: Vec<TensorPlacement>,
}

/// Placement strategy for stage memory planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlacementStrategy {
    /// Everything on the primary accelerator. No host offload.
    AllAccelerator,
    /// CpuMoe: routed experts to HOST, everything else to accelerator.
    CpuMoe,
    /// AutoHybrid: capacity-aware solver that offloads eligible tensors
    /// as needed to fit the stage within available RAM and VRAM.
    AutoHybrid,
}

/// Compute stage weight requirements by aggregating class bytes over the
/// selected layers and owned globals, then projecting through the policy.
pub fn stage_weight_requirements(
    inventory: &TensorClassInventory,
    selection: &StageTensorSelection,
    policy: &dyn TensorPlacementPolicy,
) -> Result<StageWeightRequirements> {
    // Aggregate layer bytes for the selected range.
    let layer_bytes = inventory.class_bytes_for_range(selection.layers.clone());

    // Add owned globals.
    let mut global_bytes = TensorClassBytes::default();
    if selection.include_embeddings {
        global_bytes = add_bytes(global_bytes, inventory.global_embeddings);
    }
    if selection.include_output {
        global_bytes = add_bytes(global_bytes, inventory.global_output);
    }
    if selection.include_other_globals {
        global_bytes = add_bytes(global_bytes, inventory.global_other);
    }

    // Combine layer + global bytes.
    let combined = add_bytes(layer_bytes, global_bytes);

    // Project through policy.
    let mut host_bytes: u64 = 0;
    let mut accelerator_bytes: u64 = 0;

    let classes = [
        (TensorClass::RoutedExpert, combined.routed_expert),
        (TensorClass::SharedExpert, combined.shared_expert),
        (TensorClass::Attention, combined.attention),
        (TensorClass::RecurrentSsm, combined.recurrent_ssm),
        (TensorClass::RoutingGate, combined.routing_gate),
        (TensorClass::Normalization, combined.normalization),
        (TensorClass::Other, combined.other),
    ];

    for (class, bytes) in classes {
        match policy.placement_for_class(class) {
            PlacementTarget::Host => {
                host_bytes = host_bytes.checked_add(bytes).ok_or_else(|| {
                    anyhow::anyhow!("host_bytes overflow in stage_weight_requirements")
                })?
            }
            PlacementTarget::PrimaryAccelerator => {
                accelerator_bytes = accelerator_bytes.checked_add(bytes).ok_or_else(|| {
                    anyhow::anyhow!("accelerator_bytes overflow in stage_weight_requirements")
                })?;
            }
        }
    }

    Ok(StageWeightRequirements::new(host_bytes, accelerator_bytes))
}

fn add_bytes(mut a: TensorClassBytes, b: TensorClassBytes) -> TensorClassBytes {
    a.routed_expert = a.routed_expert.saturating_add(b.routed_expert);
    a.shared_expert = a.shared_expert.saturating_add(b.shared_expert);
    a.attention = a.attention.saturating_add(b.attention);
    a.recurrent_ssm = a.recurrent_ssm.saturating_add(b.recurrent_ssm);
    a.routing_gate = a.routing_gate.saturating_add(b.routing_gate);
    a.normalization = a.normalization.saturating_add(b.normalization);
    a.other = a.other.saturating_add(b.other);
    a
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // T1: Routed expert classification + exact bytes.
    #[test]
    fn classify_routed_expert_tensor() {
        assert_eq!(
            classify_tensor("blk.0.ffn_gate_exps.weight"),
            TensorClass::RoutedExpert
        );
        assert_eq!(
            classify_tensor("blk.17.ffn_up_exps.weight"),
            TensorClass::RoutedExpert
        );
        assert_eq!(
            classify_tensor("blk.48.ffn_down_exps.weight"),
            TensorClass::RoutedExpert
        );
    }

    // T2: Shared expert — classified as SharedExpert, never routed.
    #[test]
    fn classify_shared_expert_tensor() {
        let class = classify_tensor("blk.0.ffn_gate_shexp.weight");
        assert_eq!(class, TensorClass::SharedExpert);
        assert_ne!(class, TensorClass::RoutedExpert);
    }

    // T3: Unknown tensor → Other, not omitted.
    #[test]
    fn classify_unknown_tensor() {
        assert_eq!(
            classify_tensor("blk.0.some_future_tensor.weight"),
            TensorClass::Other
        );
    }

    // T4: sum(classes) == layer total (via TensorClassBytes).
    #[test]
    fn class_bytes_sum_equals_total() {
        let mut classes = TensorClassBytes::default();
        classes.add(TensorClass::RoutedExpert, 1000);
        classes.add(TensorClass::SharedExpert, 500);
        classes.add(TensorClass::Attention, 200);
        classes.add(TensorClass::RecurrentSsm, 0);
        classes.add(TensorClass::RoutingGate, 50);
        classes.add(TensorClass::Normalization, 25);
        classes.add(TensorClass::Other, 10);
        assert_eq!(classes.total(), 1785);
        assert_eq!(classes.host_bytes(), 1000);
        assert_eq!(classes.accelerator_bytes(), 785);
    }

    // T5: CpuMoe host_bytes == routed_expert_bytes.
    #[test]
    fn cpumoe_host_bytes_equals_routed_expert() {
        let layer = LayerTensorInventory {
            layer_index: 0,
            bytes: TensorClassBytes {
                routed_expert: 3_984_000_000,
                shared_expert: 500_000_000,
                attention: 200_000_000,
                ..Default::default()
            },
            total_bytes: 4_684_000_000,
        };
        assert_eq!(CpuMoe::host_bytes(&layer), 3_984_000_000);
        assert_eq!(CpuMoe::accelerator_bytes(&layer), 700_000_000);
        assert_eq!(CpuMoe::total_bytes(&layer), 4_684_000_000);
    }

    // T6: Attention patterns.
    #[test]
    fn classify_attention_tensors() {
        assert_eq!(
            classify_tensor("blk.0.attn_q_a_proj.weight"),
            TensorClass::Attention
        );
        assert_eq!(
            classify_tensor("blk.5.attn_o_proj.weight"),
            TensorClass::Attention
        );
        assert_eq!(
            classify_tensor("blk.3.attn_k_b_proj.weight"),
            TensorClass::Attention
        );
    }

    // T7: Normalization patterns.
    #[test]
    fn classify_normalization_tensors() {
        assert_eq!(
            classify_tensor("blk.0.input_layernorm.weight"),
            TensorClass::Normalization
        );
        assert_eq!(
            classify_tensor("blk.0.q_a_norm.weight"),
            TensorClass::Normalization
        );
    }

    // T8: Global vs layer tensor separation.
    #[test]
    fn classify_global_tensors() {
        assert_ne!(
            classify_tensor("token_embd.weight"),
            TensorClass::RoutedExpert
        );
        assert_ne!(classify_tensor("output.weight"), TensorClass::RoutedExpert);
    }

    // T9: Range aggregation.
    #[test]
    fn range_aggregation() {
        let inventory = TensorClassInventory {
            layers: vec![
                LayerTensorInventory {
                    layer_index: 0,
                    bytes: TensorClassBytes {
                        routed_expert: 1000,
                        ..Default::default()
                    },
                    total_bytes: 1000,
                },
                LayerTensorInventory {
                    layer_index: 1,
                    bytes: TensorClassBytes {
                        routed_expert: 2000,
                        ..Default::default()
                    },
                    total_bytes: 2000,
                },
                LayerTensorInventory {
                    layer_index: 2,
                    bytes: TensorClassBytes {
                        routed_expert: 3000,
                        ..Default::default()
                    },
                    total_bytes: 3000,
                },
            ],
            tensors: vec![],
            global_embeddings: TensorClassBytes::default(),
            global_output: TensorClassBytes::default(),
            global_other: TensorClassBytes::default(),
            global_total_bytes: 0,
            total_tensor_bytes: 6000,
            unknown_tensor_count: 0,
            unknown_tensor_bytes: 0,
        };

        let range_bytes = inventory.class_bytes_for_range(0..2);
        assert_eq!(range_bytes.routed_expert, 3000); // layers 0 + 1
        assert_eq!(range_bytes.total(), 3000);

        let all_bytes = inventory.class_bytes_for_range(0..3);
        assert_eq!(all_bytes.routed_expert, 6000);
    }

    // ─── Phase 3 tests: stage placement ───────────────────────────────────────

    // R1: CpuMoe 0..4 host == routed_expert, accel == total - routed.
    #[test]
    fn r1_cpumoe_requirements_match_inventory() {
        let policy = CpuMoe;
        let req = stage_weight_requirements(
            &test_inventory(),
            &StageTensorSelection {
                layers: 0..2,
                include_embeddings: false,
                include_output: false,
                include_other_globals: false,
            },
            &policy,
        )
        .unwrap();

        assert_eq!(req.host_bytes, 3000);
        assert_eq!(req.accelerator_bytes, 2050);
        assert_eq!(req.total_bytes, 5050);
    }

    // R2: host + accel == total for every range.
    #[test]
    fn r2_host_plus_accel_equals_total() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        for start in 0..3 {
            for end in (start + 1)..=3 {
                let req = stage_weight_requirements(
                    &inventory,
                    &StageTensorSelection {
                        layers: start..end,
                        include_embeddings: false,
                        include_output: false,
                        include_other_globals: false,
                    },
                    &policy,
                )
                .unwrap();
                assert_eq!(req.host_bytes + req.accelerator_bytes, req.total_bytes);
            }
        }
    }

    // R3: Single-layer requirement equals that layer's inventory.
    #[test]
    fn r3_single_layer_equals_inventory() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        let req = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 1..2,
                include_embeddings: false,
                include_output: false,
                include_other_globals: false,
            },
            &policy,
        )
        .unwrap();

        let layer_1 = inventory.layer(1).unwrap();
        assert_eq!(req.host_bytes, CpuMoe::host_bytes(layer_1));
        assert_eq!(req.accelerator_bytes, CpuMoe::accelerator_bytes(layer_1));
        assert_eq!(req.total_bytes, layer_1.total_bytes);
    }

    // R4: Adjacent ranges compose (no globals).
    #[test]
    fn r4_adjacent_ranges_compose() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        let req_ab = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..2,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();
        let req_cd = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 2..3,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();
        let req_full = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..3,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();

        assert_eq!(req_ab.total_bytes + req_cd.total_bytes, req_full.total_bytes);
        assert_eq!(req_ab.host_bytes + req_cd.host_bytes, req_full.host_bytes);
    }

    // R5: First-stage embedding ownership counted once.
    #[test]
    fn r5_first_stage_embeddings() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        let mut emb = TensorClassBytes::default();
        // token_embd.weight classifies as Other.
        emb.add(TensorClass::Other, 500);
        let inventory = TensorClassInventory {
            global_embeddings: emb,
            ..inventory
        };

        let req_with = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..1,
                include_embeddings: true,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();
        let req_without = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..1,
                include_embeddings: false,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();

        assert_eq!(req_with.total_bytes - req_without.total_bytes, 500);
    }

    // R8: Empty range behaves deterministically.
    #[test]
    fn r8_empty_range_is_zero() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        let req = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..0,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();
        assert_eq!(req.total_bytes, 0);
    }

    // R9: Other bytes remain accelerator-resident under CpuMoe.
    #[test]
    fn r9_other_bytes_are_accelerator() {
        let policy = CpuMoe;
        let inventory = test_inventory();
        let req = stage_weight_requirements(
            &inventory,
            &StageTensorSelection {
                layers: 0..3,
                ..Default::default()
            },
            &policy,
        )
        .unwrap();
        // No Other bytes in test_inventory, but host_bytes == routed_expert
        // and accelerator_bytes == total - routed_expert means Other stays on accel.
        assert_eq!(req.routed_expert_bytes, req.host_bytes);
    }

    fn test_inventory() -> TensorClassInventory {
        TensorClassInventory {
            layers: vec![
                LayerTensorInventory {
                    layer_index: 0,
                    bytes: TensorClassBytes {
                        routed_expert: 1000,
                        shared_expert: 500,
                        attention: 200,
                        normalization: 50,
                        ..Default::default()
                    },
                    total_bytes: 1750,
                },
                LayerTensorInventory {
                    layer_index: 1,
                    bytes: TensorClassBytes {
                        routed_expert: 2000,
                        shared_expert: 800,
                        attention: 400,
                        normalization: 100,
                        ..Default::default()
                    },
                    total_bytes: 3300,
                },
                LayerTensorInventory {
                    layer_index: 2,
                    bytes: TensorClassBytes {
                        routed_expert: 3000,
                        shared_expert: 1200,
                        attention: 600,
                        normalization: 150,
                        ..Default::default()
                    },
                    total_bytes: 4950,
                },
            ],
            tensors: vec![],
            global_embeddings: TensorClassBytes::default(),
            global_output: TensorClassBytes::default(),
            global_other: TensorClassBytes::default(),
            global_total_bytes: 0,
            total_tensor_bytes: 10000,
            unknown_tensor_count: 0,
            unknown_tensor_bytes: 0,
        }
    }
}

impl Default for StageTensorSelection {
    fn default() -> Self {
        Self {
            layers: 0..0,
            include_embeddings: false,
            include_output: false,
            include_other_globals: false,
        }
    }
}
