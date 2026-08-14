use std::collections::BTreeMap;
use std::fmt;
use std::ops::Range;

use anyhow::Result;

use crate::ModelInfo;
use crate::TensorInfo;

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

/// Full model tensor-class byte inventory.
#[derive(Clone, Debug)]
pub struct TensorClassInventory {
    pub layers: Vec<LayerTensorInventory>,
    pub global: TensorClassBytes,
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

// ─── Inventory builder ───────────────────────────────────────────────────────

/// Build a TensorClassInventory from a Skippy ModelInfo handle.
///
/// Reads exact tensor byte sizes from the GGUF metadata. Every tensor is
/// counted exactly once; unknown tensors go to `Other`, not omitted.
pub fn build_tensor_class_inventory(info: &ModelInfo) -> Result<TensorClassInventory> {
    let count = info.tensor_count()?;
    let mut layers: BTreeMap<u32, TensorClassBytes> = BTreeMap::new();
    let mut global = TensorClassBytes::default();
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

        match tensor.layer_index {
            Some(idx) => {
                let idx = idx as u32;
                layers.entry(idx).or_default().add(class, size);
            }
            None => {
                global.add(class, size);
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
        global,
        global_total_bytes,
        total_tensor_bytes,
        unknown_tensor_count,
        unknown_tensor_bytes,
    })
}

// ─── CpuMoe policy ───────────────────────────────────────────────────────────

/// CpuMoe placement policy: routed experts to HOST, everything else to
/// the primary accelerator.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuMoe;

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
        // These should not be routed experts.
        assert_ne!(
            classify_tensor("token_embd.weight"),
            TensorClass::RoutedExpert
        );
        assert_ne!(
            classify_tensor("output.weight"),
            TensorClass::RoutedExpert
        );
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
            global: TensorClassBytes::default(),
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
}
