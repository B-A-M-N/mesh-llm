use std::collections::BTreeMap;

use anyhow::Result;
use crate::ModelInfo;
use crate::TensorInfo;

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
    /// SSM states (RWKV-style).
    Ssm = 3,
    /// Router / gating weights (input-side).
    Routing = 4,
    /// Layer norms, embeddings, output, metadata.
    Normalization = 5,
    /// Unknown / unclassified (conservatively treated as accelerator).
    Other = 6,
}

/// Per-layer tensor-class breakdown.
#[derive(Clone, Debug, Default)]
pub struct LayerTensorClasses {
    pub layer_index: u32,
    pub routed_expert_bytes: u64,
    pub shared_expert_bytes: u64,
    pub attention_bytes: u64,
    pub ssm_bytes: u64,
    pub routing_bytes: u64,
    pub normalization_bytes: u64,
    pub other_bytes: u64,
    pub total_bytes: u64,
}

impl LayerTensorClasses {
    pub fn accelerator_bytes(&self) -> u64 {
        self.shared_expert_bytes
            + self.attention_bytes
            + self.ssm_bytes
            + self.routing_bytes
            + self.normalization_bytes
            + self.other_bytes
    }

    pub fn host_bytes(&self) -> u64 {
        self.routed_expert_bytes
    }
}

/// Full model tensor-class inventory.
#[derive(Clone, Debug)]
pub struct TensorClassInventory {
    pub model_ref: String,
    pub global_bytes: u64,
    pub layers: Vec<LayerTensorClasses>,
}

/// Qwen3.5-122B MoE layer naming conventions.
/// Each MoE layer contains:
/// - 128 routed experts (gate/up/down), e.g. `blk.0.ffn_gate_exps.weight`
/// - 1 shared expert (gate/up/down),   e.g. `blk.0.ffn_gate_shexp.weight`
/// - attention: q_a_proj, q_b_proj, k_a_proj, k_b_proj, v_a_proj, v_b_proj, o_proj
/// - norms: input_layernorm, post_attention_layernorm, q_a_norm, k_a_norm, q_b_norm, k_b_norm
/// - routing: e.g. `blk.0.ffn_gate_exps.weight` (the router for MoE)
///   Actually the router in Qwen3 is part of `blk.N.attn_q_b_proj` etc.
///   For our policy, "routing" = the gating network that selects experts.

const ROUTED_EXPERT_PATTERNS: &[&str] = &[
    "ffn_gate_exps",
    "ffn_up_exps",
    "ffn_down_exps",
];

const SHARED_EXPERT_PATTERNS: &[&str] = &[
    "ffn_gate_shexp",
    "ffn_up_shexp",
    "ffn_down_shexp",
];

const ATTENTION_PATTERNS: &[&str] = &[
    "attn_q_a_proj",
    "attn_q_b_proj",
    "attn_k_a_proj",
    "attn_k_b_proj",
    "attn_v_a_proj",
    "attn_v_b_proj",
    "attn_o_proj",
];

const SSM_PATTERNS: &[&str] = &[
    "x_proj",
    "dt_proj",
    "A_log",
    "D_broadcast",
    "out_proj",
];

const ROUTING_PATTERNS: &[&str] = &[
    "attn_gate",       // some architectures
    "router",          // explicit router
    "ffn_gate",        // router for MoE (only if not _exps or _shexp)
];

const NORMALIZATION_PATTERNS: &[&str] = &[
    "norm",
    "layernorm",
    "input_layernorm",
    "post_attention_layernorm",
    "q_a_norm",
    "q_b_norm",
    "k_a_norm",
];

/// Classify a single tensor by name (and optionally role hint).
pub fn classify_tensor(name: &str) -> TensorClass {
    // Exact match priority: routed experts first (they're the target of CpuMoe).
    for pat in ROUTED_EXPERT_PATTERNS {
        if name.contains(pat) {
            return TensorClass::RoutedExpert;
        }
    }

    for pat in SHARED_EXPERT_PATTERNS {
        if name.contains(pat) {
            return TensorClass::SharedExpert;
        }
    }

    for pat in SSM_PATTERNS {
        if name.contains(pat) {
            return TensorClass::Ssm;
        }
    }

    for pat in ATTENTION_PATTERNS {
        if name.contains(pat) {
            return TensorClass::Attention;
        }
    }

    for pat in NORMALIZATION_PATTERNS {
        if name.contains(pat) {
            return TensorClass::Normalization;
        }
    }

    // Routing must come after routed-expert check because "ffn_gate" would
    // match "ffn_gate_exps" (but we already returned above).
    for pat in ROUTING_PATTERNS {
        if name.contains(pat) {
            return TensorClass::Routing;
        }
    }

    TensorClass::Other
}

/// Build a TensorClassInventory from a Skippy model-info handle.
pub fn build_tensor_class_inventory(
    info: &ModelInfo,
    model_ref: String,
) -> Result<TensorClassInventory> {
    let count = info.tensor_count()?;
    let mut layers: BTreeMap<u32, LayerTensorClasses> = BTreeMap::new();
    let mut global_bytes: u64 = 0;

    for i in 0..count {
        let tensor = info.tensor_at(i)?;
        let name = &tensor.name;
        let class = classify_tensor(&name);
        let size = tensor.byte_size;

        let idx = match tensor.layer_index {
            Some(i) => i as u32,
            None => {
                // Global / non-layer tensor (embeddings, output, final norm, metadata).
                global_bytes += size;
                continue;
            }
        };
        let entry = layers.entry(idx).or_insert_with(|| LayerTensorClasses {
            layer_index: idx,
            ..Default::default()
        });

        match class {
            TensorClass::RoutedExpert => entry.routed_expert_bytes += size,
            TensorClass::SharedExpert => entry.shared_expert_bytes += size,
            TensorClass::Attention => entry.attention_bytes += size,
            TensorClass::Ssm => entry.ssm_bytes += size,
            TensorClass::Routing => entry.routing_bytes += size,
            TensorClass::Normalization => entry.normalization_bytes += size,
            TensorClass::Other => entry.other_bytes += size,
        }
        entry.total_bytes += size;
    }

    let layers: Vec<_> = layers.into_values().collect();

    Ok(TensorClassInventory {
        model_ref,
        global_bytes,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_routed_experts() {
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

    #[test]
    fn classify_shared_experts() {
        assert_eq!(
            classify_tensor("blk.0.ffn_gate_shexp.weight"),
            TensorClass::SharedExpert
        );
        assert_eq!(
            classify_tensor("blk.3.ffn_down_shexp.weight"),
            TensorClass::SharedExpert
        );
    }

    #[test]
    fn classify_attention() {
        assert_eq!(
            classify_tensor("blk.0.attn_q_a_proj.weight"),
            TensorClass::Attention
        );
        assert_eq!(
            classify_tensor("blk.5.attn_o_proj.weight"),
            TensorClass::Attention
        );
    }

    #[test]
    fn classify_norms() {
        assert_eq!(
            classify_tensor("blk.0.input_layernorm.weight"),
            TensorClass::Normalization
        );
        assert_eq!(
            classify_tensor("blk.0.q_a_norm.weight"),
            TensorClass::Normalization
        );
    }

    #[test]
    fn classify_unknown_as_other() {
        assert_eq!(
            classify_tensor("blk.0.some_future_tensor.weight"),
            TensorClass::Other
        );
    }
}
