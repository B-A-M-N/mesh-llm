use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::{NativeMtpDraftOrigin, NativeMtpHybridProposal};
use crate::frontend::SpeculativeDecodeConfig;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) struct NativeMtpDecodeOptions {
    pub(in crate::frontend) max_draft_tokens: usize,
    pub(in crate::frontend) min_draft_tokens: usize,
    pub(in crate::frontend) reject_cooldown_tokens: usize,
    pub(in crate::frontend) suppress_cooldown_drafts: bool,
    pub(in crate::frontend) suppress_cooldown_draft_limit: usize,
    pub(in crate::frontend) ngram_hybrid: bool,
    pub(in crate::frontend) ngram_size: usize,
    pub(in crate::frontend) ngram_initial_extension_tokens: usize,
    pub(in crate::frontend) ngram_max_proposal_tokens: usize,
    pub(in crate::frontend) ngram_tail_backoff_proposals: usize,
    pub(in crate::frontend) verify_window_min_tokens: usize,
    pub(in crate::frontend) verify_window_max_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) enum NativeMtpTrimAction {
    None,
    FullSession,
}

pub(in crate::frontend) fn native_mtp_trim_action(
    committed_positions: usize,
    consumed_positions: usize,
) -> NativeMtpTrimAction {
    if committed_positions == consumed_positions {
        NativeMtpTrimAction::None
    } else {
        NativeMtpTrimAction::FullSession
    }
}

impl NativeMtpDecodeOptions {
    pub(in crate::frontend) fn from_config(config: &SpeculativeDecodeConfig) -> Self {
        Self {
            max_draft_tokens: config.native_mtp.max_draft_tokens.max(1),
            min_draft_tokens: config
                .native_mtp
                .min_draft_tokens
                .min(config.native_mtp.max_draft_tokens.max(1)),
            reject_cooldown_tokens: config.native_mtp.reject_cooldown_tokens,
            suppress_cooldown_drafts: config.native_mtp.suppress_cooldown_drafts,
            suppress_cooldown_draft_limit: config.native_mtp.suppress_cooldown_draft_limit,
            ngram_hybrid: config.extension.is_some() && config.ngram.is_some(),
            ngram_size: config.ngram.as_ref().map_or(0, |ngram| ngram.min_ngram),
            ngram_initial_extension_tokens: config
                .extension
                .as_ref()
                .map_or(0, |extension| extension.initial_tokens),
            ngram_max_proposal_tokens: config
                .extension
                .as_ref()
                .map_or(0, |extension| extension.max_tokens),
            ngram_tail_backoff_proposals: config
                .extension
                .as_ref()
                .map_or(0, |extension| extension.tail_backoff_proposals),
            verify_window_min_tokens: config.verify_window.min_tokens.max(1),
            verify_window_max_tokens: config.verify_window.max_tokens.max(1),
        }
    }

    pub(in crate::frontend) fn verify_window_bounds(self) -> (usize, usize) {
        let min = self.verify_window_min_tokens.max(1);
        (min, self.verify_window_max_tokens.max(min))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::frontend) struct AdaptiveVerifyWindow {
    min_tokens: usize,
    max_tokens: usize,
    current_tokens: usize,
}

impl AdaptiveVerifyWindow {
    pub(in crate::frontend) fn new(options: NativeMtpDecodeOptions) -> Self {
        let (min_tokens, max_tokens) = options.verify_window_bounds();
        Self {
            min_tokens,
            max_tokens,
            current_tokens: max_tokens.min(2).max(min_tokens),
        }
    }

    pub(in crate::frontend) fn width(self, available_tokens: usize) -> usize {
        self.current_tokens.min(available_tokens)
    }

    pub(in crate::frontend) fn observe(&mut self, full_accept: bool) -> bool {
        let previous = self.current_tokens;
        if full_accept {
            self.current_tokens = self.current_tokens.saturating_add(1).min(self.max_tokens);
        } else {
            self.current_tokens = self.current_tokens.saturating_sub(1).max(self.min_tokens);
        }
        self.current_tokens != previous
    }

    pub(in crate::frontend) fn current_tokens(self) -> usize {
        self.current_tokens
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::frontend) struct NativeMtpDecodeCounters {
    suppressed_cooldown_draft_count: usize,
    verify_window_verification_count: usize,
    initial_serial_verification_count: usize,
    initial_serial_accepted_count: usize,
    serial_after_gap_verification_count: usize,
    serial_after_gap_accepted_count: usize,
    verify_next_verification_count: usize,
    verify_next_accepted_count: usize,
    verify_next_draft_available_count: usize,
    verify_next_draft_adopted_count: usize,
    hybrid_native_prefix_available_count: usize,
    hybrid_ngram_continuation_available_count: usize,
    hybrid_ngram_mtp_prefix_agreement_count: usize,
    hybrid_ngram_mtp_prefix_disagreement_count: usize,
    hybrid_proposal_token_count: usize,
    hybrid_accepted_token_count: usize,
    hybrid_accepted_tail_token_count: usize,
    hybrid_native_mtp_token_count: usize,
    hybrid_ngram_token_count: usize,
    hybrid_pure_ngram_proposal_count: usize,
    hybrid_accepted_native_mtp_token_count: usize,
    hybrid_ngram_tail_rejection_count: usize,
    hybrid_ngram_sidecar_backoff_count: usize,
    adaptive_verify_window_count: usize,
    adaptive_verify_window_width_sum: usize,
    adaptive_verify_window_width_min: usize,
    adaptive_verify_window_width_max: usize,
    adaptive_verify_window_grow_count: usize,
    adaptive_verify_window_shrink_count: usize,
    // Phase 0 survival curve: per logical composite proposal, how far the draft
    // stayed correct, split by source. `eligible_ge[k]` counts proposals that
    // actually offered depth k+1 (index 0 = depth 1); `survived_ge[k]` counts
    // those accepted through depth k+1. P(A>=k) = survived_ge[k-1]/eligible_ge[k-1].
    // Lets us measure acceptance depth `g` per source x workload (benchmark-side)
    // to decide whether pipelining depth pays. Stale-drained proposals excluded.
    logical_proposal_count: usize,
    native_eligible_ge: [usize; SURVIVAL_DEPTHS],
    native_survived_ge: [usize; SURVIVAL_DEPTHS],
    ngram_eligible_ge: [usize; SURVIVAL_DEPTHS],
    ngram_survived_ge: [usize; SURVIVAL_DEPTHS],
}

/// Survival curve tracks depths 1..=8 (native MTP max draft depth is 8).
pub(in crate::frontend) const SURVIVAL_DEPTHS: usize = 8;

impl NativeMtpDecodeCounters {
    pub(in crate::frontend) fn verify_window_verification_count(&self) -> usize {
        self.verify_window_verification_count
    }

    pub(in crate::frontend) fn observe_suppressed_cooldown_draft(&mut self) {
        self.suppressed_cooldown_draft_count += 1;
    }

    pub(in crate::frontend) fn observe_verify_window_verification(
        &mut self,
        origin: NativeMtpDraftOrigin,
        accepted: bool,
    ) {
        self.verify_window_verification_count += 1;
        match origin {
            NativeMtpDraftOrigin::InitialSerial => {
                self.initial_serial_verification_count += 1;
                if accepted {
                    self.initial_serial_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::SerialAfterGap => {
                self.serial_after_gap_verification_count += 1;
                if accepted {
                    self.serial_after_gap_accepted_count += 1;
                }
            }
            NativeMtpDraftOrigin::VerifyNext => {
                self.verify_next_verification_count += 1;
                if accepted {
                    self.verify_next_accepted_count += 1;
                }
            }
        }
    }

    pub(in crate::frontend) fn observe_verify_next_draft(
        &mut self,
        available: bool,
        adopted: bool,
    ) {
        if available {
            self.verify_next_draft_available_count += 1;
        }
        if adopted {
            self.verify_next_draft_adopted_count += 1;
        }
    }

    pub(in crate::frontend) fn observe_hybrid_proposal(
        &mut self,
        proposal: &NativeMtpHybridProposal,
        accepted_token_count: usize,
    ) {
        self.hybrid_native_prefix_available_count +=
            usize::from(proposal.native_mtp_token_count() > 0);
        self.hybrid_ngram_continuation_available_count +=
            usize::from(proposal.ngram_span_available());
        self.hybrid_ngram_mtp_prefix_agreement_count +=
            usize::from(proposal.ngram_mtp_prefix_agreed());
        self.hybrid_ngram_mtp_prefix_disagreement_count +=
            usize::from(proposal.ngram_mtp_prefix_disagreed());
        self.hybrid_proposal_token_count += proposal.tokens().len();
        self.hybrid_accepted_token_count += accepted_token_count;
        self.hybrid_accepted_tail_token_count +=
            accepted_token_count.saturating_sub(proposal.native_mtp_token_count());
        self.hybrid_accepted_native_mtp_token_count +=
            accepted_token_count.min(proposal.native_mtp_token_count());
        self.hybrid_native_mtp_token_count += proposal.native_mtp_token_count();
        self.hybrid_ngram_token_count += proposal.ngram_token_count();
        self.hybrid_pure_ngram_proposal_count += usize::from(proposal.is_pure_ngram());
        self.observe_survival(
            proposal.native_mtp_token_count(),
            proposal.ngram_token_count(),
            accepted_token_count,
        );
    }

    /// Feed the per-source survival curve for one retired logical proposal.
    ///
    /// The proposal is a single linear candidate stream: `native` MTP tokens
    /// first, then `ngram` tail tokens. `accepted` is how many led tokens the
    /// target actually confirmed (including the free-target token when it
    /// matched the next buffered candidate — the caller already folds that in).
    ///
    /// Decomposition (expert-reviewed): native survival is `min(accepted, N)`;
    /// ngram survival only counts once the native prefix fully survived, so a
    /// native failure never contaminates ngram quality. `eligible_ge[k]` is
    /// only incremented when the source actually offered depth `k+1`.
    fn observe_survival(&mut self, native: usize, ngram: usize, accepted: usize) {
        self.logical_proposal_count += 1;
        let native_survived = accepted.min(native);
        // Tail is only eligible/measured after the whole native prefix survived.
        let ngram_survived = if accepted >= native {
            (accepted - native).min(ngram)
        } else {
            0
        };
        for depth in 1..=SURVIVAL_DEPTHS {
            let idx = depth - 1;
            if native >= depth {
                self.native_eligible_ge[idx] += 1;
                if native_survived >= depth {
                    self.native_survived_ge[idx] += 1;
                }
            }
            // Ngram tail depth is only meaningful when the native prefix fully
            // survived (else the tail was never actually reached/verified).
            if ngram >= depth && accepted >= native {
                self.ngram_eligible_ge[idx] += 1;
                if ngram_survived >= depth {
                    self.ngram_survived_ge[idx] += 1;
                }
            }
        }
    }

    pub(in crate::frontend) fn observe_ngram_tail_rejection(&mut self) {
        self.hybrid_ngram_tail_rejection_count += 1;
        self.hybrid_ngram_sidecar_backoff_count += 1;
    }

    pub(in crate::frontend) fn observe_adaptive_verify_window(
        &mut self,
        width: usize,
        previous_width: usize,
        next_width: usize,
    ) {
        self.adaptive_verify_window_count += 1;
        self.adaptive_verify_window_width_sum += width;
        self.adaptive_verify_window_width_min = if self.adaptive_verify_window_width_min == 0 {
            width
        } else {
            self.adaptive_verify_window_width_min.min(width)
        };
        self.adaptive_verify_window_width_max = self.adaptive_verify_window_width_max.max(width);
        self.adaptive_verify_window_grow_count += usize::from(next_width > previous_width);
        self.adaptive_verify_window_shrink_count += usize::from(next_width < previous_width);
    }

    pub(in crate::frontend) fn insert_summary_attrs(
        &self,
        attrs: &mut BTreeMap<String, Value>,
        options: NativeMtpDecodeOptions,
    ) {
        attrs.insert(
            "llama_stage.native_mtp.reject_cooldown_tokens".to_string(),
            json!(options.reject_cooldown_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.max_draft_tokens".to_string(),
            json!(options.max_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.min_draft_tokens".to_string(),
            json!(options.min_draft_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_drafts".to_string(),
            json!(options.suppress_cooldown_drafts),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppress_cooldown_draft_limit".to_string(),
            json!(options.suppress_cooldown_draft_limit),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_hybrid".to_string(),
            json!(options.ngram_hybrid),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_size".to_string(),
            json!(options.ngram_size),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_initial_extension_tokens".to_string(),
            json!(options.ngram_initial_extension_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_max_proposal_tokens".to_string(),
            json!(options.ngram_max_proposal_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.ngram_tail_backoff_proposals".to_string(),
            json!(options.ngram_tail_backoff_proposals),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_window_min_tokens".to_string(),
            json!(options.verify_window_min_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_window_max_tokens".to_string(),
            json!(options.verify_window_max_tokens),
        );
        attrs.insert(
            "llama_stage.native_mtp.suppressed_cooldown_draft_count".to_string(),
            json!(self.suppressed_cooldown_draft_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_window_verification_count".to_string(),
            json!(self.verify_window_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_verification_count".to_string(),
            json!(self.initial_serial_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.initial_serial_accepted_count".to_string(),
            json!(self.initial_serial_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_verification_count".to_string(),
            json!(self.serial_after_gap_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.serial_after_gap_accepted_count".to_string(),
            json!(self.serial_after_gap_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_verification_count".to_string(),
            json!(self.verify_next_verification_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_accepted_count".to_string(),
            json!(self.verify_next_accepted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_available_count".to_string(),
            json!(self.verify_next_draft_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.verify_next_draft_adopted_count".to_string(),
            json!(self.verify_next_draft_adopted_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_native_prefix_available_count".to_string(),
            json!(self.hybrid_native_prefix_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_continuation_available_count".to_string(),
            json!(self.hybrid_ngram_continuation_available_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_mtp_prefix_agreement_count".to_string(),
            json!(self.hybrid_ngram_mtp_prefix_agreement_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_mtp_prefix_disagreement_count".to_string(),
            json!(self.hybrid_ngram_mtp_prefix_disagreement_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_proposal_token_count".to_string(),
            json!(self.hybrid_proposal_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_accepted_token_count".to_string(),
            json!(self.hybrid_accepted_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_accepted_tail_token_count".to_string(),
            json!(self.hybrid_accepted_tail_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_native_mtp_token_count".to_string(),
            json!(self.hybrid_native_mtp_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_token_count".to_string(),
            json!(self.hybrid_ngram_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_accepted_native_mtp_token_count".to_string(),
            json!(self.hybrid_accepted_native_mtp_token_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_tail_rejection_count".to_string(),
            json!(self.hybrid_ngram_tail_rejection_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_ngram_sidecar_backoff_count".to_string(),
            json!(self.hybrid_ngram_sidecar_backoff_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.hybrid_pure_ngram_proposal_count".to_string(),
            json!(self.hybrid_pure_ngram_proposal_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_count".to_string(),
            json!(self.adaptive_verify_window_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_width_sum".to_string(),
            json!(self.adaptive_verify_window_width_sum),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_width_min".to_string(),
            json!(self.adaptive_verify_window_width_min),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_width_max".to_string(),
            json!(self.adaptive_verify_window_width_max),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_grow_count".to_string(),
            json!(self.adaptive_verify_window_grow_count),
        );
        attrs.insert(
            "llama_stage.native_mtp.adaptive_verify_window_shrink_count".to_string(),
            json!(self.adaptive_verify_window_shrink_count),
        );
    }

    fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert(
            "native_mtp_verify_window_verifications".to_string(),
            json!(self.verify_window_verification_count),
        );
        timings.insert(
            "native_mtp_hybrid_native_prefix_available".to_string(),
            json!(self.hybrid_native_prefix_available_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_continuation_available".to_string(),
            json!(self.hybrid_ngram_continuation_available_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_mtp_prefix_agreements".to_string(),
            json!(self.hybrid_ngram_mtp_prefix_agreement_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_mtp_prefix_disagreements".to_string(),
            json!(self.hybrid_ngram_mtp_prefix_disagreement_count),
        );
        timings.insert(
            "native_mtp_hybrid_proposed_tokens".to_string(),
            json!(self.hybrid_proposal_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_accepted_tokens".to_string(),
            json!(self.hybrid_accepted_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_accepted_tail_tokens".to_string(),
            json!(self.hybrid_accepted_tail_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_native_tokens".to_string(),
            json!(self.hybrid_native_mtp_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_tokens".to_string(),
            json!(self.hybrid_ngram_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_accepted_native_tokens".to_string(),
            json!(self.hybrid_accepted_native_mtp_token_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_tail_rejections".to_string(),
            json!(self.hybrid_ngram_tail_rejection_count),
        );
        timings.insert(
            "native_mtp_hybrid_ngram_sidecar_backoffs".to_string(),
            json!(self.hybrid_ngram_sidecar_backoff_count),
        );
        timings.insert(
            "native_mtp_hybrid_pure_ngram_proposals".to_string(),
            json!(self.hybrid_pure_ngram_proposal_count),
        );
        timings.insert(
            "native_mtp_adaptive_verify_windows".to_string(),
            json!(self.adaptive_verify_window_count),
        );
        timings.insert(
            "native_mtp_adaptive_verify_window_width_sum".to_string(),
            json!(self.adaptive_verify_window_width_sum),
        );
        timings.insert(
            "native_mtp_adaptive_verify_window_width_min".to_string(),
            json!(self.adaptive_verify_window_width_min),
        );
        timings.insert(
            "native_mtp_adaptive_verify_window_width_max".to_string(),
            json!(self.adaptive_verify_window_width_max),
        );
        timings.insert(
            "native_mtp_adaptive_verify_window_grows".to_string(),
            json!(self.adaptive_verify_window_grow_count),
        );
        timings.insert(
            "native_mtp_adaptive_verify_window_shrinks".to_string(),
            json!(self.adaptive_verify_window_shrink_count),
        );
        // Phase 0 survival curve (arrays indexed by depth-1, i.e. [0]=depth 1).
        // P(A>=k) per source = survived_ge[k-1] / eligible_ge[k-1].
        timings.insert(
            "native_mtp_survival_logical_proposals".to_string(),
            json!(self.logical_proposal_count),
        );
        timings.insert(
            "native_mtp_survival_native_eligible_ge".to_string(),
            json!(self.native_eligible_ge),
        );
        timings.insert(
            "native_mtp_survival_native_survived_ge".to_string(),
            json!(self.native_survived_ge),
        );
        timings.insert(
            "native_mtp_survival_ngram_eligible_ge".to_string(),
            json!(self.ngram_eligible_ge),
        );
        timings.insert(
            "native_mtp_survival_ngram_survived_ge".to_string(),
            json!(self.ngram_survived_ge),
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::frontend) struct NativeMtpDecodeTelemetry {
    options: NativeMtpDecodeOptions,
    counters: NativeMtpDecodeCounters,
}

impl NativeMtpDecodeTelemetry {
    pub(in crate::frontend) fn new(
        options: NativeMtpDecodeOptions,
        counters: NativeMtpDecodeCounters,
    ) -> Self {
        Self { options, counters }
    }

    pub(in crate::frontend) fn composite_proposal_totals(self) -> Option<(u64, u64)> {
        (self.counters.hybrid_proposal_token_count > 0).then_some((
            self.counters.hybrid_proposal_token_count as u64,
            self.counters.hybrid_accepted_token_count as u64,
        ))
    }

    pub(in crate::frontend) fn insert_response_timings(
        self,
        timings: &mut BTreeMap<String, Value>,
    ) {
        timings.insert(
            "native_mtp_ngram_hybrid_enabled".to_string(),
            json!(self.options.ngram_hybrid),
        );
        timings.insert(
            "native_mtp_ngram_size".to_string(),
            json!(self.options.ngram_size),
        );
        timings.insert(
            "native_mtp_ngram_initial_extension_tokens".to_string(),
            json!(self.options.ngram_initial_extension_tokens),
        );
        timings.insert(
            "native_mtp_ngram_max_proposal_tokens".to_string(),
            json!(self.options.ngram_max_proposal_tokens),
        );
        timings.insert(
            "native_mtp_ngram_tail_backoff_proposals".to_string(),
            json!(self.options.ngram_tail_backoff_proposals),
        );
        timings.insert(
            "native_mtp_verify_window_min_tokens".to_string(),
            json!(self.options.verify_window_min_tokens),
        );
        timings.insert(
            "native_mtp_verify_window_max_tokens".to_string(),
            json!(self.options.verify_window_max_tokens),
        );
        self.counters.insert_response_timings(timings);
    }
}

#[cfg(test)]
mod tests {
    use super::super::CompositeProposalProvider;
    use super::*;

    // Survival decomposition: native=3, ngram=4, all 7 accepted (+ free target
    // folded in by caller => accepted can exceed proposal len; min() caps it).
    #[test]
    fn survival_full_accept_counts_both_sources_to_their_depth() {
        let mut c = NativeMtpDecodeCounters::default();
        c.observe_survival(3, 4, 7);
        assert_eq!(c.logical_proposal_count, 1);
        // native eligible+survived at depths 1..=3
        assert_eq!(&c.native_eligible_ge[0..3], &[1, 1, 1]);
        assert_eq!(&c.native_survived_ge[0..3], &[1, 1, 1]);
        assert_eq!(c.native_eligible_ge[3], 0);
        // ngram eligible+survived at depths 1..=4
        assert_eq!(&c.ngram_eligible_ge[0..4], &[1, 1, 1, 1]);
        assert_eq!(&c.ngram_survived_ge[0..4], &[1, 1, 1, 1]);
    }

    // Native fails partway: ngram tail must NOT be counted eligible (it was
    // never reached), so native failure can't contaminate ngram quality.
    #[test]
    fn survival_native_partial_reject_excludes_ngram_tail() {
        let mut c = NativeMtpDecodeCounters::default();
        c.observe_survival(3, 4, 2); // only 2 of 3 native accepted
        assert_eq!(&c.native_eligible_ge[0..3], &[1, 1, 1]);
        assert_eq!(&c.native_survived_ge[0..3], &[1, 1, 0]); // survived depth 1,2 not 3
        // ngram never reached -> not eligible at any depth
        assert_eq!(c.ngram_eligible_ge.iter().sum::<usize>(), 0);
        assert_eq!(c.ngram_survived_ge.iter().sum::<usize>(), 0);
    }

    // Ngram partial: native fully survived, ngram accepted 2 of 4.
    #[test]
    fn survival_ngram_partial_accept() {
        let mut c = NativeMtpDecodeCounters::default();
        c.observe_survival(3, 4, 5); // 3 native + 2 ngram
        assert_eq!(&c.native_survived_ge[0..3], &[1, 1, 1]);
        assert_eq!(&c.ngram_eligible_ge[0..4], &[1, 1, 1, 1]);
        assert_eq!(&c.ngram_survived_ge[0..4], &[1, 1, 0, 0]);
    }

    // Pure-ngram proposal (native=0): everything is tail from depth 1.
    #[test]
    fn survival_pure_ngram_proposal() {
        let mut c = NativeMtpDecodeCounters::default();
        c.observe_survival(0, 5, 3); // 0 native, 3 of 5 ngram accepted
        assert_eq!(c.native_eligible_ge.iter().sum::<usize>(), 0);
        assert_eq!(&c.ngram_eligible_ge[0..5], &[1, 1, 1, 1, 1]);
        assert_eq!(&c.ngram_survived_ge[0..5], &[1, 1, 1, 0, 0]);
    }

    fn options() -> NativeMtpDecodeOptions {
        NativeMtpDecodeOptions {
            max_draft_tokens: 1,
            min_draft_tokens: 0,
            reject_cooldown_tokens: 0,
            suppress_cooldown_drafts: false,
            suppress_cooldown_draft_limit: 0,
            ngram_hybrid: true,
            ngram_size: 2,
            ngram_initial_extension_tokens: 2,
            ngram_max_proposal_tokens: 4,
            ngram_tail_backoff_proposals: 2,
            verify_window_min_tokens: 1,
            verify_window_max_tokens: 4,
        }
    }

    fn composite_proposal() -> NativeMtpHybridProposal {
        CompositeProposalProvider::from_options(options()).propose(
            &[9],
            &[1, 2, 3, 9, 1, 2, 3, 9, 1, 2, 3],
            4,
        )
    }

    #[test]
    fn decode_options_preserve_configured_initial_extension_width() {
        let config = SpeculativeDecodeConfig {
            extension: Some(crate::frontend::NgramExtensionConfig {
                initial_tokens: 3,
                max_tokens: 7,
                tail_backoff_proposals: 2,
            }),
            ngram: Some(crate::frontend::NgramProposalConfig {
                kind: crate::frontend::NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: 4,
                max_proposal_tokens: 7,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let options = NativeMtpDecodeOptions::from_config(&config);

        assert_eq!(options.ngram_initial_extension_tokens, 3);
        assert_eq!(options.ngram_max_proposal_tokens, 7);
    }

    #[test]
    fn counters_track_verify_window_verification_by_origin() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::InitialSerial, true);
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::SerialAfterGap, false);
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::VerifyNext, true);
        counters.observe_verify_next_draft(true, false);
        counters.observe_verify_next_draft(true, true);
        counters.observe_suppressed_cooldown_draft();
        counters.observe_hybrid_proposal(&composite_proposal(), 3);
        counters.observe_ngram_tail_rejection();
        counters.observe_adaptive_verify_window(2, 2, 3);

        let mut attrs = BTreeMap::new();
        counters.insert_summary_attrs(
            &mut attrs,
            NativeMtpDecodeOptions {
                max_draft_tokens: 3,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 6,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 2,
                ngram_hybrid: true,
                ngram_size: 8,
                ngram_initial_extension_tokens: 2,
                ngram_max_proposal_tokens: 4,
                ngram_tail_backoff_proposals: 6,
                verify_window_min_tokens: 1,
                verify_window_max_tokens: 4,
            },
        );

        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_window_verification_count"),
            Some(&json!(3))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.initial_serial_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.serial_after_gap_accepted_count"),
            Some(&json!(0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_accepted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_available_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.verify_next_draft_adopted_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.suppressed_cooldown_draft_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.reject_cooldown_tokens"),
            Some(&json!(6))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.ngram_initial_extension_tokens"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_accepted_tail_token_count"),
            Some(&json!(2))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_accepted_native_mtp_token_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_ngram_mtp_prefix_agreement_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_ngram_mtp_prefix_disagreement_count"),
            Some(&json!(0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_ngram_tail_rejection_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.hybrid_ngram_sidecar_backoff_count"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.adaptive_verify_window_grow_count"),
            Some(&json!(1))
        );
    }

    #[test]
    fn short_candidate_does_not_look_like_adaptive_window_growth() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_adaptive_verify_window(1, 4, 4);

        let mut attrs = BTreeMap::new();
        counters.insert_summary_attrs(&mut attrs, options());

        assert_eq!(
            attrs.get("llama_stage.native_mtp.adaptive_verify_window_width_sum"),
            Some(&json!(1))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.adaptive_verify_window_grow_count"),
            Some(&json!(0))
        );
        assert_eq!(
            attrs.get("llama_stage.native_mtp.adaptive_verify_window_shrink_count"),
            Some(&json!(0))
        );
    }

    #[test]
    fn response_timings_show_hybrid_widening_evidence() {
        let mut counters = NativeMtpDecodeCounters::default();
        counters.observe_verify_window_verification(NativeMtpDraftOrigin::InitialSerial, true);
        counters.observe_hybrid_proposal(&composite_proposal(), 3);
        counters.observe_adaptive_verify_window(2, 2, 3);
        let telemetry = NativeMtpDecodeTelemetry::new(
            NativeMtpDecodeOptions {
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
                ngram_hybrid: true,
                ngram_size: 8,
                ngram_initial_extension_tokens: 2,
                ngram_max_proposal_tokens: 4,
                ngram_tail_backoff_proposals: 6,
                verify_window_min_tokens: 1,
                verify_window_max_tokens: 4,
            },
            counters,
        );

        let mut timings = BTreeMap::new();
        telemetry.insert_response_timings(&mut timings);

        assert_eq!(
            timings.get("native_mtp_ngram_hybrid_enabled"),
            Some(&json!(true))
        );
        assert_eq!(
            timings.get("native_mtp_ngram_initial_extension_tokens"),
            Some(&json!(2))
        );
        assert_eq!(
            timings.get("native_mtp_hybrid_native_tokens"),
            Some(&json!(1))
        );
        assert_eq!(
            timings.get("native_mtp_hybrid_accepted_tail_tokens"),
            Some(&json!(2))
        );
        assert_eq!(
            timings.get("native_mtp_hybrid_accepted_native_tokens"),
            Some(&json!(1))
        );
        assert_eq!(
            timings.get("native_mtp_hybrid_ngram_mtp_prefix_agreements"),
            Some(&json!(1))
        );
        assert_eq!(
            timings.get("native_mtp_adaptive_verify_window_grows"),
            Some(&json!(1))
        );
    }

    #[test]
    fn adaptive_verify_window_starts_at_two_then_grows_and_shrinks() {
        let mut window = AdaptiveVerifyWindow::new(options());

        assert_eq!(window.current_tokens(), 2);
        assert_eq!(window.width(1), 1);
        assert!(window.observe(true));
        assert_eq!(window.current_tokens(), 3);
        assert!(window.observe(false));
        assert_eq!(window.current_tokens(), 2);
        assert!(window.observe(false));
        assert_eq!(window.current_tokens(), 1);
        assert!(!window.observe(false));
    }

    #[test]
    fn composite_proposal_totals_include_pure_ngram_candidates() {
        let mut counters = NativeMtpDecodeCounters::default();
        let proposal = CompositeProposalProvider::from_options(options()).propose(
            &[],
            &[0, 0, 2, 3, 9, 1, 7, 8, 2, 3],
            4,
        );
        counters.observe_hybrid_proposal(&proposal, 4);

        assert_eq!(
            NativeMtpDecodeTelemetry::new(options(), counters).composite_proposal_totals(),
            Some((4, 4))
        );
    }
}
