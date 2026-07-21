use std::collections::VecDeque;

use super::{NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpHybridProposal};

/// The candidate portion of one dispatched asynchronous verify window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct PipelinedCandidateWindow {
    proposal_tokens: Vec<i32>,
    expected_free_target: Option<i32>,
    native_mtp_token_count: usize,
}

impl PipelinedCandidateWindow {
    pub(in crate::frontend) fn proposal_tokens(&self) -> &[i32] {
        &self.proposal_tokens
    }

    pub(in crate::frontend) fn expected_free_target(&self) -> Option<i32> {
        self.expected_free_target
    }

    pub(in crate::frontend) fn native_mtp_token_count(&self) -> usize {
        self.native_mtp_token_count
    }
}

/// Owns a deeper composite candidate while asynchronous windows consume it.
/// Each planned window reserves the target's free-advance candidate as the
/// next window's optimistic current token, preventing duplicate KV positions.
#[derive(Debug)]
pub(in crate::frontend) struct CompositeProposalPipeline {
    proposal: NativeMtpHybridProposal,
    origin: Option<NativeMtpDraftOrigin>,
    candidates: VecDeque<i32>,
    parallel_verify_width: usize,
    dispatched_native_mtp_token_count: usize,
    accepted_tokens: usize,
    next_draft: Option<NativeMtpDraft>,
    /// Sealed-tail invariant (expert-reviewed): once a dispatched window had no
    /// reserved free-target candidate (`expected_free_target == None`), the
    /// pipeline's optimistic frontier is no longer represented by a buffered
    /// bridge token, so appending more candidates behind it would splice onto a
    /// stale `pipelined_current`. Once sealed, refill is refused and the pipeline
    /// must retire/reseed. This is the flag the ledger-free continuous refill
    /// relies on for correctness.
    sealed: bool,
}

impl CompositeProposalPipeline {
    pub(in crate::frontend) fn new(
        proposal: NativeMtpHybridProposal,
        origin: Option<NativeMtpDraftOrigin>,
        parallel_verify_width: usize,
    ) -> Self {
        Self {
            candidates: proposal.tokens().iter().copied().collect(),
            proposal,
            origin,
            parallel_verify_width: parallel_verify_width.max(1),
            dispatched_native_mtp_token_count: 0,
            accepted_tokens: 0,
            next_draft: None,
            sealed: false,
        }
    }

    pub(in crate::frontend) fn next_window(
        &mut self,
        verify_width: usize,
    ) -> Option<PipelinedCandidateWindow> {
        let verify_width = verify_width
            .min(self.parallel_verify_width)
            .min(self.candidates.len());
        if verify_width == 0 {
            return None;
        }
        let native_mtp_token_count = self
            .proposal
            .native_mtp_token_count()
            .saturating_sub(self.dispatched_native_mtp_token_count)
            .min(verify_width);
        let proposal_tokens = self.candidates.drain(..verify_width).collect();
        self.dispatched_native_mtp_token_count += native_mtp_token_count;
        let expected_free_target = self.candidates.pop_front();
        if expected_free_target.is_none() {
            // No reserved bridge token: the optimistic frontier is no longer
            // represented by a buffered candidate. Seal the tail so refill can
            // never splice behind this window (sealed-tail invariant).
            self.sealed = true;
        }
        Some(PipelinedCandidateWindow {
            proposal_tokens,
            expected_free_target,
            native_mtp_token_count,
        })
    }

    /// The uncommitted optimistic suffix (proposal tokens past what the target
    /// has committed). Safe to use ONLY as read-only lookup context for the
    /// n-gram cache; it must never be indexed into the cache as committed
    /// history.
    pub(in crate::frontend) fn optimistic_suffix(&self) -> &[i32] {
        let tokens = self.proposal.tokens();
        &tokens[self.accepted_tokens.min(tokens.len())..]
    }

    /// Whether the pipeline may still accept refilled candidates. False once the
    /// tail is sealed (a window was dispatched without a reserved free target).
    pub(in crate::frontend) fn can_refill(&self) -> bool {
        !self.sealed
    }

    /// Append n-gram tail candidates for continuous refill. Extends both the
    /// proposal (for accounting/telemetry) and the dispatchable candidate deque.
    /// Refuses to append when the tail is sealed (returns 0), preserving the
    /// sealed-tail invariant. Returns the number of candidates appended.
    pub(in crate::frontend) fn append_ngram_candidates(&mut self, tokens: &[i32]) -> usize {
        if self.sealed || tokens.is_empty() {
            return 0;
        }
        self.proposal.append_ngram_tokens(tokens);
        self.candidates.extend(tokens.iter().copied());
        tokens.len()
    }

    pub(in crate::frontend) fn proposal(&self) -> &NativeMtpHybridProposal {
        &self.proposal
    }

    pub(in crate::frontend) fn origin(&self) -> Option<NativeMtpDraftOrigin> {
        self.origin
    }

    pub(in crate::frontend) fn has_remaining_candidates(&self) -> bool {
        !self.candidates.is_empty()
    }

    pub(in crate::frontend) fn candidate_len(&self) -> usize {
        self.candidates.len()
    }

    pub(in crate::frontend) fn observe_accepted(&mut self, count: usize) {
        self.accepted_tokens += count;
    }

    pub(in crate::frontend) fn accepted_tokens(&self) -> usize {
        self.accepted_tokens
    }

    pub(in crate::frontend) fn set_next_draft(
        &mut self,
        native_mtp_enabled: bool,
        draft: Option<NativeMtpDraft>,
    ) {
        self.next_draft = native_mtp_enabled.then_some(draft).flatten();
    }

    pub(in crate::frontend) fn next_draft(&self) -> Option<&NativeMtpDraft> {
        self.next_draft.as_ref()
    }

    pub(in crate::frontend) fn take_next_draft(&mut self) -> Option<NativeMtpDraft> {
        self.next_draft.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(tokens: Vec<i32>, native_mtp_tokens: usize) -> NativeMtpHybridProposal {
        let ngram_span_available = native_mtp_tokens < tokens.len();
        NativeMtpHybridProposal::from_parts(tokens, native_mtp_tokens, ngram_span_available)
    }

    #[test]
    fn reserves_free_target_as_the_next_optimistic_current_token() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3, 4], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );

        let first = pipeline.next_window(2).unwrap();
        assert_eq!(first.proposal_tokens(), &[9, 1]);
        assert_eq!(first.expected_free_target(), Some(2));
        assert_eq!(first.native_mtp_token_count(), 1);

        let second = pipeline.next_window(2).unwrap();
        assert_eq!(second.proposal_tokens(), &[3, 4]);
        assert_eq!(second.expected_free_target(), None);
        assert_eq!(second.native_mtp_token_count(), 0);
    }

    #[test]
    fn supports_a_pure_ngram_candidate() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None, 2);

        let window = pipeline.next_window(2).unwrap();
        assert_eq!(window.proposal_tokens(), &[1, 2]);
        assert_eq!(window.expected_free_target(), Some(3));
        assert_eq!(window.native_mtp_token_count(), 0);
        assert!(!pipeline.has_remaining_candidates());
    }

    #[test]
    fn pure_ngram_pipeline_discards_verify_next_native_mtp_drafts() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None, 2);

        pipeline.set_next_draft(
            false,
            Some(NativeMtpDraft {
                tokens: vec![4],
                proposal_compute_us: 12,
            }),
        );

        assert!(pipeline.next_draft().is_none());
    }

    #[test]
    fn caps_each_dispatched_window_to_the_parallel_width() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None, 1);

        let window = pipeline.next_window(4).unwrap();

        assert_eq!(window.proposal_tokens(), &[1]);
        assert_eq!(window.expected_free_target(), Some(2));
    }

    #[test]
    fn records_the_matching_prefix_of_a_rejected_window() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );

        let _ = pipeline.next_window(2).unwrap();
        pipeline.observe_accepted(1);

        assert_eq!(pipeline.accepted_tokens(), 1);
        assert!(
            pipeline
                .proposal()
                .ngram_tail_rejected(pipeline.accepted_tokens())
        );
    }

    #[test]
    fn later_ngram_rejection_does_not_reject_an_accepted_native_prefix() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3, 4], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );

        let first = pipeline.next_window(2).unwrap();
        assert_eq!(first.proposal_tokens(), &[9, 1]);
        assert_eq!(first.expected_free_target(), Some(2));
        pipeline.observe_accepted(3);

        let second = pipeline.next_window(2).unwrap();
        assert_eq!(second.proposal_tokens(), &[3, 4]);
        pipeline.observe_accepted(0);

        assert!(
            !pipeline
                .proposal()
                .native_mtp_prefix_rejected(pipeline.accepted_tokens())
        );
        assert!(
            pipeline
                .proposal()
                .ngram_tail_rejected(pipeline.accepted_tokens())
        );
    }

    #[test]
    fn optimistic_suffix_tracks_accepted_tokens() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );
        assert_eq!(pipeline.optimistic_suffix(), &[9, 1, 2, 3]);
        pipeline.observe_accepted(2);
        assert_eq!(pipeline.optimistic_suffix(), &[2, 3]);
        // accepted beyond len saturates, no panic
        pipeline.observe_accepted(10);
        assert_eq!(pipeline.optimistic_suffix(), &[] as &[i32]);
    }

    #[test]
    fn append_ngram_candidates_extends_proposal_and_deque() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );
        // consume the first window (width 2), leaving candidates [2,3] then
        // reserving 2 as free target -> candidates [3]
        let first = pipeline.next_window(2).unwrap();
        assert_eq!(first.proposal_tokens(), &[9, 1]);
        assert_eq!(first.expected_free_target(), Some(2));
        assert!(pipeline.can_refill());
        assert_eq!(pipeline.append_ngram_candidates(&[4, 5]), 2);
        assert_eq!(pipeline.proposal().tokens(), &[9, 1, 2, 3, 4, 5]);
        assert_eq!(pipeline.proposal().ngram_token_count(), 5);
        // next window can now draw the refilled candidates
        let second = pipeline.next_window(2).unwrap();
        assert_eq!(second.proposal_tokens(), &[3, 4]);
        assert_eq!(second.expected_free_target(), Some(5));
    }

    #[test]
    fn sealed_tail_refuses_refill_after_window_without_free_target() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );
        // width-2 window drains both candidates, leaving none to reserve as the
        // free target -> tail seals.
        let first = pipeline.next_window(2).unwrap();
        assert_eq!(first.proposal_tokens(), &[9, 1]);
        assert_eq!(first.expected_free_target(), None);
        assert!(!pipeline.can_refill());
        // refill is refused once sealed (sealed-tail invariant)
        assert_eq!(pipeline.append_ngram_candidates(&[4, 5]), 0);
        assert_eq!(pipeline.proposal().tokens(), &[9, 1]);
    }

    #[test]
    fn append_empty_is_noop() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
            2,
        );
        assert_eq!(pipeline.append_ngram_candidates(&[]), 0);
        assert_eq!(pipeline.proposal().tokens(), &[9, 1, 2]);
    }
}
