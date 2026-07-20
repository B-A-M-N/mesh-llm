use std::collections::VecDeque;

use super::{NativeMtpDraft, NativeMtpDraftOrigin, NativeMtpHybridProposal};

/// One candidate predicted by one exact-shape positional target evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct PipelinedCandidateWindow {
    proposal_token: i32,
    native_mtp_token_count: usize,
}

impl PipelinedCandidateWindow {
    pub(in crate::frontend) fn proposal_token(&self) -> i32 {
        self.proposal_token
    }

    pub(in crate::frontend) fn native_mtp_token_count(&self) -> usize {
        self.native_mtp_token_count
    }
}

/// Owns a composite candidate while positional windows consume it.
///
/// Every window evaluates exactly one input token and predicts exactly one
/// candidate token, matching native target decode's batch shape. Pipeline depth
/// comes only from consecutive positions in flight, never a wider target batch.
#[derive(Debug)]
pub(in crate::frontend) struct CompositeProposalPipeline {
    proposal: NativeMtpHybridProposal,
    origin: Option<NativeMtpDraftOrigin>,
    candidates: VecDeque<i32>,
    dispatched_native_mtp_token_count: usize,
    accepted_tokens: usize,
    next_draft: Option<NativeMtpDraft>,
}

impl CompositeProposalPipeline {
    pub(in crate::frontend) fn new(
        proposal: NativeMtpHybridProposal,
        origin: Option<NativeMtpDraftOrigin>,
    ) -> Self {
        Self {
            candidates: proposal.tokens().iter().copied().collect(),
            proposal,
            origin,
            dispatched_native_mtp_token_count: 0,
            accepted_tokens: 0,
            next_draft: None,
        }
    }

    pub(in crate::frontend) fn next_window(&mut self) -> Option<PipelinedCandidateWindow> {
        let proposal_token = self.candidates.pop_front()?;
        let native_mtp_token_count = self
            .proposal
            .native_mtp_token_count()
            .saturating_sub(self.dispatched_native_mtp_token_count)
            .min(1);
        self.dispatched_native_mtp_token_count += native_mtp_token_count;
        Some(PipelinedCandidateWindow {
            proposal_token,
            native_mtp_token_count,
        })
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

    /// The uncommitted optimistic suffix, including tokens already dispatched
    /// but not yet committed. The N-gram cache may read this suffix while its
    /// index remains restricted to committed target history.
    pub(in crate::frontend) fn optimistic_suffix(&self) -> &[i32] {
        &self.proposal.tokens()[self.accepted_tokens.min(self.proposal.tokens().len())..]
    }

    pub(in crate::frontend) fn append_ngram_candidates(&mut self, tokens: &[i32]) -> usize {
        self.proposal.append_ngram_tokens(tokens);
        self.candidates.extend(tokens.iter().copied());
        tokens.len()
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
    fn plans_one_exact_target_position_per_window() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_window().unwrap();
        assert_eq!(first.proposal_token(), 9);
        assert_eq!(first.native_mtp_token_count(), 1);

        let second = pipeline.next_window().unwrap();
        assert_eq!(second.proposal_token(), 1);
        assert_eq!(second.native_mtp_token_count(), 0);
    }

    #[test]
    fn supports_a_pure_ngram_candidate() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None);

        let window = pipeline.next_window().unwrap();
        assert_eq!(window.proposal_token(), 1);
        assert_eq!(window.native_mtp_token_count(), 0);
        assert!(pipeline.has_remaining_candidates());
    }

    #[test]
    fn pure_ngram_pipeline_discards_verify_next_native_mtp_drafts() {
        let mut pipeline = CompositeProposalPipeline::new(proposal(vec![1, 2, 3], 0), None);

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
    fn records_the_matching_prefix_of_a_rejected_window() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let _ = pipeline.next_window().unwrap();
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
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_window().unwrap();
        assert_eq!(first.proposal_token(), 9);
        pipeline.observe_accepted(1);

        let second = pipeline.next_window().unwrap();
        assert_eq!(second.proposal_token(), 1);
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
    fn appends_an_optimistic_ngram_suffix_without_committing_it() {
        let mut pipeline = CompositeProposalPipeline::new(
            proposal(vec![9, 1, 2, 3], 1),
            Some(NativeMtpDraftOrigin::InitialSerial),
        );

        let first = pipeline.next_window().unwrap();
        assert_eq!(first.proposal_token(), 9);
        assert_eq!(pipeline.optimistic_suffix(), &[9, 1, 2, 3]);

        pipeline.observe_accepted(1);
        assert_eq!(pipeline.optimistic_suffix(), &[1, 2, 3]);
        assert_eq!(pipeline.append_ngram_candidates(&[4, 5]), 2);
        assert_eq!(pipeline.optimistic_suffix(), &[1, 2, 3, 4, 5]);
        assert_eq!(pipeline.proposal().tokens(), &[9, 1, 2, 3, 4, 5]);
        assert_eq!(pipeline.proposal().ngram_token_count(), 5);

        let second = pipeline.next_window().unwrap();
        assert_eq!(second.proposal_token(), 1);
    }
}
