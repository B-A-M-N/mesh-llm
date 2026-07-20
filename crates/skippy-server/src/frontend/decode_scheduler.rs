use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use openai_frontend::{OpenAiError, OpenAiResult};

const PIPELINE_PROFILE_MIN_OBSERVATIONS: usize = 8;
const PIPELINE_PROFILE_MAX_OBSERVATIONS: usize = 32;
const PIPELINE_PROFIT_MARGIN: f64 = 1.15;
const PIPELINE_BOOTSTRAP_LATENCY_MULTIPLIER: f64 = 2.0;
// Depth one cannot reveal whether dependent work overlaps verification. Probe the
// smallest useful parallel window once, then let the latency policy choose depth.
const PIPELINE_BOOTSTRAP_PROBE_DEPTH: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VerifyWindowPipelineConfig {
    depth: usize,
}

impl VerifyWindowPipelineConfig {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
        }
    }

    pub(super) fn depth(self) -> usize {
        self.depth
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct VerifyWindowPipelineStats {
    depth: usize,
    bootstrap_probe_depth: usize,
    target_depth: usize,
    target_depth_max: usize,
    target_depth_updates: usize,
    direct_prediction_return: bool,
    opened_windows: usize,
    max_in_flight: usize,
    recovery_epochs: usize,
    stale_marked: usize,
    stale_discarded: usize,
    stale_drain_ms: f64,
    stale_stage0_compute_ms: f64,
    stale_forward_write_ms: f64,
    stale_downstream_wait_ms: f64,
    stale_verify_elapsed_ms: f64,
    policy_observed_windows: usize,
    policy_continuation_windows: usize,
    policy_permit_checks: usize,
    policy_permits: usize,
    policy_suppressed: usize,
    occupancy_ms_by_depth: Vec<f64>,
    occupancy_total_ms: f64,
    occupancy_parallel_ms: f64,
    occupancy_target_full_ms: f64,
    occupancy_full_ms: f64,
    occupancy_average_in_flight: f64,
}

impl VerifyWindowPipelineStats {
    pub(super) fn insert_response_timings(
        &self,
        timings: &mut BTreeMap<String, serde_json::Value>,
    ) {
        timings.insert(
            "verify_window_depth".to_string(),
            serde_json::json!(self.depth),
        );
        timings.insert(
            "verify_window_bootstrap_probe_depth".to_string(),
            serde_json::json!(self.bootstrap_probe_depth),
        );
        timings.insert(
            "verify_window_target_depth".to_string(),
            serde_json::json!(self.target_depth),
        );
        timings.insert(
            "verify_window_target_depth_max".to_string(),
            serde_json::json!(self.target_depth_max),
        );
        timings.insert(
            "verify_window_target_depth_updates".to_string(),
            serde_json::json!(self.target_depth_updates),
        );
        timings.insert(
            "verify_window_direct_prediction_return".to_string(),
            serde_json::json!(self.direct_prediction_return),
        );
        timings.insert(
            "verify_window_opened".to_string(),
            serde_json::json!(self.opened_windows),
        );
        timings.insert(
            "verify_window_max_in_flight".to_string(),
            serde_json::json!(self.max_in_flight),
        );
        timings.insert(
            "verify_window_recovery_epochs".to_string(),
            serde_json::json!(self.recovery_epochs),
        );
        timings.insert(
            "verify_window_stale_marked".to_string(),
            serde_json::json!(self.stale_marked),
        );
        timings.insert(
            "verify_window_stale_discarded".to_string(),
            serde_json::json!(self.stale_discarded),
        );
        timings.insert(
            "verify_window_stale_drain_ms".to_string(),
            serde_json::json!(self.stale_drain_ms),
        );
        timings.insert(
            "verify_window_stale_stage0_compute_ms".to_string(),
            serde_json::json!(self.stale_stage0_compute_ms),
        );
        timings.insert(
            "verify_window_stale_forward_write_ms".to_string(),
            serde_json::json!(self.stale_forward_write_ms),
        );
        timings.insert(
            "verify_window_stale_downstream_wait_ms".to_string(),
            serde_json::json!(self.stale_downstream_wait_ms),
        );
        timings.insert(
            "verify_window_stale_verify_elapsed_ms".to_string(),
            serde_json::json!(self.stale_verify_elapsed_ms),
        );
        timings.insert(
            "verify_window_policy_observed_windows".to_string(),
            serde_json::json!(self.policy_observed_windows),
        );
        timings.insert(
            "verify_window_policy_continuation_windows".to_string(),
            serde_json::json!(self.policy_continuation_windows),
        );
        timings.insert(
            "verify_window_policy_permit_checks".to_string(),
            serde_json::json!(self.policy_permit_checks),
        );
        timings.insert(
            "verify_window_policy_permits".to_string(),
            serde_json::json!(self.policy_permits),
        );
        timings.insert(
            "verify_window_policy_suppressed".to_string(),
            serde_json::json!(self.policy_suppressed),
        );
        timings.insert(
            "verify_window_occupancy_ms_by_depth".to_string(),
            serde_json::json!(self.occupancy_ms_by_depth),
        );
        timings.insert(
            "verify_window_occupancy_total_ms".to_string(),
            serde_json::json!(self.occupancy_total_ms),
        );
        timings.insert(
            "verify_window_occupancy_parallel_ms".to_string(),
            serde_json::json!(self.occupancy_parallel_ms),
        );
        timings.insert(
            "verify_window_occupancy_parallel_fraction".to_string(),
            serde_json::json!(fraction(
                self.occupancy_parallel_ms,
                self.occupancy_total_ms
            )),
        );
        timings.insert(
            "verify_window_occupancy_target_full_ms".to_string(),
            serde_json::json!(self.occupancy_target_full_ms),
        );
        timings.insert(
            "verify_window_occupancy_target_full_fraction".to_string(),
            serde_json::json!(fraction(
                self.occupancy_target_full_ms,
                self.occupancy_total_ms
            )),
        );
        timings.insert(
            "verify_window_occupancy_full_ms".to_string(),
            serde_json::json!(self.occupancy_full_ms),
        );
        timings.insert(
            "verify_window_occupancy_full_fraction".to_string(),
            serde_json::json!(fraction(self.occupancy_full_ms, self.occupancy_total_ms)),
        );
        timings.insert(
            "verify_window_occupancy_average_in_flight".to_string(),
            serde_json::json!(self.occupancy_average_in_flight),
        );
    }
}

fn fraction(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[derive(Debug, Default)]
struct VerifyWindowWidthProfile {
    observations: VecDeque<VerifyWindowProfileObservation>,
    continuation_windows: usize,
    stage0_compute_ms: f64,
    launch_ms: f64,
    verify_elapsed_ms: f64,
}

#[derive(Debug, Clone, Copy)]
struct VerifyWindowProfileObservation {
    continues: bool,
    stage0_compute_ms: f64,
    launch_ms: f64,
    verify_elapsed_ms: f64,
}

impl VerifyWindowWidthProfile {
    fn observe(
        &mut self,
        continues: bool,
        stage0_compute_ms: f64,
        _forward_write_ms: f64,
        verify_elapsed_ms: f64,
    ) {
        let stage0_compute_ms = stage0_compute_ms.max(0.0);
        let launch_ms = stage0_compute_ms.max(f64::EPSILON);
        let observation = VerifyWindowProfileObservation {
            continues,
            stage0_compute_ms,
            launch_ms,
            verify_elapsed_ms: verify_elapsed_ms.max(launch_ms),
        };
        self.observations.push_back(observation);
        self.continuation_windows = self
            .continuation_windows
            .saturating_add(usize::from(continues));
        self.stage0_compute_ms += observation.stage0_compute_ms;
        self.launch_ms += observation.launch_ms;
        self.verify_elapsed_ms += observation.verify_elapsed_ms;
        if self.observations.len() > PIPELINE_PROFILE_MAX_OBSERVATIONS {
            let expired = self
                .observations
                .pop_front()
                .expect("profile exceeded its non-empty bound");
            self.continuation_windows = self
                .continuation_windows
                .saturating_sub(usize::from(expired.continues));
            self.stage0_compute_ms -= expired.stage0_compute_ms;
            self.launch_ms -= expired.launch_ms;
            self.verify_elapsed_ms -= expired.verify_elapsed_ms;
        }
    }

    fn recommended_depth(&self, max_depth: usize) -> Option<usize> {
        if max_depth < 2 || self.observations.is_empty() {
            return None;
        }
        let observations = self.observations.len() as f64;
        let continuation_rate = self.continuation_windows as f64 / observations;
        let average_launch_ms = self.launch_ms / observations;
        let average_verify_ms = self.verify_elapsed_ms / observations;
        let hideable_latency_ms = (average_verify_ms - average_launch_ms).max(0.0);
        let expected_overlap_ms = continuation_rate * hideable_latency_ms;
        let expected_stale_ms = (1.0 - continuation_rate) * average_launch_ms;
        let enough_evidence = self.observations.len() >= PIPELINE_PROFILE_MIN_OBSERVATIONS;
        let latency_dominated_bootstrap =
            average_verify_ms >= average_launch_ms * PIPELINE_BOOTSTRAP_LATENCY_MULTIPLIER;
        if (!enough_evidence && !latency_dominated_bootstrap)
            || expected_overlap_ms <= expected_stale_ms * PIPELINE_PROFIT_MARGIN
        {
            return None;
        }
        let latency_cover_depth = (average_verify_ms / average_launch_ms).ceil() as usize;
        Some(latency_cover_depth.clamp(PIPELINE_BOOTSTRAP_PROBE_DEPTH, max_depth))
    }

    fn probe_or_recommended_depth(&self, max_depth: usize) -> Option<usize> {
        self.recommended_depth(max_depth).or_else(|| {
            (max_depth > 1 && self.observations.len() < PIPELINE_PROFILE_MIN_OBSERVATIONS)
                .then_some(PIPELINE_BOOTSTRAP_PROBE_DEPTH)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct VerifyWindow {
    pub(super) id: i32,
    pub(super) base_position: usize,
    pub(super) decode_step: usize,
}

#[derive(Debug)]
pub(super) struct VerifyWindowScheduler {
    config: VerifyWindowPipelineConfig,
    next_id: i32,
    in_flight: VecDeque<VerifyWindow>,
    stats: VerifyWindowPipelineStats,
    width_profiles: BTreeMap<usize, VerifyWindowWidthProfile>,
    target_depth: usize,
    occupancy_ms_by_depth: Vec<f64>,
    occupancy_target_full_ms: f64,
    occupancy_changed: Instant,
}

impl VerifyWindowScheduler {
    pub(super) fn new(config: VerifyWindowPipelineConfig) -> Self {
        Self {
            config,
            next_id: 1,
            in_flight: VecDeque::new(),
            stats: VerifyWindowPipelineStats {
                depth: config.depth(),
                bootstrap_probe_depth: if config.depth() >= PIPELINE_BOOTSTRAP_PROBE_DEPTH {
                    PIPELINE_BOOTSTRAP_PROBE_DEPTH
                } else {
                    0
                },
                target_depth: 1,
                target_depth_max: 1,
                ..VerifyWindowPipelineStats::default()
            },
            width_profiles: BTreeMap::new(),
            target_depth: 1,
            occupancy_ms_by_depth: vec![0.0; config.depth().saturating_add(1)],
            occupancy_target_full_ms: 0.0,
            occupancy_changed: Instant::now(),
        }
    }

    pub(super) fn has_capacity(&self) -> bool {
        self.in_flight.len() < self.target_depth
    }

    pub(super) fn depth(&self) -> usize {
        self.config.depth()
    }

    pub(super) fn mark_direct_prediction_return(&mut self) {
        self.stats.direct_prediction_return = true;
    }

    pub(super) fn observe_pipeline_profile(
        &mut self,
        width: usize,
        continues: bool,
        stage0_compute_ms: f64,
        forward_write_ms: f64,
        verify_elapsed_ms: f64,
    ) {
        if width == 0 {
            return;
        }
        self.stats.policy_observed_windows = self.stats.policy_observed_windows.saturating_add(1);
        self.stats.policy_continuation_windows = self
            .stats
            .policy_continuation_windows
            .saturating_add(usize::from(continues));
        self.width_profiles.entry(width).or_default().observe(
            continues,
            stage0_compute_ms,
            forward_write_ms,
            verify_elapsed_ms,
        );
    }

    pub(super) fn permit_pipeline_width(&mut self, width: usize) -> bool {
        self.stats.policy_permit_checks = self.stats.policy_permit_checks.saturating_add(1);
        let target_depth = if self.config.depth() > 1 {
            self.width_profiles
                .get(&width)
                .map_or(Some(PIPELINE_BOOTSTRAP_PROBE_DEPTH), |profile| {
                    profile.probe_or_recommended_depth(self.config.depth())
                })
        } else {
            None
        };
        let permitted = target_depth.is_some();
        if permitted {
            self.stats.policy_permits = self.stats.policy_permits.saturating_add(1);
            self.set_target_depth(target_depth.unwrap_or(1));
        } else {
            self.stats.policy_suppressed = self.stats.policy_suppressed.saturating_add(1);
            self.set_target_depth(1);
        }
        permitted
    }

    pub(super) fn insert_policy_telemetry_attrs(
        &self,
        attrs: &mut BTreeMap<String, serde_json::Value>,
    ) {
        attrs.insert(
            "llama_stage.verify_window.bootstrap_probe_depth".to_string(),
            serde_json::json!(self.stats.bootstrap_probe_depth),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_observed_windows".to_string(),
            serde_json::json!(self.stats.policy_observed_windows),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_continuation_windows".to_string(),
            serde_json::json!(self.stats.policy_continuation_windows),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_permit_checks".to_string(),
            serde_json::json!(self.stats.policy_permit_checks),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_permits".to_string(),
            serde_json::json!(self.stats.policy_permits),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_suppressed".to_string(),
            serde_json::json!(self.stats.policy_suppressed),
        );
        attrs.insert(
            "llama_stage.verify_window.pipeline_policy_profitable_widths".to_string(),
            serde_json::json!(
                self.width_profiles
                    .values()
                    .filter(|profile| { profile.recommended_depth(self.config.depth()).is_some() })
                    .count()
            ),
        );
    }

    pub(super) fn open(
        &mut self,
        base_position: usize,
        decode_step: usize,
    ) -> OpenAiResult<VerifyWindow> {
        if !self.has_capacity() {
            return Err(OpenAiError::backend(
                "verify window pipeline depth exceeded",
            ));
        }
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| OpenAiError::backend("verify window id overflow"))?;
        let window = VerifyWindow {
            id,
            base_position,
            decode_step,
        };
        self.record_occupancy();
        self.in_flight.push_back(window.clone());
        self.stats.opened_windows = self.stats.opened_windows.saturating_add(1);
        self.stats.max_in_flight = self.stats.max_in_flight.max(self.in_flight.len());
        Ok(window)
    }

    pub(super) fn complete_next(&mut self, reply_window_id: i32) -> OpenAiResult<VerifyWindow> {
        let Some(window) = self.in_flight.front() else {
            return Err(OpenAiError::backend(
                "verify window reply arrived with no in-flight window",
            ));
        };
        if window.id != reply_window_id {
            return Err(OpenAiError::backend(format!(
                "verify window reply out of order: got {reply_window_id}, expected {}",
                window.id
            )));
        }
        self.record_occupancy();
        Ok(self.in_flight.pop_front().expect("checked non-empty queue"))
    }

    #[cfg(test)]
    pub(super) fn discard_stale(&mut self) -> usize {
        let discarded = self.in_flight.len();
        self.record_occupancy();
        self.in_flight.clear();
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(discarded);
        discarded
    }

    pub(super) fn record_stale_discarded(&mut self, count: usize, drain_ms: f64) {
        self.stats.stale_discarded = self.stats.stale_discarded.saturating_add(count);
        self.stats.stale_drain_ms += drain_ms;
    }

    pub(super) fn mark_recovery_epoch(&mut self, stale_count: usize) {
        self.stats.recovery_epochs = self.stats.recovery_epochs.saturating_add(1);
        self.mark_stale(stale_count);
    }

    pub(super) fn mark_stale(&mut self, stale_count: usize) {
        self.stats.stale_marked = self.stats.stale_marked.saturating_add(stale_count);
    }

    pub(super) fn record_stale_execution(
        &mut self,
        drain_ms: f64,
        stage0_compute_ms: f64,
        forward_write_ms: f64,
        downstream_wait_ms: f64,
        verify_elapsed_ms: f64,
    ) {
        self.record_stale_discarded(1, drain_ms);
        self.stats.stale_stage0_compute_ms += stage0_compute_ms.max(0.0);
        self.stats.stale_forward_write_ms += forward_write_ms.max(0.0);
        self.stats.stale_downstream_wait_ms += downstream_wait_ms.max(0.0);
        self.stats.stale_verify_elapsed_ms += verify_elapsed_ms.max(0.0);
    }

    pub(super) fn in_flight_len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn stale_discard_count(&self) -> usize {
        self.stats.stale_discarded
    }

    pub(super) fn stats(&self) -> VerifyWindowPipelineStats {
        let mut stats = self.stats.clone();
        let mut occupancy_ms_by_depth = self.occupancy_ms_by_depth.clone();
        let current_interval_ms = self.occupancy_changed.elapsed().as_secs_f64() * 1_000.0;
        if let Some(bucket) = occupancy_ms_by_depth.get_mut(self.in_flight.len()) {
            *bucket += current_interval_ms;
        }
        let occupancy_total_ms = occupancy_ms_by_depth.iter().sum::<f64>();
        let occupancy_parallel_ms = occupancy_ms_by_depth.iter().skip(2).sum::<f64>();
        let occupancy_full_ms = occupancy_ms_by_depth
            .get(self.config.depth())
            .copied()
            .unwrap_or_default();
        let weighted_ms = occupancy_ms_by_depth
            .iter()
            .enumerate()
            .map(|(depth, elapsed_ms)| depth as f64 * elapsed_ms)
            .sum::<f64>();
        stats.target_depth = self.target_depth;
        stats.occupancy_ms_by_depth = occupancy_ms_by_depth;
        stats.occupancy_total_ms = occupancy_total_ms;
        stats.occupancy_parallel_ms = occupancy_parallel_ms;
        stats.occupancy_target_full_ms = self.occupancy_target_full_ms
            + if self.in_flight.len() >= self.target_depth {
                current_interval_ms
            } else {
                0.0
            };
        stats.occupancy_full_ms = occupancy_full_ms;
        stats.occupancy_average_in_flight = fraction(weighted_ms, occupancy_total_ms);
        stats
    }

    fn set_target_depth(&mut self, target_depth: usize) {
        let target_depth = target_depth.clamp(1, self.config.depth());
        if self.target_depth == target_depth {
            return;
        }
        self.record_occupancy();
        self.target_depth = target_depth;
        self.stats.target_depth = target_depth;
        self.stats.target_depth_max = self.stats.target_depth_max.max(target_depth);
        self.stats.target_depth_updates = self.stats.target_depth_updates.saturating_add(1);
    }

    fn record_occupancy(&mut self) {
        let elapsed_ms = self.occupancy_changed.elapsed().as_secs_f64() * 1_000.0;
        if let Some(bucket) = self.occupancy_ms_by_depth.get_mut(self.in_flight.len()) {
            *bucket += elapsed_ms;
        }
        if self.in_flight.len() >= self.target_depth {
            self.occupancy_target_full_ms += elapsed_ms;
        }
        self.occupancy_changed = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_depth_and_requires_fifo_reply_ids() {
        let config = VerifyWindowPipelineConfig { depth: 2 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.set_target_depth(2);
        let first = scheduler.open(10, 0).unwrap();
        let second = scheduler.open(11, 1).unwrap();

        assert!(scheduler.open(12, 2).is_err());
        assert!(scheduler.complete_next(second.id).is_err());
        assert_eq!(scheduler.in_flight_len(), 2);
        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
        assert_eq!(first.id, 1);
        assert_eq!(scheduler.stats().depth, 2);
        assert_eq!(scheduler.stats().opened_windows, 2);
        assert_eq!(scheduler.stats().max_in_flight, 2);
        assert!(!scheduler.stats().direct_prediction_return);
    }

    #[test]
    fn discards_stale_windows_after_divergence() {
        let config = VerifyWindowPipelineConfig { depth: 3 };
        let mut scheduler = VerifyWindowScheduler::new(config);
        scheduler.set_target_depth(3);
        scheduler.open(10, 0).unwrap();
        scheduler.open(11, 1).unwrap();
        scheduler.open(12, 2).unwrap();

        assert_eq!(scheduler.discard_stale(), 3);
        assert_eq!(scheduler.stale_discard_count(), 3);
        assert_eq!(scheduler.in_flight_len(), 0);
        assert_eq!(scheduler.stats().stale_discarded, 3);
    }

    #[test]
    fn stale_recovery_tracks_marked_and_completed_work_separately() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 3 });
        scheduler.set_target_depth(3);
        let first = scheduler.open(10, 0).unwrap();
        let second = scheduler.open(11, 1).unwrap();
        let third = scheduler.open(12, 2).unwrap();

        scheduler.complete_next(first.id).unwrap();
        scheduler.mark_recovery_epoch(2);
        scheduler.complete_next(second.id).unwrap();
        scheduler.record_stale_execution(3.0, 4.0, 5.0, 6.0, 15.0);
        scheduler.complete_next(third.id).unwrap();
        scheduler.record_stale_execution(7.0, 8.0, 9.0, 10.0, 27.0);

        let stats = scheduler.stats();
        assert_eq!(stats.recovery_epochs, 1);
        assert_eq!(stats.stale_marked, 2);
        assert_eq!(stats.stale_discarded, 2);
        assert_eq!(stats.stale_drain_ms, 10.0);
        assert_eq!(stats.stale_stage0_compute_ms, 12.0);
        assert_eq!(stats.stale_forward_write_ms, 14.0);
        assert_eq!(stats.stale_downstream_wait_ms, 16.0);
        assert_eq!(stats.stale_verify_elapsed_ms, 42.0);
    }

    #[test]
    fn pipeline_policy_probes_until_it_has_width_specific_evidence() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS - 1 {
            scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 30.0);
        }
        assert!(scheduler.permit_pipeline_width(2));

        scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 30.0);
        assert!(scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_bootstraps_an_unobserved_width_at_depth_two() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 8 });

        assert!(scheduler.permit_pipeline_width(4));
        assert_eq!(scheduler.target_depth, 2);
        assert_eq!(scheduler.stats().target_depth_max, 2);
    }

    #[test]
    fn pipeline_policy_bootstraps_latency_dominated_width_immediately() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 8 });
        scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 100.0);

        assert!(scheduler.permit_pipeline_width(2));
        assert_eq!(scheduler.target_depth, 5);
        assert_eq!(scheduler.stats().target_depth_max, 5);
    }

    #[test]
    fn pipeline_policy_suppresses_low_acceptance_local_work() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for index in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, index < 2, 31.0, 0.0, 55.0);
        }

        assert!(!scheduler.permit_pipeline_width(2));
        assert_eq!(scheduler.stats().policy_permit_checks, 1);
        assert_eq!(scheduler.stats().policy_permits, 0);
        assert_eq!(scheduler.stats().policy_suppressed, 1);
    }

    #[test]
    fn pipeline_policy_profiles_each_verify_width_independently() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for index in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(1, index < 7, 20.0, 0.0, 100.0);
            scheduler.observe_pipeline_profile(2, index < 2, 31.0, 0.0, 55.0);
        }

        assert!(scheduler.permit_pipeline_width(1));
        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_depth_one_never_admits_dependent_work() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 1 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 100.0);
        }

        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_adapts_when_recent_acceptance_changes() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MAX_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 100.0);
        }
        assert!(scheduler.permit_pipeline_width(2));

        for _ in 0..PIPELINE_PROFILE_MAX_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, false, 20.0, 0.0, 100.0);
        }
        assert!(!scheduler.permit_pipeline_width(2));
    }

    #[test]
    fn pipeline_policy_counters_are_exposed_in_response_timings() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        for _ in 0..PIPELINE_PROFILE_MIN_OBSERVATIONS {
            scheduler.observe_pipeline_profile(2, true, 20.0, 0.0, 100.0);
        }
        assert!(scheduler.permit_pipeline_width(2));
        let mut timings = BTreeMap::new();
        scheduler.stats().insert_response_timings(&mut timings);

        assert_eq!(
            timings["verify_window_policy_observed_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            timings["verify_window_policy_continuation_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(timings["verify_window_policy_permit_checks"], 1);
        assert_eq!(timings["verify_window_policy_permits"], 1);
        assert_eq!(timings["verify_window_policy_suppressed"], 0);
        assert_eq!(
            timings["verify_window_bootstrap_probe_depth"],
            PIPELINE_BOOTSTRAP_PROBE_DEPTH
        );

        let mut attrs = BTreeMap::new();
        scheduler.insert_policy_telemetry_attrs(&mut attrs);
        assert_eq!(
            attrs["llama_stage.verify_window.bootstrap_probe_depth"],
            PIPELINE_BOOTSTRAP_PROBE_DEPTH
        );
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_observed_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_continuation_windows"],
            serde_json::json!(PIPELINE_PROFILE_MIN_OBSERVATIONS)
        );
        assert_eq!(
            attrs["llama_stage.verify_window.pipeline_policy_profitable_widths"],
            1
        );
    }

    #[test]
    fn occupancy_timings_measure_parallel_and_full_depth_time() {
        let mut scheduler = VerifyWindowScheduler::new(VerifyWindowPipelineConfig { depth: 2 });
        scheduler.set_target_depth(2);
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(2);
        let first = scheduler.open(10, 0).unwrap();
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(3);
        let second = scheduler.open(11, 1).unwrap();
        scheduler.occupancy_changed = Instant::now() - std::time::Duration::from_millis(4);

        let stats = scheduler.stats();
        assert_eq!(stats.occupancy_ms_by_depth.len(), 3);
        assert!(stats.occupancy_ms_by_depth[0] >= 1.5);
        assert!(stats.occupancy_ms_by_depth[1] >= 2.5);
        assert!(stats.occupancy_ms_by_depth[2] >= 3.5);
        assert!(stats.occupancy_parallel_ms >= 3.5);
        assert!(stats.occupancy_target_full_ms >= 3.5);
        assert!(stats.occupancy_full_ms >= 3.5);
        assert!(stats.occupancy_average_in_flight > 1.0);

        assert_eq!(scheduler.complete_next(first.id).unwrap(), first);
        assert_eq!(scheduler.complete_next(second.id).unwrap(), second);
    }
}
