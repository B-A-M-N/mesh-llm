use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SpeculativeCreditDenial {
    Disabled,
    PipelineOccupied,
    FairShare,
    GlobalLimit,
    FairQueue,
    LockPoisoned,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SpeculativeCreditSnapshot {
    pub(super) global_limit: usize,
    pub(super) effective_limit: usize,
    pub(super) global_in_use: usize,
    pub(super) global_max_in_use: usize,
    pub(super) active_requests: usize,
    pub(super) request_held: usize,
    pub(super) request_fair_share: usize,
    pub(super) queued_requests: usize,
}

pub(super) struct SpeculativeCreditAttempt {
    pub(super) credit: Option<SpeculativeCredit>,
    pub(super) denial: Option<SpeculativeCreditDenial>,
    pub(super) snapshot: SpeculativeCreditSnapshot,
}

#[derive(Clone, Debug)]
pub(super) struct SpeculativeCreditPool {
    inner: Arc<Mutex<SpeculativeCreditState>>,
}

#[derive(Debug)]
struct SpeculativeCreditState {
    limit: usize,
    in_use: usize,
    max_in_use: usize,
    requests: BTreeMap<u64, RequestCreditState>,
    waiters: VecDeque<u64>,
}

#[derive(Debug, Default)]
struct RequestCreditState {
    active: bool,
    held: usize,
}

impl SpeculativeCreditPool {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SpeculativeCreditState {
                limit,
                in_use: 0,
                max_in_use: 0,
                requests: BTreeMap::new(),
                waiters: VecDeque::new(),
            })),
        }
    }

    pub(super) fn register(&self, request_id: u64) -> SpeculativeCreditRequest {
        let registered = self.inner.lock().is_ok_and(|mut state| {
            let request = state.requests.entry(request_id).or_default();
            if request.active {
                return false;
            }
            request.active = true;
            true
        });
        SpeculativeCreditRequest {
            pool: Arc::downgrade(&self.inner),
            request_id,
            registered,
        }
    }

    pub(super) fn capacity_snapshot(
        &self,
        active_generation_requests: usize,
    ) -> SpeculativeCreditSnapshot {
        let Ok(state) = self.inner.lock() else {
            return SpeculativeCreditSnapshot::default();
        };
        let active_generation_requests = active_generation_requests
            .max(state.active_requests())
            .max(1);
        state.snapshot(0, active_generation_requests)
    }
}

pub(super) struct SpeculativeCreditRequest {
    pool: Weak<Mutex<SpeculativeCreditState>>,
    request_id: u64,
    registered: bool,
}

impl SpeculativeCreditRequest {
    pub(super) fn try_acquire(
        &self,
        active_generation_requests: usize,
    ) -> SpeculativeCreditAttempt {
        let Some(pool) = self.pool.upgrade() else {
            return denied_attempt(SpeculativeCreditDenial::Disabled);
        };
        let Ok(mut state) = pool.lock() else {
            return denied_attempt(SpeculativeCreditDenial::LockPoisoned);
        };
        if !self.registered || state.limit == 0 {
            return state.denied(
                self.request_id,
                active_generation_requests.max(1),
                SpeculativeCreditDenial::Disabled,
            );
        }
        let active_generation_requests = active_generation_requests
            .max(state.active_requests())
            .max(1);
        let effective_limit = state.effective_limit(active_generation_requests);
        state.prune_waiters(active_generation_requests);
        if effective_limit == 0 {
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::PipelineOccupied,
            );
        }
        let fair_share = state.fair_share(active_generation_requests);
        let held = state
            .requests
            .get(&self.request_id)
            .map_or(0, |request| request.held);
        if held >= fair_share {
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::FairShare,
            );
        }
        if state.in_use >= effective_limit {
            state.enqueue(self.request_id);
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::GlobalLimit,
            );
        }
        if state
            .waiters
            .front()
            .is_some_and(|waiting| *waiting != self.request_id)
        {
            state.enqueue(self.request_id);
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::FairQueue,
            );
        }
        if state.waiters.front() == Some(&self.request_id) {
            state.waiters.pop_front();
        }
        let Some(request) = state.requests.get_mut(&self.request_id) else {
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::Disabled,
            );
        };
        if !request.active {
            return state.denied(
                self.request_id,
                active_generation_requests,
                SpeculativeCreditDenial::Disabled,
            );
        }
        request.held = request.held.saturating_add(1);
        state.in_use = state.in_use.saturating_add(1);
        state.max_in_use = state.max_in_use.max(state.in_use);
        let snapshot = state.snapshot(self.request_id, active_generation_requests);
        drop(state);
        SpeculativeCreditAttempt {
            credit: Some(SpeculativeCredit {
                pool: Arc::downgrade(&pool),
                request_id: self.request_id,
            }),
            denial: None,
            snapshot,
        }
    }
}

impl Drop for SpeculativeCreditRequest {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        let Ok(mut state) = pool.lock() else {
            return;
        };
        state.waiters.retain(|waiting| *waiting != self.request_id);
        let remove = state
            .requests
            .get_mut(&self.request_id)
            .is_some_and(|request| {
                request.active = false;
                request.held == 0
            });
        if remove {
            state.requests.remove(&self.request_id);
        }
    }
}

pub(super) struct SpeculativeCredit {
    pool: Weak<Mutex<SpeculativeCreditState>>,
    request_id: u64,
}

impl Drop for SpeculativeCredit {
    fn drop(&mut self) {
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        let Ok(mut state) = pool.lock() else {
            return;
        };
        let remove = state
            .requests
            .get_mut(&self.request_id)
            .is_some_and(|request| {
                request.held = request.held.saturating_sub(1);
                !request.active && request.held == 0
            });
        state.in_use = state.in_use.saturating_sub(1);
        if remove {
            state.requests.remove(&self.request_id);
        }
        state.prune_inactive_waiters();
    }
}

impl SpeculativeCreditState {
    fn active_requests(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.active)
            .count()
    }

    fn effective_limit(&self, active_generation_requests: usize) -> usize {
        // Every live generation already owns one unmetered progress window.
        // Subtract those base windows from configured pipeline capacity so
        // spare depth is never replicated for every request.
        self.limit
            .saturating_add(1)
            .saturating_sub(active_generation_requests)
    }

    fn fair_share(&self, active_generation_requests: usize) -> usize {
        self.effective_limit(active_generation_requests)
            .div_ceil(active_generation_requests.max(1))
    }

    fn enqueue(&mut self, request_id: u64) {
        if !self.waiters.contains(&request_id) {
            self.waiters.push_back(request_id);
        }
    }

    fn prune_waiters(&mut self, active_generation_requests: usize) {
        let fair_share = self.fair_share(active_generation_requests);
        self.waiters.retain(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|request| request.active && request.held < fair_share)
        });
    }

    fn prune_inactive_waiters(&mut self) {
        self.waiters.retain(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|request| request.active)
        });
    }

    fn snapshot(
        &self,
        request_id: u64,
        active_generation_requests: usize,
    ) -> SpeculativeCreditSnapshot {
        SpeculativeCreditSnapshot {
            global_limit: self.limit,
            effective_limit: self.effective_limit(active_generation_requests),
            global_in_use: self.in_use,
            global_max_in_use: self.max_in_use,
            active_requests: active_generation_requests,
            request_held: self
                .requests
                .get(&request_id)
                .map_or(0, |request| request.held),
            request_fair_share: self.fair_share(active_generation_requests),
            queued_requests: self.waiters.len(),
        }
    }

    fn denied(
        &self,
        request_id: u64,
        active_generation_requests: usize,
        denial: SpeculativeCreditDenial,
    ) -> SpeculativeCreditAttempt {
        SpeculativeCreditAttempt {
            credit: None,
            denial: Some(denial),
            snapshot: self.snapshot(request_id, active_generation_requests),
        }
    }
}

fn denied_attempt(denial: SpeculativeCreditDenial) -> SpeculativeCreditAttempt {
    SpeculativeCreditAttempt {
        credit: None,
        denial: Some(denial),
        snapshot: SpeculativeCreditSnapshot::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fair_share_prevents_one_request_from_monopolizing_credits() {
        let pool = SpeculativeCreditPool::new(3);
        let first = pool.register(1);
        let second = pool.register(2);

        let first_credit = first.try_acquire(2);
        assert!(first_credit.credit.is_some());
        let denied = first.try_acquire(2);
        assert_eq!(denied.denial, Some(SpeculativeCreditDenial::FairShare));
        let second_credit = second.try_acquire(2);
        assert!(second_credit.credit.is_some());
        assert_eq!(second_credit.snapshot.global_max_in_use, 2);
    }

    #[test]
    fn queued_request_gets_next_released_credit() {
        let pool = SpeculativeCreditPool::new(2);
        let first = pool.register(1);
        let second = pool.register(2);

        let first_credit = first.try_acquire(2).credit.unwrap();
        assert_eq!(
            second.try_acquire(2).denial,
            Some(SpeculativeCreditDenial::GlobalLimit)
        );
        drop(first_credit);
        assert_eq!(
            first.try_acquire(2).denial,
            Some(SpeculativeCreditDenial::FairQueue)
        );
        assert!(second.try_acquire(2).credit.is_some());
    }

    #[test]
    fn request_drop_with_outstanding_credit_releases_without_leaking() {
        let pool = SpeculativeCreditPool::new(1);
        let request = pool.register(1);
        let credit = request.try_acquire(1).credit.unwrap();
        drop(request);
        drop(credit);

        let next = pool.register(2);
        let attempt = next.try_acquire(1);
        assert!(attempt.credit.is_some());
        assert_eq!(attempt.snapshot.global_in_use, 1);
    }

    #[test]
    fn duplicate_registration_drop_does_not_deactivate_owner() {
        let pool = SpeculativeCreditPool::new(2);
        let owner = pool.register(1);
        let duplicate = pool.register(1);

        assert!(!duplicate.registered);
        drop(duplicate);

        let attempt = owner.try_acquire(1);
        assert!(attempt.credit.is_some());
        assert_eq!(attempt.snapshot.active_requests, 1);
    }

    #[test]
    fn disabled_pool_fails_closed() {
        let pool = SpeculativeCreditPool::new(0);
        let request = pool.register(1);
        let attempt = request.try_acquire(1);

        assert!(attempt.credit.is_none());
        assert_eq!(attempt.denial, Some(SpeculativeCreditDenial::Disabled));
    }

    #[test]
    fn concurrent_requests_never_exceed_the_global_limit() {
        let pool = SpeculativeCreditPool::new(10);
        let requests = (0..8)
            .map(|request_id| pool.register(request_id))
            .collect::<Vec<_>>();
        let attempts = requests
            .iter()
            .map(|request| request.try_acquire(8))
            .collect::<Vec<_>>();

        assert_eq!(
            attempts
                .iter()
                .filter(|attempt| attempt.credit.is_some())
                .count(),
            3
        );
        assert!(
            attempts
                .iter()
                .all(|attempt| attempt.snapshot.global_max_in_use <= 3)
        );
    }

    #[test]
    fn base_progress_windows_can_fully_occupy_pipeline_capacity() {
        let pool = SpeculativeCreditPool::new(7);
        let requests = (0..8)
            .map(|request_id| pool.register(request_id))
            .collect::<Vec<_>>();

        let attempt = requests[0].try_acquire(8);

        assert_eq!(attempt.snapshot.effective_limit, 0);
        assert_eq!(
            attempt.denial,
            Some(SpeculativeCreditDenial::PipelineOccupied)
        );
    }
}
