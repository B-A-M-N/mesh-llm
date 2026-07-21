//! Pre-warmed pool of direct-prediction-return sockets for stage 0.
//!
//! Opening a return sink over a WAN mesh split is unreliable when done cold on
//! the generation hot path: each request opens a fresh TCP connection through
//! the local bridge alias, which spins up a fresh QUIC bi-stream and blocks on a
//! ready handshake. Under normal WAN jitter that intermittently fails
//! (`failed to fill whole buffer`, EAGAIN), which drops the request to the
//! slower serial upstream-reply path and prevents speculative pipelining.
//!
//! The forward path never has this problem because `PersistentStageLanePool`
//! pre-warms and reuses its lanes. This pool applies the same idea to the return
//! channel: a single long-lived maintenance worker keeps `capacity` return
//! sockets pre-warmed (connect + ready handshake done, NOT bound to a request).
//! A request binds one cheaply by writing `PredictionReturnOpen`.
//!
//! Design notes (from expert review):
//! - One maintenance worker owns all prepares: serialized refill (no capacity
//!   overshoot), no per-request thread stampede, and proactive rotation so the
//!   pool stays warm even while idle (a parked socket cannot be liveness-probed
//!   reliably, so it is rotated out on age before it can silently rot).
//! - `checkout` is FIFO (`pop_front`) and purges expired sockets, so a burst
//!   after idle does not hand out stale sockets.
//! - This pool removes the cold-handshake intermittency (the "prepared" state).
//!   It does NOT by itself prove the tail selected the socket end-to-end; that
//!   is the selection-ACK layer on the receiver. Direct-only pipelining must
//!   gate on that confirmation, not merely on a successful bind here.
//! - Scope: targets stage 0's immediate downstream, which is the real tail only
//!   in a two-stage topology. Longer chains fall back to the cold open.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;
use std::time::Duration;
use std::time::Instant;

use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::WireActivationDType;

use crate::binary_transport::direct_return;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;

/// A pre-warmed return socket: connected + ready-handshaked, not yet bound to a
/// request. `prepared_at` drives age-based rotation.
struct PreparedReturnSink {
    id: u64,
    stream: std::net::TcpStream,
    prepared_at: Instant,
}

/// Rotate a parked socket out this long after preparation. Chosen below
/// [`PREPARED_SINK_MAX_AGE`] so the maintenance worker replaces a socket
/// *before* it would be rejected at checkout, keeping the pool continuously warm
/// even under idle/sporadic traffic.
const PREPARED_SINK_REFRESH_AGE: Duration = Duration::from_secs(15);

/// Hard cutoff: a socket older than this is never handed out (safety net if the
/// maintenance worker is briefly starved). A parked socket's TCP half is the
/// loopback bridge, so a dead remote QUIC leg can leave it looking healthy
/// indefinitely; age bounds how long a silently-dead socket can be used.
const PREPARED_SINK_MAX_AGE: Duration = Duration::from_secs(30);

/// Maintenance worker wake interval. Bounds how quickly the pool refills after a
/// checkout and how promptly it rotates aging sockets. Checkout also signals the
/// worker so refill is prompt under load rather than waiting a full interval.
const MAINTENANCE_INTERVAL: Duration = Duration::from_secs(2);

/// Whether a prepared socket parked since `prepared_at` is still fresh enough to
/// hand out at checkout. Pure so the age policy can be tested directly.
fn prepared_sink_is_fresh(prepared_at: Instant, now: Instant, max_age: Duration) -> bool {
    now.saturating_duration_since(prepared_at) <= max_age
}

/// Whether the maintenance worker should proactively rotate a socket out
/// (replace before it can expire at checkout). Pure for testability.
fn prepared_sink_should_rotate(prepared_at: Instant, now: Instant, refresh_age: Duration) -> bool {
    now.saturating_duration_since(prepared_at) >= refresh_age
}

struct PoolState {
    sinks: VecDeque<PreparedReturnSink>,
    shutdown: bool,
}

pub(in crate::frontend) struct PreparedPredictionReturnPool {
    config: StageConfig,
    wire_dtype: WireActivationDType,
    telemetry: Telemetry,
    state: Mutex<PoolState>,
    /// Signals the maintenance worker to run a refill/rotate pass promptly
    /// (e.g. right after a checkout) instead of waiting for the next interval.
    wake: Condvar,
    next_sink_id: AtomicU64,
    capacity: usize,
    worker: Mutex<Option<JoinHandle<()>>>,
    worker_started: AtomicBool,
}

impl PreparedPredictionReturnPool {
    /// Build the pool and start its maintenance worker. Returns `None` when
    /// there is no downstream (single-stage: no return channel needed).
    pub(in crate::frontend) fn new(
        config: &StageConfig,
        capacity: usize,
        wire_dtype: WireActivationDType,
        telemetry: Telemetry,
    ) -> Option<Arc<Self>> {
        config.downstream.as_ref()?;
        let capacity = capacity.max(1);
        let pool = Arc::new(Self {
            config: config.clone(),
            wire_dtype,
            telemetry,
            state: Mutex::new(PoolState {
                sinks: VecDeque::with_capacity(capacity),
                shutdown: false,
            }),
            wake: Condvar::new(),
            next_sink_id: AtomicU64::new(0),
            capacity,
            worker: Mutex::new(None),
            worker_started: AtomicBool::new(false),
        });
        pool.start_worker();
        Some(pool)
    }

    fn start_worker(self: &Arc<Self>) {
        if self.worker_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let worker_pool = Arc::clone(self);
        let handle = std::thread::Builder::new()
            .name("prediction-return-pool".to_string())
            .spawn(move || worker_pool.maintenance_loop())
            .ok();
        if let Ok(mut slot) = self.worker.lock() {
            *slot = handle;
        }
    }

    /// Take a pre-warmed socket and bind it to `(request_id, session_id)` by
    /// writing `PredictionReturnOpen`. Returns `None` when the pool is empty or
    /// every pooled socket is stale/failed to bind — callers then fall back to
    /// the cold open. Never blocks on a network connect; signals the maintenance
    /// worker to replenish the consumed slot.
    pub(in crate::frontend) fn checkout_bound(
        &self,
        request_id: u64,
        session_id: u64,
    ) -> Option<std::net::TcpStream> {
        let result = loop {
            let Some(mut sink) = self.pop_fresh_sink() else {
                break None;
            };
            match direct_return::bind_downstream_prediction_return_socket(
                &mut sink.stream,
                request_id,
                session_id,
                self.wire_dtype,
            ) {
                Ok(()) => {
                    self.emit_checkout(sink.id, true, None);
                    break Some(sink.stream);
                }
                Err(error) => {
                    // A prepared socket that fails to bind was silently dead;
                    // drop it and try the next pooled socket. `bind` is a local
                    // write, so this only catches already-broken sockets — true
                    // end-to-end confirmation is the receiver's selection ACK.
                    self.emit_checkout(
                        sink.id,
                        false,
                        Some(direct_return::classify_return_failure_phase(&format!(
                            "{error:#}"
                        ))),
                    );
                    continue;
                }
            }
        };
        // Wake the maintenance worker to refill the consumed slot promptly,
        // whether or not we got a socket (empty pool should refill fast too).
        self.wake.notify_one();
        result
    }

    fn pop_fresh_sink(&self) -> Option<PreparedReturnSink> {
        let now = Instant::now();
        let mut state = self.state.lock().ok()?;
        while let Some(sink) = state.sinks.pop_front() {
            if prepared_sink_is_fresh(sink.prepared_at, now, PREPARED_SINK_MAX_AGE) {
                return Some(sink);
            }
            // aged out; drop and continue
        }
        None
    }

    fn maintenance_loop(self: Arc<Self>) {
        loop {
            if self.is_shutdown() {
                return;
            }
            self.purge_and_refill();
            // Wait for a checkout signal or the maintenance interval, whichever
            // comes first. Timeout drives idle rotation; the signal drives
            // prompt refill under load.
            let Ok(state) = self.state.lock() else {
                return;
            };
            let (_guard, _timeout) = match self.wake.wait_timeout(state, MAINTENANCE_INTERVAL) {
                Ok(pair) => pair,
                Err(_) => return,
            };
        }
    }

    /// Purge expired/near-expiry sockets and prepare new ones up to capacity.
    /// Network prepares happen OUTSIDE the state lock; the lock is only held for
    /// bookkeeping, so a slow WAN handshake never blocks checkout.
    fn purge_and_refill(&self) {
        // Phase 1 (locked): drop sockets due for rotation, count the deficit.
        let deficit = {
            let now = Instant::now();
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if state.shutdown {
                return;
            }
            state.sinks.retain(|s| {
                !prepared_sink_should_rotate(s.prepared_at, now, PREPARED_SINK_REFRESH_AGE)
            });
            self.capacity.saturating_sub(state.sinks.len())
        };
        // Phase 2 (unlocked): prepare up to `deficit` sockets. Serialized here
        // because only this worker prepares, so capacity cannot be overshot.
        for _ in 0..deficit {
            if self.is_shutdown() {
                return;
            }
            match self.prepare_one() {
                Ok(sink) => {
                    let Ok(mut state) = self.state.lock() else {
                        return;
                    };
                    if state.shutdown {
                        return;
                    }
                    // Re-check capacity under lock in case checkout added nothing
                    // but rotation math changed; never exceed capacity.
                    if state.sinks.len() < self.capacity {
                        state.sinks.push_back(sink);
                    }
                }
                Err(error) => {
                    let mut attrs = lifecycle_attrs(&self.config);
                    attrs.insert(
                        "llama_stage.prediction_return.failure_phase".to_string(),
                        json!(direct_return::classify_return_failure_phase(&format!(
                            "{error:#}"
                        ))),
                    );
                    self.telemetry
                        .emit("stage.prediction_return_prepare_failed", attrs);
                    // Downstream transiently unreachable; stop this pass and try
                    // again next interval rather than hammering.
                    return;
                }
            }
        }
    }

    fn prepare_one(&self) -> anyhow::Result<PreparedReturnSink> {
        let stream = direct_return::prepare_downstream_prediction_return_socket(&self.config)?;
        Ok(PreparedReturnSink {
            id: self.next_sink_id.fetch_add(1, Ordering::Relaxed),
            stream,
            prepared_at: Instant::now(),
        })
    }

    fn is_shutdown(&self) -> bool {
        self.state.lock().map(|s| s.shutdown).unwrap_or(true)
    }

    fn emit_checkout(&self, sink_id: u64, bound: bool, failure_phase: Option<&'static str>) {
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.prediction_return_prepared_sink_id".to_string(),
            json!(sink_id),
        );
        attrs.insert(
            "llama_stage.prediction_return_prepared_bound".to_string(),
            json!(bound),
        );
        if let Some(phase) = failure_phase {
            attrs.insert(
                "llama_stage.prediction_return.failure_phase".to_string(),
                json!(phase),
            );
        }
        self.telemetry
            .emit("stage.prediction_return_prepared_checkout", attrs);
    }
}

impl Drop for PreparedPredictionReturnPool {
    fn drop(&mut self) {
        // Signal shutdown, wake the worker, and join it so parked sockets are
        // dropped (closed) deterministically before teardown.
        if let Ok(mut state) = self.state.lock() {
            state.shutdown = true;
            state.sinks.clear();
        }
        self.wake.notify_all();
        if let Ok(mut slot) = self.worker.lock()
            && let Some(handle) = slot.take()
        {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_within_max_age() {
        let now = Instant::now();
        let prepared = now - Duration::from_secs(10);
        assert!(prepared_sink_is_fresh(prepared, now, PREPARED_SINK_MAX_AGE));
    }

    #[test]
    fn stale_past_max_age() {
        let now = Instant::now();
        let prepared = now - Duration::from_secs(31);
        assert!(!prepared_sink_is_fresh(
            prepared,
            now,
            PREPARED_SINK_MAX_AGE
        ));
    }

    #[test]
    fn exactly_at_max_age_is_fresh() {
        let now = Instant::now();
        let prepared = now - PREPARED_SINK_MAX_AGE;
        assert!(prepared_sink_is_fresh(prepared, now, PREPARED_SINK_MAX_AGE));
    }

    #[test]
    fn future_prepared_at_is_fresh_not_panic() {
        let now = Instant::now();
        let prepared = now + Duration::from_secs(5);
        assert!(prepared_sink_is_fresh(prepared, now, PREPARED_SINK_MAX_AGE));
    }

    #[test]
    fn rotates_at_refresh_age_before_max_age() {
        let now = Instant::now();
        // A socket past the refresh age but before max age should rotate (worker
        // replaces it) yet still be handable if reached at checkout — proving
        // refresh < max keeps the pool warm ahead of expiry.
        let prepared = now - Duration::from_secs(20);
        assert!(prepared_sink_should_rotate(
            prepared,
            now,
            PREPARED_SINK_REFRESH_AGE
        ));
        assert!(prepared_sink_is_fresh(prepared, now, PREPARED_SINK_MAX_AGE));
    }

    #[test]
    fn does_not_rotate_before_refresh_age() {
        let now = Instant::now();
        let prepared = now - Duration::from_secs(5);
        assert!(!prepared_sink_should_rotate(
            prepared,
            now,
            PREPARED_SINK_REFRESH_AGE
        ));
    }

    #[test]
    fn refresh_age_is_below_max_age() {
        // Invariant that keeps the pool warm: rotate before expiry.
        assert!(PREPARED_SINK_REFRESH_AGE < PREPARED_SINK_MAX_AGE);
    }
}
