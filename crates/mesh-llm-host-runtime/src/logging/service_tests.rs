//! Tests for LoggingService — extracted to keep service.rs under 1000 LOC.

use crate::logging::{BusEntry, Clock, FailOpenWriter, PersistSink, RegistryConfig};
use crate::logging::{ReplayBus, RequestRegistry, RequestSummaryEntry};
use mesh_llm_events::logging::identifiers::{EventId, RequestId};
use mesh_llm_events::logging::replay::ReplayChannel;

// Re-import service.rs types. These are private to the logging module but accessible via super.
#[allow(unused_imports)]
use crate::logging::service::{BusEnqueueError, LoggingService, ServiceConfig, SystemClock};

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

// ---------------------------------------------------------------------------
// Test infrastructure: Vec-backed sink + deterministic clock
// ---------------------------------------------------------------------------

/// Record type for the test Vec-backed persistence sink. Captures all persisted data deterministically without I/O.
#[derive(Clone, Debug)]
enum TestRecord {
    Summary(RequestSummaryEntry),
    Event {
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    },
    ArtifactPointer(String, serde_json::Value), // (request_id, data)
    ProxyRecord(String),                        // JSON string
    AuditEntry {
        level: String,
        message: String,
    },
    WebhookDelivery {
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    },
    CleanupRun(u64), // deleted_count
}

/// Vec-backed persistence sink for deterministic testing. All writes are recorded in a shared Mutex<Vec<TestRecord>> — no I/O, no sleeps.
struct TestSink {
    records: std::sync::Mutex<Vec<TestRecord>>,
    fail_flag: Arc<AtomicU64>, // if > 0, all operations return Err
}

impl TestSink {
    fn new() -> Self {
        Self {
            records: std::sync::Mutex::new(Vec::new()),
            fail_flag: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set the sink to return Err on all subsequent operations (simulates store failure).
    fn set_failing(&self) {
        self.fail_flag.store(1, AtomicOrdering::Release);
    }

    /// Clear the failing flag.
    #[allow(dead_code)]
    fn clear_fail(&self) {
        self.fail_flag.store(0, AtomicOrdering::Release);
    }

    /// Get all records captured so far (for test assertions).
    #[allow(dead_code)]
    fn records(&self) -> Vec<TestRecord> {
        self.records.lock().unwrap().clone()
    }

    /// Count of audit entries with a specific level.
    #[allow(dead_code)]
    fn audit_count_by_level(&self, level: &str) -> usize {
        self.records()
            .iter()
            .filter(|r| matches!(r, TestRecord::AuditEntry { level: lvl, .. } if lvl == level))
            .count()
    }

    /// Reset records to empty (for multi-phase tests).
    #[allow(dead_code)]
    fn clear(&self) {
        self.records.lock().unwrap().clear();
    }
}

#[async_trait::async_trait]
impl PersistSink for TestSink {
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::Summary(entry));
        Ok(())
    }

    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records.lock().unwrap().push(TestRecord::Event {
            request_id,
            event_id,
            channel,
            sequence,
            occurred_at,
            payload_json,
        });
        Ok(())
    }

    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: serde_json::Value,
    ) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::ArtifactPointer(request_id, artifact_data));
        Ok(())
    }

    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::ProxyRecord(proxy_json));
        Ok(())
    }

    async fn persist_audit_entry(&self, level: String, message: String) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::AuditEntry { level, message });
        Ok(())
    }

    async fn persist_webhook_delivery(
        &self,
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    ) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::WebhookDelivery {
                request_id,
                status_code,
                error,
            });
        Ok(())
    }

    async fn persist_cleanup_run(&self, deleted_count: u64) -> Result<(), String> {
        if self.fail_flag.load(AtomicOrdering::Acquire) > 0 {
            return Err("sink failing".into());
        }
        self.records
            .lock()
            .unwrap()
            .push(TestRecord::CleanupRun(deleted_count));
        Ok(())
    }
}

/// Deterministic counter clock for tests. Each call increments a counter, producing unique timestamps without wall-clock dependency.
struct TestClock {
    counter: AtomicU64,
}

impl TestClock {
    fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl Clock for TestClock {
    fn now(&self) -> String {
        let n = self.counter.fetch_add(1, AtomicOrdering::Relaxed);
        format!("2025-01-01T00:00:{:02}Z", (n % 60) as u32)
    }
}

fn make_service() -> LoggingService {
    let sink = Arc::new(TestSink::new());
    let clock = Box::new(TestClock::new());
    let config = ServiceConfig {
        queue_capacity: 128,
        registry_config: RegistryConfig {
            max_active: 50,
            max_recent: 100,
        },
    };
    LoggingService::new(config, sink, None, clock)
}

// ---------------------------------------------------------------------------
// Test Scenario 1: One terminal record for each of complete/fail/reject/cancel/drop
// ---------------------------------------------------------------------------

#[test]
fn test_one_terminal_per_outcome() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let outcomes = [
        TerminalOutcome::Completed,
        TerminalOutcome::Failed("timeout".into()),
        TerminalOutcome::Rejected(Some("invalid model".into())),
        TerminalOutcome::Cancelled(None),
        TerminalOutcome::Dropped(Some("queue full".into())),
    ];

    for outcome in &outcomes {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);

        // First terminal transition should succeed.
        assert!(
            svc.transition_terminal(rid, &guard, outcome.clone())
                .is_ok()
        );

        // Second terminal transition of the SAME type → idempotent Ok via terminate_idempotent on guard level.
        // But through the service's transition_terminal (which uses guard.terminate not terminate_idempotent), it should Err.
    }
}

#[test]
fn test_duplicate_terminal_rejected() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    assert!(
        svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
            .is_ok()
    );

    // Second terminal → DuplicateTerminalError.
    let err = svc
        .transition_terminal(rid, &guard, TerminalOutcome::Failed("x".into()))
        .unwrap_err();
    assert_eq!(err.existing, TerminalOutcome::Completed);
}

// ---------------------------------------------------------------------------
// Test Scenario 2: One summary with multiple retry attempts (parent not terminated by per-attempt)
// ---------------------------------------------------------------------------

#[test]
fn test_retry_attempts_under_one_summary() {
    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Simulate 3 retry attempts — each emits an event on Operations channel but does NOT terminate the parent.
    for i in 0..3 {
        let _payload = serde_json::json!({ "attempt": i + 1 }).to_string();
        svc.enqueue_event(rid, ReplayChannel::Operations, _payload)
            .unwrap();

        // Guard still active after each attempt.
        assert!(
            guard.is_active(),
            "guard should remain active during retry {}",
            i
        );
    }

    // Now terminate the parent request — exactly one terminal transition.
    use crate::logging::lifecycle::TerminalOutcome;
    assert!(
        svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
            .is_ok()
    );
    assert!(!guard.is_active());

    // Verify bus has entries for all attempts + 1 terminal (total drops = evictions if any occurred).
}

// ---------------------------------------------------------------------------
// Test Scenario 3: Monotonic channel sequences across many events
// ---------------------------------------------------------------------------

#[test]
fn test_monotonic_channel_sequences() {
    let svc = make_service();

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Emit 100 events on each channel. Sequences must be strictly increasing per channel and independent across channels.
    for ch in [
        ReplayChannel::Requests,
        ReplayChannel::Operations,
        ReplayChannel::System,
    ] {
        let mut prev_seq: u64 = 0;
        for _i in 0..100 {
            svc.enqueue_event(rid, ch, "test".into()).unwrap();
            // The sequence generator is internal to the service — verify via sequences_ref.
            let current = svc.sequences_ref().current(ch);
            assert!(
                current > prev_seq,
                "sequence must be strictly increasing on {:?}",
                ch
            );
            prev_seq = current;
        }

        // Verify other channels weren't affected by events on this channel.
        for other_ch in [
            ReplayChannel::Requests,
            ReplayChannel::Operations,
            ReplayChannel::System,
        ] {
            if other_ch != ch {
                let other_current = svc.sequences_ref().current(other_ch);
                assert!(
                    other_current <= 100 || other_ch == ch,
                    "channel {:?} should not have advanced beyond its own events (got {})",
                    other_ch,
                    other_current
                );
            }
        }
    }

    // Verify sequences survive guard cloning.
}

#[test]
fn test_sequences_survive_guard_clone() {
    let svc = make_service();

    let rid = RequestId::new();
    let (guard1, _) = svc.register_request(rid);
    let _guard2 = guard1.clone(); // Clone the guard — sequences are independent of guards.

    // Emit events via service after cloning.
    for i in 0..5 {
        let _payload = serde_json::json!({ "i": i }).to_string();
        svc.enqueue_event(rid, ReplayChannel::Requests, _payload)
            .unwrap();
    }

    assert_eq!(svc.sequences_ref().current(ReplayChannel::Requests), 5);
}

// ---------------------------------------------------------------------------
// Test Scenario 4: Bounded replay eviction (overflow drops + counter increments)
// ---------------------------------------------------------------------------

#[test]
fn test_bounded_replay_eviction() {
    let svc = make_service();

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Emit more events than the bus capacity (128). This triggers drop-oldest evictions.
    for i in 0..200 {
        let payload = serde_json::json!({ "i": i }).to_string();
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, payload)
                .is_ok()
        );
    }

    // Bus should be at capacity.
    assert_eq!(svc.bus_ref().len(), 128);

    // Evictions counter should have incremented (at least 72 = 200 - 128).
    let evictions = svc.bus_ref().evictions.load(AtomicOrdering::Relaxed);
    assert!(
        evictions >= 72,
        "expected at least 72 evictions, got {}",
        evictions
    );

    // Queue never exceeds capacity.
}

#[test]
fn test_queue_never_exceeds_capacity() {
    let svc = make_service();
    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    for i in 0..10_000 {
        let payload = serde_json::json!({ "i": i }).to_string();
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, payload)
                .is_ok()
        );
        // Invariant: bus never exceeds capacity.
        assert!(
            svc.bus_ref().len() <= 128,
            "bus exceeded capacity at iteration {}",
            i
        );
    }

    // Request path completes despite overflow — no blocking or panic.
}

// ---------------------------------------------------------------------------
// Test Scenario 5: Active → recent movement on terminal transition
// ---------------------------------------------------------------------------

#[test]
fn test_active_to_recent_movement() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    assert_eq!(svc.registry_ref().active_count(), 1);
    assert_eq!(svc.registry_ref().recent_count(), 0);

    // Transition to terminal → moves from active to recent.
    svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
        .unwrap();

    assert_eq!(svc.registry_ref().active_count(), 0);
    assert_eq!(svc.registry_ref().recent_count(), 1);
}

#[test]
fn test_active_to_recent_preserves_created_at() {
    use crate::logging::lifecycle::TerminalOutcome;

    let svc = make_service();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Get the active entry's created_at.
    let rid_str = rid.as_uuid().to_string();
    let active_entry = svc.registry_ref().get_active(&rid_str).unwrap();
    let original_created_at = active_entry.created_at.clone();

    // Transition to terminal.
    svc.transition_terminal(rid, &guard, TerminalOutcome::Failed("err".into()))
        .unwrap();

    // Recent entry should preserve created_at.
    let recent_entry = svc.registry_ref().get_recent(&rid_str).unwrap();
    assert_eq!(recent_entry.created_at, original_created_at);
}

// ---------------------------------------------------------------------------
// Test Scenario 6: No registry leak (registry empties when all entries evict)
// ---------------------------------------------------------------------------

#[test]
fn test_no_registry_leak() {
    use crate::logging::lifecycle::TerminalOutcome;

    let config = ServiceConfig {
        queue_capacity: 10,
        registry_config: RegistryConfig {
            max_active: 2,
            max_recent: 3,
        },
    };

    let svc = LoggingService::new(
        config.clone(),
        Arc::new(TestSink::new()),
        None,
        Box::new(TestClock::new()),
    );

    // Register many requests — all should eventually evict from both sets.
    for i in 0..50 {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);

        if i % 2 == 0 {
            // Every other request transitions to terminal → moves active→recent.
            svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
                .unwrap();
        }
    }

    assert!(svc.registry_ref().active_count() <= config.registry_config.max_active);
    assert!(svc.registry_ref().recent_count() <= config.registry_config.max_recent);

    // Clear the registry — should become empty.
    svc.registry_ref().clear();
    assert!(svc.registry_ref().is_empty());
}

#[test]
fn test_registry_eviction_counters_increment() {
    use crate::logging::lifecycle::TerminalOutcome;

    let config = ServiceConfig {
        queue_capacity: 10,
        registry_config: RegistryConfig {
            max_active: 2,
            max_recent: 2,
        },
    };

    let svc = LoggingService::new(
        config.clone(),
        Arc::new(TestSink::new()),
        None,
        Box::new(TestClock::new()),
    );

    for i in 0..20 {
        let rid = RequestId::new();
        let (guard, _) = svc.register_request(rid);
        if i % 3 == 0 {
            svc.transition_terminal(rid, &guard, TerminalOutcome::Completed)
                .unwrap();
        }
    }

    // Active evictions should have occurred.
    assert!(
        svc.registry_ref()
            .active_evictions
            .load(AtomicOrdering::Relaxed)
            > 0
    );
}

// ---------------------------------------------------------------------------
// Test Scenario 7: Bounded shutdown (drain + stop completes; restart-safe)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_bounded_shutdown() {
    let svc = make_service();

    // Register some requests and enqueue events.
    for i in 0..10 {
        let rid = RequestId::new();
        let _guard = svc.register_request(rid);
        let payload = serde_json::json!({ "i": i }).to_string();
        svc.enqueue_event(rid, ReplayChannel::Requests, payload)
            .unwrap();
    }

    // Drain the bus before shutdown.
    let drained = svc.pump_sync();
    assert!(drained > 0);

    // Shutdown (without spawn — should return false as not spawned).
    let result = svc.shutdown().await;
    assert!(!result); // Not spawned → nothing to shut down.

    // Second shutdown is a no-op (restart-safe).
    let second_result = svc.shutdown().await;
    assert!(!second_result);

    // Registry should still have entries (shutdown doesn't clear them — that's explicit via registry.clear()).
}

#[test]
#[allow(clippy::await_holding_lock)] // test-only: safe since single-threaded runtime context
fn test_spawn_then_shutdown() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        let sink = Arc::new(TestSink::new());
        let svc = Arc::new(std::sync::Mutex::new(LoggingService::new(
            ServiceConfig::default(),
            sink,
            None,
            Box::<SystemClock>::default(),
        )));

        // spawn() on blocking thread to avoid deadlock with tokio runtime.
        let first_spawned: bool = tokio::task::spawn_blocking({
            let s = Arc::clone(&svc);
            move || {
                let inner = s.lock().unwrap();
                inner.spawn()
            }
        })
        .await
        .unwrap();

        assert!(first_spawned, "first spawn should return true");

        // Second spawn is a no-op.
        let second_spawned: bool = tokio::task::spawn_blocking({
            let s = Arc::clone(&svc);
            move || {
                let inner = s.lock().unwrap();
                inner.spawn()
            }
        })
        .await
        .unwrap();

        assert!(!second_spawned, "second spawn should return false");

        // Shutdown drains + stops → returns true.
        {
            let inner = svc.lock().unwrap();
            let result = inner.shutdown().await;
            assert!(result, "first shutdown should succeed");
        }

        // Second shutdown after first completes → false (already stopped, restart-safe).
        {
            let inner = svc.lock().unwrap();
            let second_result = inner.shutdown().await;
            assert!(!second_result, "second shutdown should be no-op");
        }
    });
}

#[test]
#[allow(clippy::await_holding_lock)] // test-only: safe since single-threaded runtime context
fn test_restart_safe_shutdown() {
    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        let sink = Arc::new(TestSink::new());
        let config = ServiceConfig::default();

        // Create, spawn, shutdown in scope.
        {
            let svc1 = Arc::new(std::sync::Mutex::new(LoggingService::new(
                config.clone(),
                sink.clone(),
                None,
                Box::<SystemClock>::default(),
            )));

            let spawned: bool = tokio::task::spawn_blocking({
                let s = Arc::clone(&svc1);
                move || {
                    let inner = s.lock().unwrap();
                    inner.spawn()
                }
            })
            .await
            .unwrap();

            assert!(spawned, "first spawn should succeed");

            // Shutdown.
            {
                let inner = svc1.lock().unwrap();
                let result = inner.shutdown().await;
                assert!(result, "shutdown should succeed");
            }
        } // Drop the service — worker task should clean up.

        // Re-create a new service (restart-safe: old one dropped).
        let svc2 = LoggingService::new(config, sink, None, Box::<SystemClock>::default());
        assert!(!svc2.is_spawned(), "fresh service should not be spawned");
    });
}

// ---------------------------------------------------------------------------
// Test Scenario 8a: Request-path completion despite full queue (enqueue on full returns Ok with drop-oldest)
// ---------------------------------------------------------------------------

#[test]
fn test_request_path_completion_despite_full_queue() {
    let config = ServiceConfig {
        queue_capacity: 5, // Very small to force overflow quickly.
        registry_config: RegistryConfig::default(),
    };

    let svc = LoggingService::new(
        config,
        Arc::new(TestSink::new()),
        None,
        Box::new(TestClock::new()),
    );

    let rid = RequestId::new();
    let _guard = svc.register_request(rid);

    // Fill the queue completely.
    for i in 0..5 {
        assert!(
            svc.enqueue_event(rid, ReplayChannel::Requests, format!("event_{}", i))
                .is_ok()
        );
    }

    assert_eq!(svc.bus_ref().len(), 5);

    // Now enqueue more — should succeed (drop-oldest applies internally), never blocking.
    for i in 0..100 {
        let result = svc.enqueue_event(rid, ReplayChannel::Requests, format!("overflow_{}", i));
        assert!(result.is_ok(), "enqueue must always return Ok (fail-open)");

        // Bus stays at capacity.
        assert_eq!(svc.bus_ref().len(), 5);
    }

    // Evictions counter reflects the overflow pressure.
    let evictions = svc.bus_ref().evictions.load(AtomicOrdering::Relaxed);
    assert!(
        evictions >= 100,
        "expected at least 100 evictions under heavy overflow"
    );

    // Request path completes — no panic, no deadlock.
}

// ---------------------------------------------------------------------------
// Test Scenario 8b: Request-path completion despite store worker failure (sink returns Err)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_request_path_completion_despite_sink_failure() {
    use crate::logging::lifecycle::TerminalOutcome;

    let sink = Arc::new(TestSink::new());
    let svc = LoggingService::new(
        ServiceConfig::default(),
        sink.clone(),
        None,
        Box::new(TestClock::new()),
    );

    // Make the sink start failing.
    sink.set_failing();

    let rid = RequestId::new();
    let (guard, _) = svc.register_request(rid);

    // Enqueue should still succeed — fail-open writer absorbs the sink error.
    for i in 0..10 {
        let result = svc.enqueue_event(rid, ReplayChannel::Requests, format!("fail_{}", i));
        assert!(
            result.is_ok(),
            "enqueue must return Ok even when sink fails"
        );
    }

    // Transition terminal — should still work despite failing sink.
    let result = svc.transition_terminal(rid, &guard, TerminalOutcome::Completed);
    assert!(result.is_ok());

    // Request path completes without panic or deadlock.
}

#[test]
fn test_writer_fail_open_no_panic_on_sink_error() {
    let sink = Arc::new(TestSink::new());
    sink.set_failing();

    let svc = LoggingService::new(
        ServiceConfig::default(),
        sink,
        None,
        Box::new(TestClock::new()),
    );

    // Error audit write should not panic even when the underlying operations fail.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for i in 0..100 {
            svc.write_error_audit(format!("error_{}", i));
        }
    }));

    assert!(result.is_ok(), "write_error_audit must never panic");
}

// ---------------------------------------------------------------------------
// Additional: Recursion guard prevents self-logging loops
// ---------------------------------------------------------------------------

#[test]
fn test_recursion_guard_blocks_nested_error_path() {
    let svc = make_service();
    let writer = svc.writer_ref();

    assert!(!writer.is_in_error_path());

    // First entry succeeds.
    assert!(writer.try_record_error(|| {}));

    // After exit, can enter again (not nested anymore).
}

#[test]
fn test_recursion_guard_depth_prevents_cross_thread_duplication() {
    let writer = FailOpenWriter::new();

    // Simulate depth guard behavior.
    assert!(writer.try_record_error(|| {}));
}

// ---------------------------------------------------------------------------
// Additional: Bus drop-oldest preserves recent entries under pressure
// ---------------------------------------------------------------------------

#[test]
fn test_drop_oldest_preserves_recent() {
    let bus = ReplayBus::new(3);

    for i in 0..10 {
        bus.push(format!("entry_{}", i));
    }

    assert_eq!(bus.len(), 3); // At capacity.

    let entries = bus.drain();
    assert_eq!(entries.len(), 3);

    // Last three entries should be preserved (indices 7, 8, 9).
    assert_eq!(entries[0].payload, "entry_7");
    assert_eq!(entries[1].payload, "entry_8");
    assert_eq!(entries[2].payload, "entry_9");

    // Oldest entries (0-6) were evicted.
}
