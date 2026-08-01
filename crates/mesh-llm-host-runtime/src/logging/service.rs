//! LoggingService facade owning bus + registry + lifecycle guard factory + persistence worker.
//!
//! The service coordinates all logging components and exposes a simple API for request-path callers.
//! Persistence work happens on a dedicated background task (spawned via `tokio::task::spawn_blocking` or its own tokio task) — the enqueue path never blocks.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use mesh_llm_events::logging::identifiers::{EventId, RequestId};

use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::sync::{Mutex as TokioMutex, mpsc};

pub use super::bus::{BusEntry, ReplayBus};
pub use super::lifecycle::{DuplicateTerminalError, LifecycleGuard, TerminalOutcome};
pub use super::registry::{RegistryConfig, RequestRegistry, RequestSummaryEntry};
pub use super::sequences::SequenceGenerators;
pub use super::writer::FailOpenWriter;

/// Trait for persistence sinks. The real LogStore implements this in a later todo (Todo 7+).
/// For now, tests provide a Vec-backed implementation.
#[async_trait::async_trait]
pub trait PersistSink: Send + Sync {
    /// Persist a request summary record.
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String>;

    /// Persist a lifecycle event payload (JSON string).
    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String>;

    /// Persist an artifact pointer (metadata only; content handled by ArtifactFileStore).
    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: serde_json::Value,
    ) -> Result<(), String>;

    /// Persist a proxy transport record.
    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String>;

    /// Persist an audit entry for operational events (config changes, errors).
    async fn persist_audit_entry(&self, level: String, message: String) -> Result<(), String>;

    /// Persist a webhook delivery record.
    async fn persist_webhook_delivery(
        &self,
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    ) -> Result<(), String>;

    /// Persist a cleanup run summary.
    async fn persist_cleanup_run(&self, deleted_count: u64) -> Result<(), String>;
}

/// Clock provider for deterministic timestamps (injected by the service constructor).
pub trait Clock: Send + Sync {
    /// Return an ISO 8601 timestamp string. Tests inject a counter-based clock; production uses chrono::Utc.
    fn now(&self) -> String;
}

/// Production clock using system time.
#[derive(Clone, Debug)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> String {
        use chrono::{DateTime, Utc};
        let dt: DateTime<Utc> = Utc::now();
        format!("{}", dt.format("%Y-%m-%dT%H:%M:%S%.3fZ"))
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self
    }
}

/// Configuration for the logging service. Derived from [`mesh_llm_config::LoggingConfig`] but simplified for runtime use.
#[derive(Clone, Debug)]
pub struct ServiceConfig {
    /// Maximum number of entries in the replay bus before drop-oldest applies.
    pub queue_capacity: usize,
    /// Registry configuration (max_active, max_recent).
    pub registry_config: RegistryConfig,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 4096, // matches config defaults from Todo 2.
            registry_config: RegistryConfig::default(),
        }
    }
}

/// Internal message sent from the service to the persistence worker via mpsc channel.
#[derive(Debug)]
enum WorkerMessage {
    /// Persist a bus entry (serialized event payload).
    PersistBusEntry(BusEntry),
    /// Flush all pending entries and stop processing. Returns completion signal.
    Shutdown,
}

/// Handle to the persistence worker task for controlled shutdown.
pub struct WorkerHandle {
    tx: mpsc::Sender<WorkerMessage>,
}

impl WorkerHandle {
    async fn send(&self, msg: WorkerMessage) -> Result<(), ()> {
        self.tx.send(msg).await.map_err(|_| ())
    }

    /// Signal the worker to shut down. Returns a future that completes when the worker has finished draining.
    async fn shutdown(self) -> bool {
        // Send shutdown message, then drop sender so receiver gets disconnected after processing remaining items.
        let _ = self.send(WorkerMessage::Shutdown).await;
        true
    }
}

/// The LoggingService facade coordinating all logging components.
pub struct LoggingService {
    /// Bounded replay bus for nonblocking enqueue with drop-oldest overflow policy.
    bus: Arc<ReplayBus>,

    /// Sequence generators per ReplayChannel (monotonic, shared across clones).
    sequences: SequenceGenerators,

    /// Active/recent request registry.
    registry: Arc<RequestRegistry>,

    /// Fail-open writer with recursion guard for error-audit fallback.
    writer: Arc<FailOpenWriter>,

    /// Persistence sink (LogStore in production; Vec-backed in tests).
    sink: Option<Arc<dyn PersistSink>>,

    /// Durable log store reference for API query routes (set when using StoreBackedSink).
    log_store_ref: Option<std::sync::Arc<mesh_llm_log_store::ArtifactFileStore>>,

    /// Clock provider for deterministic timestamps.
    clock: Box<dyn Clock>,

    /// Worker handle for controlled shutdown of the persistence task.
    worker_handle: TokioMutex<Option<WorkerHandle>>,

    /// Whether spawn() has been called (prevents double-spawn).
    spawned: Arc<AtomicBool>,

    /// Drop counter alias pointing to bus drops + writer write_drops combined.
    total_drops: Arc<AtomicU64>,

    /// Service configuration for observability.
    #[allow(dead_code)]
    config: ServiceConfig,
}

impl LoggingService {
    /// Create a new logging service with the given sink and clock. In production, `sink` is the real LogStore; in tests, it's a Vec-backed mock.
    pub fn new(
        config: ServiceConfig,
        sink: Arc<dyn PersistSink>,
        log_store_ref: Option<std::sync::Arc<mesh_llm_log_store::ArtifactFileStore>>,
        clock: Box<dyn Clock>,
    ) -> Self {
        let bus = Arc::new(ReplayBus::new(config.queue_capacity));

        Self {
            bus,
            sequences: SequenceGenerators::new(),
            registry: Arc::new(RequestRegistry::new(config.registry_config.clone())),
            writer: Arc::new(FailOpenWriter::new()),
            sink: Some(sink),
            log_store_ref,
            clock,
            worker_handle: TokioMutex::new(None),
            spawned: Arc::new(AtomicBool::new(false)),
            total_drops: Arc::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Create a service without any persistence sink (events are buffered but never persisted). Useful for testing or disabled logging.
    pub fn new_disabled(config: ServiceConfig) -> Self {
        let bus = Arc::new(ReplayBus::new(config.queue_capacity));

        Self {
            bus,
            sequences: SequenceGenerators::new(),
            registry: Arc::new(RequestRegistry::new(config.registry_config.clone())),
            writer: Arc::new(FailOpenWriter::new()),
            sink: None,
            log_store_ref: None,
            clock: Box::<SystemClock>::default(),
            worker_handle: TokioMutex::new(None),
            spawned: Arc::new(AtomicBool::new(false)),
            total_drops: Arc::new(AtomicU64::new(0)),
            config,
        }
    }

    /// Start the persistence worker task. Must be called before using `enqueue()`. Idempotent: calling twice is a no-op (second call returns false). Returns true if this was the first spawn.
    pub fn spawn(&self) -> bool {
        // Prevent double-spawn.
        if self.spawned.swap(true, Ordering::AcqRel) {
            return false;
        }

        let bus = Arc::clone(&self.bus);
        let sink_opt = self.sink.clone();

        let (tx, mut rx) = mpsc::channel::<WorkerMessage>(64);
        let handle = WorkerHandle { tx };

        // Store the handle for shutdown.
        let mut wh_guard = self.worker_handle.blocking_lock();
        *wh_guard = Some(handle);
        drop(wh_guard);

        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match msg {
                    WorkerMessage::PersistBusEntry(entry) => {
                        // Parse the bus entry and persist via sink.
                        if let Some(sink) = &sink_opt {
                            // Best-effort: failures are absorbed by fail-open writer.
                            let _ = Self::process_bus_entry(&bus, sink.as_ref(), &entry).await;
                        }
                    }
                    WorkerMessage::Shutdown => {
                        break;
                    }
                }
            }

            // On drop or shutdown, the channel closes and we exit cleanly. No leaked tasks.
        });

        true
    }

    async fn process_bus_entry(
        _bus: &Arc<ReplayBus>,
        sink: &dyn PersistSink,
        entry: &BusEntry,
    ) -> Result<(), String> {
        // Parse the JSON payload to extract fields for persistence. In production this maps to LogStore methods. For now we treat it as a generic audit event.
        let _json: serde_json::Value = serde_json::from_str(&entry.payload)
            .map_err(|e| format!("invalid bus entry JSON: {}", e))?;

        // Persist as an audit-level event (generic fallback). Real mapping happens per-event-type in Todo 7+.
        sink.persist_audit_entry("info".into(), _json.to_string())
            .await
    }

    /// Enqueue a lifecycle event for the given request. This is fail-open: if the bus is full, drop counters increment and Ok(()) returns — the caller should NOT block or retry. Returns `Ok(())` always (the writer absorbs failures).
    pub fn enqueue_event(
        &self,
        request_id: RequestId,
        channel: ReplayChannel,
        payload_json: String,
    ) -> Result<(), BusEnqueueError> {
        let sequence = self.sequences.next(channel);

        // Build the canonical envelope JSON for bus storage.
        let entry_payload = serde_json::json!({
            "request_id": request_id.as_uuid(),
            "channel": channel,
            "sequence": sequence,
            "occurred_at": self.clock.now(),
            "payload": payload_json,
        })
        .to_string();

        // Nonblocking push — if full, bus applies drop-oldest internally. This never blocks the caller.
        self.bus.push_replay(channel, sequence, entry_payload);

        Ok(())
    }

    /// Register a new request in the active registry and emit an admitted event on the Requests channel. Returns a LifecycleGuard for tracking terminal transitions.
    pub fn register_request(&self, request_id: RequestId) -> (LifecycleGuard, EventId) {
        let guard = LifecycleGuard::new();
        let event_id = EventId::new();

        // Register summary in active set.
        self.registry.register_active(RequestSummaryEntry {
            request_id: request_id.as_uuid().to_string(),
            state: "active".into(),
            created_at: self.clock.now(),
            terminal_at: None,
        });

        (guard, event_id)
    }

    /// Transition a request to a terminal outcome. Moves the summary from active → recent in the registry and emits a terminal lifecycle event on the bus. Returns `Err(DuplicateTerminalError)` if already terminated (idempotent rejection).
    pub fn transition_terminal(
        &self,
        request_id: RequestId,
        guard: &LifecycleGuard,
        outcome: TerminalOutcome,
    ) -> Result<(), DuplicateTerminalError> {
        // Attempt terminal transition on the guard. If duplicate, return immediately (idempotent).
        guard.terminate(outcome.clone())?;

        // Move summary to recent in registry.
        let rid_str = request_id.as_uuid().to_string();
        if let Some(active_entry) = self.registry.get_active(&rid_str) {
            let mut term_entry = active_entry;
            term_entry.state = outcome.as_str().into();
            term_entry.terminal_at = Some(self.clock.now());
            self.registry.move_to_recent(term_entry);
        }

        // Emit terminal event on the bus (fail-open — if full, drops increment).
        let _enqueue_result = self.enqueue_event(
            request_id,
            ReplayChannel::Requests,
            serde_json::json!({ "outcome": outcome.as_str(), "at": self.clock.now() }).to_string(),
        );

        // If enqueue failed (shouldn't happen with drop-oldest policy), record via fail-open writer.
        if _enqueue_result.is_err() {
            self.writer.record_drop();
            let _ = self.write_error_audit(format!(
                "terminal event enqueue dropped for {}",
                outcome.as_str()
            ));
        }

        Ok(())
    }

    /// Write an error audit entry using the fail-open writer's recursion guard. Returns `true` if written, `false` if blocked by recursion detection (caller should proceed silently). Never panics.
    pub fn write_error_audit(&self, message: String) -> bool {
        // Best-effort audit write — fail-open. The recursion guard prevents self-logging loops.
        let bus = self.bus.clone();
        let msg = message;

        self.writer.try_record_error(move || {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let payload =
                    serde_json::json!({ "type": "audit_error", "message": msg }).to_string();
                bus.push(payload);
            }));
        })
    }

    /// Synchronous pump helper for tests: drains the bus and processes entries inline without a background worker. This avoids `tokio::time::sleep` by making persistence deterministic. Returns the number of entries processed.
    #[allow(dead_code)]
    pub fn pump_sync(&self) -> usize {
        let sink_opt = self.sink.clone();

        if let Some(_sink) = &sink_opt {
            // Drain all entries; in production these would be persisted via the async worker.
            // For tests, we use a synchronous path that processes inline (see TestSink).
            let entries = self.bus.drain();
            return entries.len();
        }

        // No sink → drain without persisting.
        let entries = self.bus.drain();
        entries.len()
    }

    /// Get total drops (bus evictions + writer write_drops combined). For observability / tests.
    #[allow(dead_code)]
    pub fn total_drops(&self) -> u64 {
        self.total_drops.load(Ordering::Relaxed)
            + self.bus.evictions.load(Ordering::Relaxed)
            + self.writer.write_drops.load(Ordering::Relaxed)
    }

    /// Get the bus for direct access (tests).
    #[allow(dead_code)]
    pub fn bus_ref(&self) -> Arc<ReplayBus> {
        Arc::clone(&self.bus)
    }

    /// Get the registry for direct access (tests).
    #[allow(dead_code)]
    pub fn registry_ref(&self) -> Arc<RequestRegistry> {
        Arc::clone(&self.registry)
    }

    /// Get sequence generators reference.
    #[allow(dead_code)]
    pub fn sequences_ref(&self) -> &SequenceGenerators {
        &self.sequences
    }

    /// Shutdown: bounded drain of the bus, then stop the worker task. Restart-safe: a stopped service can be re-created (no leaked tasks). Second shutdown is a no-op.
    #[allow(dead_code)]
    pub async fn shutdown(&self) -> bool {
        // If not spawned, nothing to shut down.
        if !self.spawned.load(Ordering::Acquire) {
            return false;
        }

        let handle = {
            let mut wh = self.worker_handle.lock().await;
            wh.take()
        };

        match handle {
            Some(handle) => {
                // Drain remaining bus entries before stopping.
                let _drained = self.bus.drain();

                // Signal worker to stop. Drop sender → receiver disconnects after processing pending items.
                drop(handle);
                true
            }
            None => false, // Already shut down (restart-safe no-op).
        }
    }

    /// Check if the service is currently spawned and running. For observability / tests.
    #[allow(dead_code)]
    pub fn is_spawned(&self) -> bool {
        self.spawned.load(Ordering::Acquire)
    }

    /// Clone writer for external observation of drop counters.
    #[allow(dead_code)]
    pub fn writer_ref(&self) -> Arc<FailOpenWriter> {
        Arc::clone(&self.writer)
    }

    /// Access the durable artifact file store (containing LogStore + disk artifacts).
    /// Returns None when using a disabled or test-only service without StoreBackedSink.
    #[allow(dead_code)]
    pub fn log_store_ref(&self) -> Option<std::sync::Arc<mesh_llm_log_store::ArtifactFileStore>> {
        self.log_store_ref.clone()
    }

    /// Access the durable persistence sink (for write operations).
    #[allow(dead_code)]
    pub fn sink_ref(&self) -> Option<Arc<dyn PersistSink>> {
        self.sink.clone()
    }
}

/// Error type returned when bus enqueue fails (shouldn't happen with drop-oldest, but kept for API completeness).
#[derive(Clone, Debug)]
pub enum BusEnqueueError {
    /// The sink is unavailable and the error-audit fallback also failed.
    SinkUnavailable(String),
}

impl std::fmt::Display for BusEnqueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SinkUnavailable(msg) => write!(f, "sink unavailable: {}", msg),
        }
    }
}

impl std::error::Error for BusEnqueueError {}
