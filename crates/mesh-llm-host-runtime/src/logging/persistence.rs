//! Concrete `PersistSink` backed by `LogStore` + `ArtifactFileStore`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use mesh_llm_log_store::{ArtifactFileStore, Clock, LogStore, LogStoreError, RealClock};

use crate::logging::registry::RequestSummaryEntry;
use crate::logging::service::PersistSink;
use mesh_llm_events::logging::replay::ReplayChannel;

/// A `PersistSink` that durably stores log entries in SQLite via `LogStore`
/// (accessed through the owning `ArtifactFileStore`) and artifact files via
/// `ArtifactFileStore`.
///
/// Construction: `StoreBackedSink::open(foundation)` opens the store and
/// artifact store under the foundation's `store_dir` / `artifact_dir`.
/// The `ArtifactFileStore` owns the `LogStore`; access the store via
/// `artifact_store.store_ref()`.
pub struct StoreBackedSink {
    artifact_store: Arc<ArtifactFileStore>,
}

impl StoreBackedSink {
    /// Open a store-backed sink rooted at the given store and artifact
    /// directories, using the system clock.
    pub fn open(
        store_dir: std::path::PathBuf,
        artifact_dir: std::path::PathBuf,
    ) -> Result<Self, LogStoreError> {
        let clock: Arc<dyn Clock> = Arc::new(RealClock);
        let log_store = LogStore::open(&store_dir, clock.clone())?;
        let artifact_store = ArtifactFileStore::open(artifact_dir, clock, log_store)?;
        Ok(Self {
            artifact_store: Arc::new(artifact_store),
        })
    }

    /// Open a store-backed sink with an injected clock (for tests).
    pub fn open_with_clock(
        store_dir: std::path::PathBuf,
        artifact_dir: std::path::PathBuf,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, LogStoreError> {
        let log_store = LogStore::open(&store_dir, clock.clone())?;
        let artifact_store = ArtifactFileStore::open(artifact_dir, clock, log_store)?;
        Ok(Self {
            artifact_store: Arc::new(artifact_store),
        })
    }

    /// Access the underlying artifact file store (for startup recovery and
    /// scheduled cleanup).
    pub fn artifact_store(&self) -> &Arc<ArtifactFileStore> {
        &self.artifact_store
    }

    /// Access the underlying log store (for queries and cleanup).
    pub fn log_store(&self) -> &LogStore {
        self.artifact_store.store_ref()
    }

    /// Map a `LogStoreError` to the string error expected by `PersistSink`.
    fn map_err(e: LogStoreError) -> String {
        e.to_string()
    }
}

#[async_trait]
impl PersistSink for StoreBackedSink {
    async fn persist_summary(&self, entry: RequestSummaryEntry) -> Result<(), String> {
        let store = self.log_store();
        store
            .insert_summary(
                &entry.request_id,
                None,
                None,
                None,
                None,
                &entry.created_at,
                None,
                None,
                None,
            )
            .map_err(Self::map_err)
    }

    async fn persist_event(
        &self,
        request_id: String,
        event_id: String,
        _channel: ReplayChannel,
        _sequence: u64,
        occurred_at: String,
        payload_json: String,
    ) -> Result<(), String> {
        let store = self.log_store();
        store
            .insert_lifecycle_event(&request_id, &event_id, &payload_json, &occurred_at)
            .map_err(Self::map_err)
    }

    async fn persist_artifact_pointer(
        &self,
        request_id: String,
        artifact_data: Value,
    ) -> Result<(), String> {
        let store = self.log_store();
        let artifact_id = Uuid::new_v4().to_string();
        let occurred_at = store.now();
        let kind = artifact_data
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let metadata = if artifact_data.is_null() {
            None
        } else {
            Some(artifact_data.to_string())
        };
        store
            .insert_artifact_pointer(
                &artifact_id,
                &request_id,
                &occurred_at,
                kind,
                metadata.as_deref(),
            )
            .map_err(Self::map_err)
    }

    async fn persist_proxy_record(&self, proxy_json: String) -> Result<(), String> {
        let store = self.log_store();
        let parsed: Value =
            serde_json::from_str(&proxy_json).map_err(|e| format!("invalid proxy json: {e}"))?;
        let attempt_id = Uuid::new_v4().to_string();
        let request_id = parsed
            .get("request_id")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let occurred_at = store.now();
        let target = parsed
            .get("target")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let provider = parsed.get("provider").and_then(Value::as_str);
        let engine = parsed.get("engine").and_then(Value::as_str);
        let started_at = parsed.get("started_at").and_then(Value::as_str);
        let completed_at = parsed.get("completed_at").and_then(Value::as_str);
        let status_code = parsed.get("status_code").and_then(Value::as_i64);
        let error_msg = parsed.get("error").and_then(Value::as_str);
        store
            .insert_proxy_record(
                &attempt_id,
                &request_id,
                &occurred_at,
                target,
                provider,
                engine,
                started_at,
                completed_at,
                status_code,
                error_msg,
            )
            .map_err(Self::map_err)
    }

    async fn persist_audit_entry(&self, level: String, message: String) -> Result<(), String> {
        let store = self.log_store();
        let entry_id = Uuid::new_v4().to_string();
        let occurred_at = store.now();
        let detail = serde_json::json!({ "level": level, "message": message }).to_string();
        store
            .insert_audit_entry(
                &entry_id,
                None,
                &occurred_at,
                "system",
                "log",
                Some(&detail),
            )
            .map_err(Self::map_err)
    }

    async fn persist_webhook_delivery(
        &self,
        request_id: Option<String>,
        status_code: u16,
        error: Option<String>,
    ) -> Result<(), String> {
        let store = self.log_store();
        let delivery_id = Uuid::new_v4().to_string();
        let occurred_at = store.now();
        store
            .insert_webhook_delivery(
                &delivery_id,
                request_id.as_deref(),
                &occurred_at,
                "webhook",
                1,
                Some(status_code as i64),
                None,
                error.as_deref(),
            )
            .map_err(Self::map_err)
    }

    async fn persist_cleanup_run(&self, deleted_count: u64) -> Result<(), String> {
        let store = self.log_store();
        let run_id = Uuid::new_v4().to_string();
        let occurred_at = store.now();
        store
            .insert_cleanup_run(
                &run_id,
                &occurred_at,
                "scheduled",
                &occurred_at,
                deleted_count as i64,
                None,
            )
            .map_err(Self::map_err)
    }
}
