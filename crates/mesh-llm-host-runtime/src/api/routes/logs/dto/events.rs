use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_log_store::EventRecord;
use serde::Serialize;

use super::safe_metadata;
use crate::api::routes::logs::error::LogsError;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::api::routes::logs) struct EventDto {
    event_id: String,
    request_id: String,
    occurred_at: String,
    kind: &'static str,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    attempt_id: Option<String>,
    status_code: Option<u16>,
    duration_ms: Option<u64>,
    tokens: Option<u64>,
}

impl TryFrom<EventRecord> for EventDto {
    type Error = LogsError;

    fn try_from(record: EventRecord) -> Result<Self, Self::Error> {
        let event = serde_json::from_str::<LifecycleEvent>(&record.payload_json)
            .map_err(|_| LogsError::StoreUnavailable)?;
        let mut dto = Self {
            event_id: record.event_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            kind: event_kind(&event),
            model: None,
            provider: None,
            engine: None,
            attempt_id: None,
            status_code: None,
            duration_ms: None,
            tokens: None,
        };
        match event {
            LifecycleEvent::Admitted { model, .. } | LifecycleEvent::StreamStarted { model } => {
                dto.model = model.map(|value| safe_metadata(&value));
            }
            LifecycleEvent::RouteSelected {
                model,
                provider,
                engine,
            } => {
                dto.model = model.map(|value| safe_metadata(&value));
                dto.provider = provider.map(|value| safe_metadata(&value));
                dto.engine = engine.map(|value| safe_metadata(&value));
            }
            LifecycleEvent::AttemptStarted { attempt_id } => {
                dto.attempt_id = attempt_id.map(|id| id.to_string());
            }
            LifecycleEvent::AttemptCompleted {
                attempt_id,
                status_code,
            } => {
                dto.attempt_id = attempt_id.map(|id| id.to_string());
                dto.status_code = status_code;
            }
            LifecycleEvent::AttemptFailed { attempt_id, .. } => {
                dto.attempt_id = attempt_id.map(|id| id.to_string());
            }
            LifecycleEvent::StreamChunk { tokens } | LifecycleEvent::StreamCompleted { tokens } => {
                dto.tokens = tokens
            }
            LifecycleEvent::Completed {
                status_code,
                duration_ms,
            } => {
                dto.status_code = status_code;
                dto.duration_ms = duration_ms;
            }
            LifecycleEvent::StreamError { .. }
            | LifecycleEvent::Failed { .. }
            | LifecycleEvent::Rejected { .. }
            | LifecycleEvent::Cancelled { .. }
            | LifecycleEvent::Dropped { .. } => {}
        }
        Ok(dto)
    }
}

fn event_kind(event: &LifecycleEvent) -> &'static str {
    match event {
        LifecycleEvent::Admitted { .. } => "admitted",
        LifecycleEvent::RouteSelected { .. } => "route_selected",
        LifecycleEvent::AttemptStarted { .. } => "attempt_started",
        LifecycleEvent::AttemptCompleted { .. } => "attempt_completed",
        LifecycleEvent::AttemptFailed { .. } => "attempt_failed",
        LifecycleEvent::StreamStarted { .. } => "stream_started",
        LifecycleEvent::StreamChunk { .. } => "stream_chunk",
        LifecycleEvent::StreamCompleted { .. } => "stream_completed",
        LifecycleEvent::StreamError { .. } => "stream_error",
        LifecycleEvent::Completed { .. } => "completed",
        LifecycleEvent::Failed { .. } => "failed",
        LifecycleEvent::Rejected { .. } => "rejected",
        LifecycleEvent::Cancelled { .. } => "cancelled",
        LifecycleEvent::Dropped { .. } => "dropped",
    }
}
