use mesh_llm_log_store::RequestRecord;
use serde::Serialize;

use super::safe_metadata;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::api::routes::logs) struct RequestDto {
    request_id: String,
    outcome: String,
    created_at: String,
    terminal_at: Option<String>,
    route: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    status_code: Option<i64>,
    source: &'static str,
}

impl RequestDto {
    pub(in crate::api::routes::logs) fn durable(record: RequestRecord) -> Self {
        Self {
            request_id: record.request_id,
            outcome: record.outcome,
            created_at: record.created_at,
            terminal_at: record.terminal_at,
            route: record.route.map(|value| safe_metadata(&value)),
            model: record.model.map(|value| safe_metadata(&value)),
            provider: record.provider.map(|value| safe_metadata(&value)),
            engine: record.engine.map(|value| safe_metadata(&value)),
            status_code: record.status_code,
            source: "durable",
        }
    }

    pub(in crate::api::routes::logs) fn active(
        record: crate::logging::RequestSummaryEntry,
        metadata: Option<RequestRecord>,
    ) -> Self {
        let (route, model, provider, engine, status_code) = metadata
            .map(|metadata| {
                (
                    metadata.route.map(|value| safe_metadata(&value)),
                    metadata.model.map(|value| safe_metadata(&value)),
                    metadata.provider.map(|value| safe_metadata(&value)),
                    metadata.engine.map(|value| safe_metadata(&value)),
                    metadata.status_code,
                )
            })
            .unwrap_or_default();
        Self {
            request_id: record.request_id,
            outcome: record.state,
            created_at: record.created_at,
            terminal_at: record.terminal_at,
            route,
            model,
            provider,
            engine,
            status_code,
            source: "active",
        }
    }

    pub(in crate::api::routes::logs) fn request_id(&self) -> &str {
        &self.request_id
    }

    pub(in crate::api::routes::logs) fn created_at(&self) -> &str {
        &self.created_at
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::api::routes::logs) struct PageDto<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
}
