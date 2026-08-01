use mesh_llm_log_store::ProxyRecord;
use serde::Serialize;

use super::safe_metadata;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::api::routes::logs) struct ProxyDto {
    attempt_id: String,
    request_id: String,
    occurred_at: String,
    target: String,
    provider: Option<String>,
    engine: Option<String>,
    started_at: Option<String>,
    completed_at: Option<String>,
    status_code: Option<i64>,
}

impl From<ProxyRecord> for ProxyDto {
    fn from(record: ProxyRecord) -> Self {
        Self {
            attempt_id: record.attempt_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            target: safe_target(&record.target),
            provider: record.provider.map(|value| safe_metadata(&value)),
            engine: record.engine.map(|value| safe_metadata(&value)),
            started_at: record.started_at,
            completed_at: record.completed_at,
            status_code: record.status_code,
        }
    }
}

fn safe_target(target: &str) -> String {
    let Ok(url) = url::Url::parse(target) else {
        return "opaque".to_string();
    };
    let Some(host) = url.host_str() else {
        return "opaque".to_string();
    };
    match url.port() {
        Some(port) => format!("{}://{host}:{port}", url.scheme()),
        None => format!("{}://{host}", url.scheme()),
    }
}
