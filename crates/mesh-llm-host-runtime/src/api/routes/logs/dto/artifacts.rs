use base64::Engine;
use mesh_llm_log_store::{ArtifactContent, ArtifactRecord};
use serde::Serialize;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(in crate::api::routes::logs) struct ArtifactDto {
    artifact_id: String,
    request_id: String,
    occurred_at: String,
    kind: String,
    media_kind: Option<String>,
    checksum: Option<String>,
    bytes: i64,
    version: i32,
    redacted: bool,
    truncated: bool,
    content_state: &'static str,
    content_base64: Option<String>,
}

impl ArtifactDto {
    pub(in crate::api::routes::logs) fn metadata(record: ArtifactRecord) -> Self {
        let content_state = artifact_state(&record);
        Self::from_parts(record, content_state, None)
    }

    pub(in crate::api::routes::logs) fn content(
        record: ArtifactRecord,
        content: ArtifactContent,
    ) -> Self {
        Self::from_parts(
            record,
            "available",
            Some(base64::engine::general_purpose::STANDARD.encode(content.bytes)),
        )
    }

    fn from_parts(
        record: ArtifactRecord,
        content_state: &'static str,
        content_base64: Option<String>,
    ) -> Self {
        Self {
            artifact_id: record.artifact_id,
            request_id: record.request_id,
            occurred_at: record.occurred_at,
            kind: record.kind,
            media_kind: record.media_kind,
            checksum: record.checksum,
            bytes: record.bytes,
            version: record.version,
            redacted: record.redacted,
            truncated: record.truncated,
            content_state,
            content_base64,
        }
    }
}

pub(in crate::api::routes::logs) fn artifact_state(record: &ArtifactRecord) -> &'static str {
    if record.corrupt {
        "corrupt"
    } else if record.missing || (record.checksum.is_none() && record.bytes == 0) {
        "missing"
    } else if record.redacted {
        "available"
    } else {
        "unavailable"
    }
}
