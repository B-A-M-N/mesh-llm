use serde::Serialize;
use tokio::net::TcpStream;

use super::super::super::http::respond_json;

#[derive(Debug)]
pub(super) enum LogsError {
    InvalidQuery(&'static str),
    InvalidCursor,
    CursorExpired,
    InvalidId,
    NotFound,
    MethodNotAllowed,
    ServiceUnavailable,
    StoreUnavailable,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
    message: &'static str,
}

impl LogsError {
    pub(super) async fn write(self, stream: &mut TcpStream) -> anyhow::Result<()> {
        let (status, code, message) = match self {
            Self::InvalidQuery(message) => (400, "invalid_query", message),
            Self::InvalidCursor => (400, "invalid_cursor", "cursor is malformed"),
            Self::CursorExpired => (400, "cursor_expired", "cursor is no longer available"),
            Self::InvalidId => (400, "invalid_id", "identifier must be a UUID"),
            Self::NotFound => (404, "not_found", "log record was not found"),
            Self::MethodNotAllowed => (405, "method_not_allowed", "route requires GET"),
            Self::ServiceUnavailable => (
                503,
                "logging_unavailable",
                "logging service is not available",
            ),
            Self::StoreUnavailable => (503, "store_unavailable", "logging store is not available"),
        };
        respond_json(
            stream,
            status,
            &ErrorResponse {
                error: ErrorBody { code, message },
            },
        )
        .await
    }
}

impl From<mesh_llm_log_store::LogStoreError> for LogsError {
    fn from(error: mesh_llm_log_store::LogStoreError) -> Self {
        match error {
            mesh_llm_log_store::LogStoreError::CursorMalformed(_) => Self::InvalidCursor,
            mesh_llm_log_store::LogStoreError::PathUnsafe { .. } => Self::InvalidId,
            mesh_llm_log_store::LogStoreError::ArtifactMissing { .. }
            | mesh_llm_log_store::LogStoreError::ArtifactCorrupt { .. } => Self::NotFound,
            mesh_llm_log_store::LogStoreError::Sqlite(_)
            | mesh_llm_log_store::LogStoreError::MigrationFailed(_)
            | mesh_llm_log_store::LogStoreError::InsertFailed(_)
            | mesh_llm_log_store::LogStoreError::DuplicateTerminalEvent { .. }
            | mesh_llm_log_store::LogStoreError::AlreadyExists { .. }
            | mesh_llm_log_store::LogStoreError::QueryFailed(_)
            | mesh_llm_log_store::LogStoreError::IoError(_)
            | mesh_llm_log_store::LogStoreError::ArtifactLimitExceeded { .. }
            | mesh_llm_log_store::LogStoreError::PrivacyNotGuaranteed => Self::StoreUnavailable,
        }
    }
}
