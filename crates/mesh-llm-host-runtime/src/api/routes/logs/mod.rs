mod dto;
mod error;
mod parse;
mod requests;

use std::sync::Arc;

use self::error::LogsError;
use super::super::MeshApi;
use crate::logging::LoggingService;

async fn service(state: &MeshApi) -> Result<Arc<LoggingService>, LogsError> {
    let inner = state.inner.lock().await;
    #[cfg(test)]
    inner
        .logging_query_accesses
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    inner
        .logging_service
        .clone()
        .ok_or(LogsError::ServiceUnavailable)
}

async fn run_blocking<T, F>(operation: F) -> Result<T, LogsError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, LogsError> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|_| LogsError::StoreUnavailable)?
}
