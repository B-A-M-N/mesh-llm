mod dto;
mod error;
mod parse;
mod requests;

use std::sync::Arc;

use tokio::net::TcpStream;

use self::error::LogsError;
use super::super::MeshApi;
use crate::logging::LoggingService;

pub(super) fn is_route(path: &str) -> bool {
    path == "/api/logs" || path.starts_with("/api/logs/")
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    request: LogsRequest<'_>,
) -> anyhow::Result<()> {
    if request.method != "GET" || !request.body.is_empty() {
        return LogsError::MethodNotAllowed.write(stream).await;
    }
    let result = match classify(request.path) {
        Route::Requests => requests::list(stream, state, request.path).await,
        Route::RequestDetail(request_id) => requests::detail(stream, state, &request_id).await,
        Route::Unknown => Err(LogsError::NotFound),
    };
    match result {
        Ok(()) => Ok(()),
        Err(error) => error.write(stream).await,
    }
}

pub(super) struct LogsRequest<'a> {
    pub(super) method: &'a str,
    pub(super) path: &'a str,
    pub(super) body: &'a str,
    pub(super) raw_request: &'a [u8],
}

enum Route {
    Requests,
    RequestDetail(String),
    Unknown,
}

fn classify(path: &str) -> Route {
    let path = path.split('?').next().unwrap_or(path);
    if path == "/api/logs/requests" {
        return Route::Requests;
    }
    let Some(remainder) = path.strip_prefix("/api/logs/requests/") else {
        return Route::Unknown;
    };
    if !remainder.contains('/') {
        return Route::RequestDetail(remainder.to_string());
    }
    Route::Unknown
}

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
