use mesh_llm_log_store::LogStoreError;
use tokio::net::TcpStream;

use super::dto::{ArtifactDto, EventDto, PageDto, artifact_state};
use super::error::LogsError;
use super::parse;
use super::{run_blocking, service};
use crate::api::{MeshApi, http::respond_json};

pub(super) async fn events(
    stream: &mut TcpStream,
    state: &MeshApi,
    request: RelatedRequest<'_>,
) -> Result<(), LogsError> {
    let request_id = parse::id(request.request_id)?;
    let query = parse::page_query(request.path)?;
    let logging = service(state).await?;
    let artifact_store = logging.log_store_ref().ok_or(LogsError::StoreUnavailable)?;
    let page = run_blocking(move || {
        let store = artifact_store.store_ref();
        if store.query_request(&request_id)?.is_none() {
            return Err(LogsError::NotFound);
        }
        store.query_events(&request_id, &query).map_err(Into::into)
    })
    .await?;
    let items = page
        .items
        .into_iter()
        .map(EventDto::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    respond_json(
        stream,
        200,
        &PageDto {
            items,
            next_cursor: page.next_cursor,
        },
    )
    .await
    .map_err(|_| LogsError::StoreUnavailable)
}

pub(super) async fn artifacts(
    stream: &mut TcpStream,
    state: &MeshApi,
    request: RelatedRequest<'_>,
) -> Result<(), LogsError> {
    let request_id = parse::id(request.request_id)?;
    let query = parse::page_query(request.path)?;
    let logging = service(state).await?;
    let artifact_store = logging.log_store_ref().ok_or(LogsError::StoreUnavailable)?;
    let page = run_blocking(move || {
        let store = artifact_store.store_ref();
        if store.query_request(&request_id)?.is_none() {
            return Err(LogsError::NotFound);
        }
        store
            .query_artifacts(&request_id, &query)
            .map_err(Into::into)
    })
    .await?;
    let response = PageDto {
        items: page.items.into_iter().map(ArtifactDto::metadata).collect(),
        next_cursor: page.next_cursor,
    };
    respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

pub(super) struct RelatedRequest<'a> {
    path: &'a str,
    request_id: &'a str,
}

impl<'a> RelatedRequest<'a> {
    pub(super) const fn new(path: &'a str, request_id: &'a str) -> Self {
        Self { path, request_id }
    }
}

pub(super) async fn artifact_content(
    stream: &mut TcpStream,
    state: &MeshApi,
    artifact_id: &str,
) -> Result<(), LogsError> {
    let artifact_id = parse::id(artifact_id)?;
    let logging = service(state).await?;
    let artifact_store = logging.log_store_ref().ok_or(LogsError::StoreUnavailable)?;
    let response = run_blocking(move || {
        let record = artifact_store
            .store_ref()
            .query_artifact(&artifact_id)?
            .ok_or(LogsError::NotFound)?;
        if artifact_state(&record) != "available" {
            return Ok(ArtifactDto::metadata(record));
        }
        match artifact_store.read_artifact(&artifact_id) {
            Ok(content) => Ok(ArtifactDto::content(record, content)),
            Err(LogStoreError::ArtifactMissing { .. }) => {
                Ok(ArtifactDto::metadata(mesh_llm_log_store::ArtifactRecord {
                    missing: true,
                    ..record
                }))
            }
            Err(LogStoreError::ArtifactCorrupt { .. }) => {
                Ok(ArtifactDto::metadata(mesh_llm_log_store::ArtifactRecord {
                    corrupt: true,
                    ..record
                }))
            }
            Err(error) => Err(error.into()),
        }
    })
    .await?;
    respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}
