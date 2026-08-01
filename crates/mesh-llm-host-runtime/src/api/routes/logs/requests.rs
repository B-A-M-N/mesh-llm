use std::collections::HashSet;

use mesh_llm_log_store::{QuerySort, RequestRecord};
use tokio::net::TcpStream;

use super::dto::{PageDto, RequestDto};
use super::error::LogsError;
use super::parse::{self, SourceFilter};
use super::{run_blocking, service};
use crate::api::{MeshApi, http::respond_json};
use crate::logging::RequestSummaryEntry;

pub(super) async fn list(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> Result<(), LogsError> {
    let parsed = parse::request_query(path)?;
    let logging = service(state).await?;
    let artifact_store = logging.log_store_ref().ok_or(LogsError::StoreUnavailable)?;
    let active = logging.registry_ref().snapshot_active();
    let response = run_blocking(move || {
        if let Some((_, cursor_id)) = &parsed.cursor_boundary {
            let active_anchor = active.iter().any(|entry| &entry.request_id == cursor_id);
            let durable_anchor = artifact_store
                .store_ref()
                .query_request(cursor_id)?
                .is_some();
            if !active_anchor && !durable_anchor {
                return Err(LogsError::CursorExpired);
            }
        }
        let mut items = Vec::new();
        let active_ids = active
            .iter()
            .map(|entry| entry.request_id.clone())
            .collect::<HashSet<_>>();
        if parsed.source != Some(SourceFilter::Durable) {
            for entry in &active {
                let metadata = artifact_store
                    .store_ref()
                    .query_request(&entry.request_id)?;
                if active_matches(entry, metadata.as_ref(), &parsed) {
                    items.push(RequestDto::active(entry.clone(), metadata));
                }
            }
        }
        if parsed.source != Some(SourceFilter::Active) {
            let mut store_query = parsed.store.clone();
            store_query.limit = store_query
                .limit
                .saturating_add(active.len())
                .saturating_add(1);
            let durable = artifact_store.store_ref().query_requests(&store_query)?;
            items.extend(
                durable
                    .items
                    .into_iter()
                    .filter(|record| !active_ids.contains(&record.request_id))
                    .map(RequestDto::durable),
            );
        }
        items.sort_by(|left, right| {
            left.created_at()
                .cmp(right.created_at())
                .then_with(|| left.request_id().cmp(right.request_id()))
        });
        if parsed.store.sort == QuerySort::Descending {
            items.reverse();
        }
        let has_more = items.len() > parsed.store.limit;
        items.truncate(parsed.store.limit);
        let next_cursor = if has_more {
            items
                .last()
                .map(|item| mesh_llm_log_store::encode_cursor(item.created_at(), item.request_id()))
        } else {
            None
        };
        Ok(PageDto { items, next_cursor })
    })
    .await?;
    respond_json(stream, 200, &response)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

pub(super) async fn detail(
    stream: &mut TcpStream,
    state: &MeshApi,
    request_id: &str,
) -> Result<(), LogsError> {
    let request_id = parse::id(request_id)?;
    let logging = service(state).await?;
    if let Some(active) = logging.registry_ref().get_active(&request_id) {
        return respond_json(stream, 200, &RequestDto::active(active, None))
            .await
            .map_err(|_| LogsError::StoreUnavailable);
    }
    let artifact_store = logging.log_store_ref().ok_or(LogsError::StoreUnavailable)?;
    let record = run_blocking(move || {
        artifact_store
            .store_ref()
            .query_request(&request_id)?
            .ok_or(LogsError::NotFound)
    })
    .await?;
    respond_json(stream, 200, &RequestDto::durable(record))
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

fn active_matches(
    entry: &RequestSummaryEntry,
    metadata: Option<&RequestRecord>,
    parsed: &parse::RequestListQuery,
) -> bool {
    let query = &parsed.store;
    if let Some(route) = &query.route
        && metadata.and_then(|record| record.route.as_ref()) != Some(route)
    {
        return false;
    }
    if let Some(model) = &query.model
        && metadata.and_then(|record| record.model.as_ref()) != Some(model)
    {
        return false;
    }
    if let Some(provider) = &query.provider
        && metadata.and_then(|record| record.provider.as_ref()) != Some(provider)
    {
        return false;
    }
    if let Some(engine) = &query.engine
        && metadata.and_then(|record| record.engine.as_ref()) != Some(engine)
    {
        return false;
    }
    if let Some(status_code) = query.status_code
        && metadata.and_then(|record| record.status_code) != Some(i64::from(status_code))
    {
        return false;
    }
    if let Some(outcome) = query.outcome
        && entry.state != outcome.as_str()
    {
        return false;
    }
    if query
        .from
        .as_ref()
        .is_some_and(|from| entry.created_at < *from)
        || query.to.as_ref().is_some_and(|to| entry.created_at > *to)
    {
        return false;
    }
    if let Some((timestamp, request_id)) = &parsed.cursor_boundary {
        let after = (entry.created_at.as_str(), entry.request_id.as_str());
        let cursor = (timestamp.as_str(), request_id.as_str());
        return match query.sort {
            QuerySort::Ascending => after > cursor,
            QuerySort::Descending => after < cursor,
        };
    }
    true
}
