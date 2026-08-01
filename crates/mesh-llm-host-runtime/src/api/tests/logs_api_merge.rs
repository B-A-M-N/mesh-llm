use super::logs_api_fixture::*;
use super::logs_api_seed::*;
use super::*;

const ACTIVE_ONLY_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000004";

async fn register_active_only_request(state: &MeshApi) -> String {
    let service = state
        .inner
        .lock()
        .await
        .logging_service
        .clone()
        .expect("logging service");
    let request_uuid = uuid::Uuid::parse_str(ACTIVE_ONLY_REQUEST_ID).expect("active-only UUID");
    service.register_request(mesh_llm_events::logging::identifiers::RequestId::from(
        request_uuid,
    ));
    service
        .registry_ref()
        .get_active(ACTIVE_ONLY_REQUEST_ID)
        .expect("active-only request")
        .created_at
}

#[tokio::test]
async fn request_list_merges_active_and_durable_without_duplicates() {
    let fixture = build_log_api_fixture().await;

    let response = log_api_get(fixture.state, "/api/logs/requests?limit=10").await;
    let body = json_body(&response);
    let items = body["items"].as_array().expect("request items");
    let ids = items
        .iter()
        .filter_map(|item| item["requestId"].as_str())
        .collect::<Vec<_>>();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(ids.len(), 3);
    assert_eq!(ids.iter().filter(|id| **id == ACTIVE_REQUEST_ID).count(), 1);
    assert_eq!(items[0]["source"], "active");
    assert!(!response.contains("/Users/private"));
    assert!(!response.contains("token=secret"));
}

#[tokio::test]
async fn metadata_filtered_active_and_durable_request_is_active_once() {
    let fixture = build_log_api_fixture().await;

    let response = log_api_get(fixture.state, "/api/logs/requests?route=responses").await;
    let body = json_body(&response);
    let matching = body["items"]
        .as_array()
        .expect("route-filtered request items")
        .iter()
        .filter(|item| item["requestId"] == ACTIVE_REQUEST_ID)
        .collect::<Vec<_>>();

    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0]["source"], "active");
}

#[tokio::test]
async fn active_source_metadata_filter_uses_durable_metadata_once() {
    let fixture = build_log_api_fixture().await;

    let response = log_api_get(
        fixture.state,
        "/api/logs/requests?source=active&route=responses",
    )
    .await;
    let body = json_body(&response);
    let matching = body["items"]
        .as_array()
        .expect("active route-filtered request items")
        .iter()
        .filter(|item| item["requestId"] == ACTIVE_REQUEST_ID)
        .collect::<Vec<_>>();

    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0]["source"], "active");
}

#[tokio::test]
async fn active_only_request_survives_active_time_and_outcome_filters() {
    let fixture = build_log_api_fixture().await;
    let created_at = register_active_only_request(&fixture.state).await;

    let response = log_api_get(
        fixture.state,
        &format!(
            "/api/logs/requests?source=active&outcome=active&from={}",
            urlencoding::encode(&created_at)
        ),
    )
    .await;
    let body = json_body(&response);
    let matching = body["items"]
        .as_array()
        .expect("active-only filtered items")
        .iter()
        .filter(|item| item["requestId"] == ACTIVE_ONLY_REQUEST_ID)
        .collect::<Vec<_>>();

    assert_eq!(matching.len(), 1);
    assert_eq!(matching[0]["source"], "active");
}

#[tokio::test]
async fn active_only_request_does_not_match_missing_durable_metadata() {
    let fixture = build_log_api_fixture().await;
    register_active_only_request(&fixture.state).await;

    let response = log_api_get(
        fixture.state,
        "/api/logs/requests?source=active&route=responses",
    )
    .await;
    let body = json_body(&response);
    let active_only_count = body["items"]
        .as_array()
        .expect("metadata-filtered active items")
        .iter()
        .filter(|item| item["requestId"] == ACTIVE_ONLY_REQUEST_ID)
        .count();

    assert_eq!(active_only_count, 0);
}
