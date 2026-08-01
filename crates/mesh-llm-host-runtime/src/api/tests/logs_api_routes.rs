use super::logs_api_fixture::*;
use super::logs_api_seed::*;
use super::*;

#[tokio::test]
async fn request_list_applies_every_filter_and_deterministic_sort() {
    let fixture = build_log_api_fixture().await;
    let path = "/api/logs/requests?limit=10&from=2026-08-01T00%3A00%3A00Z&to=2026-08-01T00%3A00%3A59Z&route=chat&model=model-a&provider=provider-a&engine=engine-a&status=201&outcome=completed&source=durable&sort=asc";

    let response = log_api_get(fixture.state, path).await;
    let body = json_body(&response);

    assert_eq!(body["items"].as_array().expect("filtered items").len(), 1);
    assert_eq!(body["items"][0]["requestId"], COMPLETED_REQUEST_ID);
}

#[tokio::test]
async fn request_list_filters_each_dimension_independently() {
    let fixture = build_log_api_fixture().await;
    for (query, expected_id) in [
        ("route=chat", COMPLETED_REQUEST_ID),
        ("model=model-a", COMPLETED_REQUEST_ID),
        ("provider=provider-a", COMPLETED_REQUEST_ID),
        ("engine=engine-a", COMPLETED_REQUEST_ID),
        ("status=201", COMPLETED_REQUEST_ID),
        ("source=active", ACTIVE_REQUEST_ID),
        ("from=2026-08-01T00%3A00%3A25Z", ACTIVE_REQUEST_ID),
    ] {
        let response = log_api_get(
            fixture.state.clone(),
            &format!("/api/logs/requests?{query}"),
        )
        .await;
        let items = json_body(&response)["items"]
            .as_array()
            .expect("independent filter items")
            .clone();
        assert_eq!(items.len(), 1, "{query}");
        assert_eq!(items[0]["requestId"], expected_id, "{query}");
    }
    let completed = log_api_get(
        fixture.state.clone(),
        "/api/logs/requests?outcome=completed&source=durable",
    )
    .await;
    assert_eq!(json_body(&completed)["items"].as_array().unwrap().len(), 2);
    let response = log_api_get(
        fixture.state,
        "/api/logs/requests?to=2026-08-01T00%3A00%3A20Z&source=durable&sort=asc",
    )
    .await;
    let body = json_body(&response);
    assert_eq!(body["items"].as_array().unwrap().len(), 2);
    assert_eq!(body["items"][0]["requestId"], COMPLETED_REQUEST_ID);
}

#[tokio::test]
async fn request_pages_are_stable_for_same_timestamp_rows() {
    let fixture = build_log_api_fixture().await;

    let first = log_api_get(
        fixture.state.clone(),
        "/api/logs/requests?limit=1&source=durable&sort=desc",
    )
    .await;
    let first_body = json_body(&first);
    let cursor = first_body["nextCursor"].as_str().expect("first cursor");
    let second = log_api_get(
        fixture.state,
        &format!(
            "/api/logs/requests?limit=1&source=durable&sort=desc&cursor={}",
            urlencoding::encode(cursor)
        ),
    )
    .await;
    let second_body = json_body(&second);

    assert_ne!(
        first_body["items"][0]["requestId"],
        second_body["items"][0]["requestId"]
    );
}

#[tokio::test]
async fn detail_events_artifacts_and_proxy_routes_return_privacy_safe_dtos() {
    let fixture = build_log_api_fixture().await;
    let detail = log_api_get(
        fixture.state.clone(),
        &format!("/api/logs/requests/{COMPLETED_REQUEST_ID}"),
    )
    .await;
    let events = log_api_get(
        fixture.state.clone(),
        &format!("/api/logs/requests/{COMPLETED_REQUEST_ID}/events"),
    )
    .await;
    let artifacts = log_api_get(
        fixture.state.clone(),
        &format!("/api/logs/requests/{COMPLETED_REQUEST_ID}/artifacts"),
    )
    .await;
    let proxy = log_api_get(
        fixture.state,
        &format!("/api/logs/proxy?request_id={COMPLETED_REQUEST_ID}"),
    )
    .await;

    assert_eq!(json_body(&detail)["requestId"], COMPLETED_REQUEST_ID);
    assert_eq!(json_body(&events)["items"][0]["kind"], "completed");
    assert_eq!(json_body(&artifacts)["items"].as_array().unwrap().len(), 3);
    assert_eq!(
        json_body(&proxy)["items"][0]["target"],
        "http://127.0.0.1:9337"
    );
    assert!(!proxy.contains("secret"));
    assert!(!proxy.contains("/private"));
}

#[tokio::test]
async fn proxy_route_applies_provider_engine_and_status_filters() {
    let fixture = build_log_api_fixture().await;

    let response = log_api_get(
        fixture.state,
        "/api/logs/proxy?provider=provider-a&engine=engine-a&status=201",
    )
    .await;
    let body = json_body(&response);

    assert_eq!(body["items"].as_array().unwrap().len(), 1);
    assert_eq!(body["items"][0]["requestId"], COMPLETED_REQUEST_ID);
}

#[tokio::test]
async fn proxy_provider_and_engine_use_privacy_safe_metadata() {
    let fixture = build_log_api_fixture().await;

    let response = log_api_get(
        fixture.state,
        &format!("/api/logs/proxy?request_id={RETRIED_REQUEST_ID}"),
    )
    .await;
    let body = json_body(&response);
    let item = &body["items"][0];

    assert_eq!(item["provider"], "[REDACTED]");
    assert_eq!(item["engine"], "[REDACTED]");
    for secret in [
        "user:secret",
        "/private",
        "/Users",
        "token=secret",
        "api_key=secret",
    ] {
        assert!(!response.contains(secret), "leaked {secret}");
    }
}

#[tokio::test]
async fn artifact_content_exposes_only_redacted_bytes_and_typed_states() {
    let fixture = build_log_api_fixture().await;
    let redacted = log_api_get(
        fixture.state.clone(),
        &format!("/api/logs/artifacts/{REDACTED_ARTIFACT_ID}"),
    )
    .await;
    let private = log_api_get(
        fixture.state.clone(),
        &format!("/api/logs/artifacts/{PRIVATE_ARTIFACT_ID}"),
    )
    .await;
    let missing = log_api_get(
        fixture.state,
        &format!("/api/logs/artifacts/{MISSING_ARTIFACT_ID}"),
    )
    .await;

    assert_eq!(json_body(&redacted)["contentState"], "available");
    assert!(json_body(&redacted)["contentBase64"].is_string());
    assert_eq!(json_body(&private)["contentState"], "unavailable");
    assert!(json_body(&private)["contentBase64"].is_null());
    assert!(!private.contains("private-content"));
    assert_eq!(json_body(&missing)["contentState"], "missing");
}
