use super::logs_api_fixture::*;
use super::*;

#[tokio::test]
async fn malformed_log_queries_return_bounded_typed_errors() {
    let fixture = build_log_api_fixture().await;
    for path in [
        "/api/logs/requests?cursor=bad",
        "/api/logs/requests?outcome=unknown",
        "/api/logs/requests?from=2026-08-02T00%3A00%3A00Z&to=2026-08-01T00%3A00%3A00Z",
        "/api/logs/requests?limit=101",
        "/api/logs/requests?sort=random",
        "/api/logs/requests?source=memory",
        "/api/logs/proxy?status=999",
    ] {
        let response = log_api_get(fixture.state.clone(), path).await;
        let body = json_body(&response);
        assert!(response.starts_with("HTTP/1.1 400 Bad Request"), "{path}");
        assert!(body["error"]["code"].as_str().is_some(), "{path}");
        assert!(response.len() < 1024, "{path}");
    }
}

#[tokio::test]
async fn stale_request_cursor_returns_typed_expired_error() {
    let fixture = build_log_api_fixture().await;
    let cursor = mesh_llm_log_store::encode_cursor(
        "2026-08-01T00:00:00Z",
        "00000000-0000-4000-8000-000000000099",
    );

    let response = log_api_get(
        fixture.state,
        &format!("/api/logs/requests?cursor={}", urlencoding::encode(&cursor)),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 400 Bad Request"));
    assert_eq!(json_body(&response)["error"]["code"], "cursor_expired");
}

#[tokio::test]
async fn traversal_and_unknown_ids_return_typed_client_errors() {
    let fixture = build_log_api_fixture().await;

    let traversal = log_api_get(
        fixture.state.clone(),
        "/api/logs/artifacts/..%2F..%2Fsecret",
    )
    .await;
    let missing = log_api_get(
        fixture.state,
        "/api/logs/requests/00000000-0000-4000-8000-000000000099",
    )
    .await;

    assert!(traversal.starts_with("HTTP/1.1 400 Bad Request"));
    assert_eq!(json_body(&traversal)["error"]["code"], "invalid_id");
    assert!(missing.starts_with("HTTP/1.1 404 Not Found"));
    assert_eq!(json_body(&missing)["error"]["code"], "not_found");
}

#[tokio::test]
async fn hostile_headers_are_rejected_before_logging_query_access() {
    for header in [
        "Host: hostile.example\r\n",
        "Host: localhost\r\nOrigin: https://hostile.example\r\n",
    ] {
        let fixture = build_log_api_fixture().await;
        let accesses = fixture
            .state
            .inner
            .lock()
            .await
            .logging_query_accesses
            .clone();
        let (addr, server) = spawn_management_test_server(fixture.state).await;
        let response = send_management_request(
            addr,
            format!("GET /api/logs/requests HTTP/1.1\r\n{header}\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();

        assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
        assert_eq!(accesses.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}

#[test]
fn non_loopback_policy_rejects_before_query_seam() {
    let accesses = std::sync::atomic::AtomicUsize::new(0);

    let trusted = crate::api::access::is_trusted_local_request(
        Some(std::net::SocketAddr::from(([192, 0, 2, 10], 40123))),
        None,
        Some("localhost:3131"),
    );

    assert!(!trusted);
    assert_eq!(accesses.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[tokio::test]
async fn store_failure_returns_typed_service_error() {
    let fixture = build_log_api_fixture().await;
    fixture
        .artifact_store
        .store_ref()
        .conn()
        .execute("DROP TABLE summaries", [])
        .expect("break query store");

    let response = log_api_get(fixture.state, "/api/logs/requests").await;

    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
    assert_eq!(json_body(&response)["error"]["code"], "store_unavailable");
}
