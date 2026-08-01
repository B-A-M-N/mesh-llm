use super::*;

#[tokio::test]
async fn existing_status_route_remains_network_readable() {
    let state = build_test_mesh_api().await;
    let (addr, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: remote.example\r\n\r\n".to_string(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn log_route_rejects_hostile_host_before_dispatch() {
    let state = build_test_mesh_api().await;
    let (addr, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/logs/requests HTTP/1.1\r\nHost: hostile.example\r\n\r\n".to_string(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 403 Forbidden"));
    server.await.unwrap().unwrap();
}

#[tokio::test]
async fn request_list_reports_unavailable_when_logging_is_disabled() {
    let state = build_test_mesh_api().await;
    let (addr, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /api/logs/requests HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 503 Service Unavailable"));
    server.await.unwrap().unwrap();
}
