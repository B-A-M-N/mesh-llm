use futures_util::SinkExt;
use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;

use super::logs_api_fixture::{
    build_log_api_fixture, build_log_api_fixture_with_service_config, log_api_get,
};
use super::logs_websocket_support::*;
use super::*;

#[tokio::test]
async fn websocket_replays_only_the_subscribed_channel_in_order() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    publish_request_event(&service, 1);
    publish_request_event(&service, 2);
    service
        .enqueue_event(
            active_request_id(),
            ReplayChannel::Operations,
            serde_json::json!({ "sequence": 3 }).to_string(),
        )
        .expect("bounded replay bus accepts operation event");

    let (mut socket, server) = connect_log_stream(fixture.state).await;
    subscribe(&mut socket, "requests", 0).await;

    let first = receive_frame(&mut socket).await;
    let second = receive_frame(&mut socket).await;

    assert_eq!(first["type"], "event");
    assert_eq!(first["channel"], "requests");
    assert_eq!(first["sequence"], 1);
    assert_eq!(second["sequence"], 2);
    close_socket(socket, server).await;
}

#[tokio::test]
async fn websocket_reconnects_from_a_cursor_without_duplicate_replay() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    publish_request_event(&service, 1);
    publish_request_event(&service, 2);

    let (mut first_socket, first_server) = connect_log_stream(fixture.state.clone()).await;
    subscribe(&mut first_socket, "requests", 0).await;
    assert_eq!(receive_frame(&mut first_socket).await["sequence"], 1);
    assert_eq!(receive_frame(&mut first_socket).await["sequence"], 2);
    close_socket(first_socket, first_server).await;

    publish_request_event(&service, 3);
    let (mut second_socket, second_server) = connect_log_stream(fixture.state).await;
    subscribe(&mut second_socket, "requests", 2).await;

    let replay = receive_frame(&mut second_socket).await;
    assert_eq!(replay["type"], "event");
    assert_eq!(replay["sequence"], 3);
    close_socket(second_socket, second_server).await;
}

#[tokio::test]
async fn websocket_honors_unsubscribe_and_explicit_replay_cursor_messages() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    publish_request_event(&service, 1);
    publish_request_event(&service, 2);
    publish_request_event(&service, 3);

    let (mut socket, server) = connect_log_stream(fixture.state).await;
    subscribe(&mut socket, "requests", 1).await;
    assert_eq!(receive_frame(&mut socket).await["sequence"], 2);
    assert_eq!(receive_frame(&mut socket).await["sequence"], 3);

    socket
        .send(Message::Text(
            serde_json::json!({ "type": "replay_cursor", "channel": "requests", "cursor": 2 })
                .to_string()
                .into(),
        ))
        .await
        .expect("send replay cursor");
    assert_eq!(receive_frame(&mut socket).await["sequence"], 3);

    socket
        .send(Message::Text(
            serde_json::json!({ "type": "unsubscribe", "channel": "requests" })
                .to_string()
                .into(),
        ))
        .await
        .expect("send unsubscribe");
    socket
        .send(Message::Text(
            serde_json::json!({ "type": "replay_cursor", "channel": "requests", "cursor": 3 })
                .to_string()
                .into(),
        ))
        .await
        .expect("send replay cursor after unsubscribe");
    let error = receive_frame(&mut socket).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["code"], "not_subscribed");
    close_socket(socket, server).await;
}

#[tokio::test]
async fn websocket_reports_a_gap_with_a_usable_rest_recovery_cursor_after_eviction() {
    let fixture = build_log_api_fixture_with_service_config(crate::logging::ServiceConfig {
        queue_capacity: 2,
        registry_config: crate::logging::RegistryConfig::default(),
    })
    .await;
    let service = log_service(&fixture.state).await;
    publish_request_event(&service, 1);
    publish_request_event(&service, 2);
    publish_request_event(&service, 3);

    let (mut socket, server) = connect_log_stream(fixture.state.clone()).await;
    subscribe(&mut socket, "requests", 0).await;

    let gap = receive_frame(&mut socket).await;
    assert_eq!(gap["type"], "gap");
    assert_eq!(gap["channel"], "requests");
    assert_eq!(gap["fromSequence"], 1);
    assert_eq!(gap["toSequence"], 1);
    assert_eq!(gap["recovery"]["endpoint"], "/api/logs/requests");
    let cursor = gap["recovery"]["cursor"]
        .as_str()
        .expect("gap provides an opaque REST cursor");
    assert!(cursor.starts_with("v1:"));
    assert_eq!(receive_frame(&mut socket).await["sequence"], 2);
    assert_eq!(receive_frame(&mut socket).await["sequence"], 3);
    close_socket(socket, server).await;

    let recovery = log_api_get(
        fixture.state,
        &format!("/api/logs/requests?cursor={}", urlencoding::encode(cursor)),
    )
    .await;
    assert!(recovery.starts_with("HTTP/1.1 200 OK"));
}

#[tokio::test(start_paused = true)]
async fn websocket_emits_a_heartbeat_without_subscription_activity() {
    let fixture = build_log_api_fixture().await;
    let (mut socket, server) = connect_log_stream(fixture.state).await;

    tokio::task::yield_now().await;
    tokio::time::advance(HEARTBEAT_INTERVAL).await;

    let heartbeat = receive_frame(&mut socket).await;
    assert_eq!(heartbeat["type"], "heartbeat");
    close_socket(socket, server).await;
}

#[tokio::test]
async fn websocket_rejects_invalid_binary_and_malformed_client_messages() {
    let fixture = build_log_api_fixture().await;
    let (mut binary_socket, binary_server) = connect_log_stream(fixture.state.clone()).await;
    binary_socket
        .send(Message::Binary(vec![1, 2, 3].into()))
        .await
        .expect("send binary frame");
    let binary_error = receive_frame(&mut binary_socket).await;
    assert_eq!(binary_error["type"], "error");
    assert_eq!(binary_error["code"], "unsupported_frame");
    close_socket(binary_socket, binary_server).await;

    let (mut malformed_socket, malformed_server) = connect_log_stream(fixture.state.clone()).await;
    malformed_socket
        .send(Message::Text("{not-json".into()))
        .await
        .expect("send malformed frame");
    let malformed_error = receive_frame(&mut malformed_socket).await;
    assert_eq!(malformed_error["type"], "error");
    assert_eq!(malformed_error["code"], "invalid_message");
    close_socket(malformed_socket, malformed_server).await;

    let (mut oversized_socket, oversized_server) = connect_log_stream(fixture.state).await;
    oversized_socket
        .send(Message::Text("x".repeat(4 * 1024 + 1).into()))
        .await
        .expect("send oversized frame");
    let oversized_error = receive_frame(&mut oversized_socket).await;
    assert_eq!(oversized_error["type"], "error");
    assert_eq!(oversized_error["code"], "message_too_large");
    close_socket(oversized_socket, oversized_server).await;
}

#[tokio::test]
async fn websocket_rejects_hostile_and_invalid_upgrade_headers_before_service_access() {
    for request in [
        upgrade_request("hostile.example", None, "dGhlIHNhbXBsZSBub25jZQ=="),
        upgrade_request(
            "localhost",
            Some("https://hostile.example"),
            "dGhlIHNhbXBsZSBub25jZQ==",
        ),
        upgrade_request("localhost", None, "not-base64"),
        format!("GET {LOG_STREAM_PATH} HTTP/1.1\r\nHost: localhost\r\nConnection: Upgrade\r\n\r\n"),
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

        let response = send_management_request(addr, request).await;
        server
            .await
            .expect("management server task joins")
            .expect("management server exits cleanly");

        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request")
                || response.starts_with("HTTP/1.1 403 Forbidden")
        );
        assert_eq!(accesses.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}

#[tokio::test]
async fn websocket_accepts_the_rfc6455_upgrade_headers() {
    let fixture = build_log_api_fixture().await;
    let (addr, server) = spawn_management_test_server(fixture.state).await;
    let response = send_management_request(
        addr,
        upgrade_request("localhost", None, "dGhlIHNhbXBsZSBub25jZQ=="),
    )
    .await;
    server
        .await
        .expect("management server task joins")
        .expect("management server exits cleanly");

    assert!(
        response.starts_with("HTTP/1.1 101 Switching Protocols"),
        "{response}"
    );
}

#[tokio::test]
async fn websocket_replay_backlog_does_not_block_concurrent_management_requests() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    for sequence in 0..512 {
        publish_large_request_event(&service, sequence);
    }

    let (addr, server) = spawn_two_connection_management_server(fixture.state).await;
    let stream = TcpStream::connect(addr)
        .await
        .expect("connect slow websocket client");
    let (mut socket, response) =
        tokio_tungstenite::client_async(format!("ws://{addr}{LOG_STREAM_PATH}"), stream)
            .await
            .expect("upgrade slow websocket client");
    assert_eq!(response.status(), ::http::StatusCode::SWITCHING_PROTOCOLS);
    subscribe(&mut socket, "requests", 0).await;

    let status = send_management_request(
        addr,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    assert!(status.starts_with("HTTP/1.1 200 OK"));

    drop(socket);
    tokio::time::timeout(SLOW_CONSUMER_TIMEOUT, server)
        .await
        .expect("raw websocket handler exits within the bound")
        .expect("management server joins")
        .expect("management server handler exits cleanly");
}

#[tokio::test]
async fn websocket_close_cancels_the_handler_without_leaving_a_connection_task() {
    let fixture = build_log_api_fixture().await;
    let (socket, server) = connect_log_stream(fixture.state).await;

    close_socket(socket, server).await;
}
