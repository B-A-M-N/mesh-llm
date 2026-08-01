use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use mesh_llm_events::logging::identifiers::RequestId;
use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinSet;
use tokio_tungstenite::tungstenite::Message;

use super::logs_api_seed::ACTIVE_REQUEST_ID;
use super::*;

pub(super) const LOG_STREAM_PATH: &str = "/api/logs/stream";
pub(super) const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
pub(super) const SLOW_CONSUMER_TIMEOUT: Duration = Duration::from_secs(2);

pub(super) type TestWebSocket = tokio_tungstenite::WebSocketStream<TcpStream>;
pub(super) type ServerTask = tokio::task::JoinHandle<anyhow::Result<()>>;

pub(super) async fn connect_log_stream(state: MeshApi) -> (TestWebSocket, ServerTask) {
    let (addr, server) = spawn_management_test_server(state).await;
    (upgrade_log_stream(addr).await, server)
}

pub(super) async fn upgrade_log_stream(addr: std::net::SocketAddr) -> TestWebSocket {
    let stream = TcpStream::connect(addr)
        .await
        .expect("connect websocket client");
    let (socket, response) =
        tokio_tungstenite::client_async(format!("ws://{addr}{LOG_STREAM_PATH}"), stream)
            .await
            .expect("upgrade logs websocket");
    assert_eq!(response.status(), ::http::StatusCode::SWITCHING_PROTOCOLS);
    socket
}

pub(super) async fn log_service(state: &MeshApi) -> Arc<crate::logging::LoggingService> {
    state
        .inner
        .lock()
        .await
        .logging_service
        .clone()
        .expect("logging test fixture provides a service")
}

pub(super) fn active_request_id() -> RequestId {
    let id = uuid::Uuid::parse_str(ACTIVE_REQUEST_ID).expect("valid active request id");
    RequestId::from(id)
}

pub(super) fn publish_request_event(service: &crate::logging::LoggingService, sequence: u64) {
    service
        .enqueue_event(
            active_request_id(),
            ReplayChannel::Requests,
            serde_json::json!({ "sequence": sequence }).to_string(),
        )
        .expect("bounded replay bus accepts test event");
}

pub(super) fn publish_large_request_event(service: &crate::logging::LoggingService, sequence: u64) {
    service
        .enqueue_event(
            active_request_id(),
            ReplayChannel::Requests,
            serde_json::json!({ "sequence": sequence, "detail": "x".repeat(8 * 1024) }).to_string(),
        )
        .expect("bounded replay bus accepts test event");
}

pub(super) async fn subscribe(socket: &mut TestWebSocket, channel: &str, cursor: u64) {
    socket
        .send(Message::Text(
            serde_json::json!({ "type": "subscribe", "channel": channel, "cursor": cursor })
                .to_string()
                .into(),
        ))
        .await
        .expect("send subscription");
}

pub(super) async fn receive_frame(socket: &mut TestWebSocket) -> serde_json::Value {
    let text = receive_text_frame(socket).await;
    serde_json::from_str(&text).expect("server frame is valid JSON")
}

pub(super) async fn receive_text_frame(socket: &mut TestWebSocket) -> String {
    let received = tokio::time::timeout(Duration::from_secs(1), socket.next())
        .await
        .expect("server frame arrives within bound")
        .expect("websocket stays open")
        .expect("websocket frame succeeds");
    let Message::Text(text) = received else {
        panic!("server frame is JSON text");
    };
    text.to_string()
}

pub(super) async fn close_socket(socket: TestWebSocket, server: ServerTask) {
    drop(socket);
    let joined = tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("websocket handler cancels within bound")
        .expect("websocket handler task joins");
    joined.expect("websocket handler exits cleanly");
}

pub(super) fn upgrade_request(host: &str, origin: Option<&str>, key: &str) -> String {
    let origin = origin.map_or_else(String::new, |value| format!("Origin: {value}\r\n"));
    format!(
        "GET {LOG_STREAM_PATH} HTTP/1.1\r\nHost: {host}\r\n{origin}Connection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: {key}\r\n\r\n"
    )
}

pub(super) async fn spawn_two_connection_management_server(
    state: MeshApi,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<anyhow::Result<()>>,
) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind management test server");
    let addr = listener.local_addr().expect("read management test address");
    let task = tokio::spawn(async move {
        let mut handlers = JoinSet::new();
        for _ in 0..2 {
            let (stream, _) = listener.accept().await.expect("accept test connection");
            let state = state.clone();
            handlers.spawn(async move { crate::api::server::handle_request(stream, &state).await });
        }
        while let Some(result) = handlers.join_next().await {
            result??;
        }
        Ok(())
    });
    (addr, task)
}
