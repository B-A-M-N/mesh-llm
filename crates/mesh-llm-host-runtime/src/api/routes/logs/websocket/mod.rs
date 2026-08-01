mod handshake;
mod protocol;
mod session;
#[cfg(test)]
mod session_tests;
mod subscriptions;

use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::protocol::{Role, WebSocketConfig};

use super::super::super::MeshApi;
use super::error::LogsError;
use super::{LogsRequest, service};

const MAX_FRAME_BYTES: usize = 16 * 1024;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    request: LogsRequest<'_>,
) -> anyhow::Result<()> {
    if request.method != "GET" || !request.body.is_empty() {
        return LogsError::MethodNotAllowed.write(stream).await;
    }

    let handshake = match handshake::validate(request.raw_request) {
        Ok(handshake) => handshake,
        Err(_) => {
            crate::api::http::respond_error(stream, 400, "Invalid WebSocket upgrade").await?;
            return Ok(());
        }
    };
    let logging = match service(state).await {
        Ok(logging) => logging,
        Err(error) => return error.write(stream).await,
    };

    handshake.write_response(stream).await?;
    let socket =
        WebSocketStream::from_raw_socket(stream, Role::Server, Some(socket_config())).await;
    session::run(socket, logging).await
}

fn socket_config() -> WebSocketConfig {
    WebSocketConfig::default()
        .max_frame_size(Some(MAX_FRAME_BYTES))
        .max_message_size(Some(MAX_FRAME_BYTES))
}
