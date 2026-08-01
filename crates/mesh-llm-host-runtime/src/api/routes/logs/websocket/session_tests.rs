use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::SinkExt;
use mesh_llm_events::logging::identifiers::RequestId;
use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::oneshot;
use tokio::time::{Duration, advance, timeout};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::Role;

use super::session::{WRITE_TIMEOUT, run};

struct StalledWriter<S> {
    inner: S,
    write_started: Option<oneshot::Sender<()>>,
}

impl<S: AsyncRead + Unpin> AsyncRead for StalledWriter<S> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.inner).poll_read(context, buffer)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for StalledWriter<S> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _: &mut Context<'_>,
        _: &[u8],
    ) -> Poll<io::Result<usize>> {
        if let Some(write_started) = self.write_started.take() {
            let _ = write_started.send(());
        }
        Poll::Pending
    }

    fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Pending
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Pending
    }
}

#[tokio::test(start_paused = true)]
async fn writer_timeout_cancels_the_session_while_the_peer_remains_open() {
    let service = Arc::new(crate::logging::LoggingService::new_disabled(
        crate::logging::ServiceConfig::default(),
    ));
    service
        .enqueue_event(
            RequestId::from(uuid::Uuid::new_v4()),
            ReplayChannel::Requests,
            serde_json::json!({ "event": "ready" }).to_string(),
        )
        .expect("test event is queued");

    let (server_io, client_io) = tokio::io::duplex(1024);
    let (write_started, write_started_rx) = oneshot::channel();
    let socket = WebSocketStream::from_raw_socket(
        StalledWriter {
            inner: server_io,
            write_started: Some(write_started),
        },
        Role::Server,
        None,
    )
    .await;
    let mut client = WebSocketStream::from_raw_socket(client_io, Role::Client, None).await;
    let session = tokio::spawn(run(socket, service));

    client
        .send(Message::Text(
            serde_json::json!({ "type": "subscribe", "channel": "requests", "cursor": 0 })
                .to_string()
                .into(),
        ))
        .await
        .expect("peer stays connected long enough to subscribe");
    write_started_rx
        .await
        .expect("writer starts the timed outbound frame while peer remains open");

    advance(WRITE_TIMEOUT).await;
    tokio::task::yield_now().await;
    advance(WRITE_TIMEOUT).await;
    tokio::task::yield_now().await;
    timeout(Duration::from_secs(1), session)
        .await
        .expect("writer timeout cancels the joined session within the bound")
        .expect("session task joins")
        .expect("session returns cleanly after writer timeout");

    drop(client);
}
