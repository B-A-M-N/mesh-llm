use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tokio::time::{MissedTickBehavior, timeout};
use tokio_tungstenite::WebSocketStream;
use tokio_tungstenite::tungstenite::Message;

use super::protocol::{
    ClientMessage, ErrorCode, MAX_SERVER_FRAME_BYTES, RestRecovery, ServerMessage,
    WebSocketChannel, event_from_entry, parse_client, request_id,
};
use super::subscriptions::Subscriptions;
use crate::logging::{BusEntry, LoggingService};

const CONNECTION_QUEUE_CAPACITY: usize = 16;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
pub(super) const WRITE_TIMEOUT: Duration = Duration::from_millis(250);

pub(super) async fn run<S>(
    socket: WebSocketStream<S>,
    logging: Arc<LoggingService>,
) -> anyhow::Result<()>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (sink, source) = socket.split();
    let (outbound, receiver) = mpsc::channel(CONNECTION_QUEUE_CAPACITY);
    let (writer_done, writer_done_rx) = oneshot::channel();

    tokio::join!(
        read_frames(source, logging, outbound, writer_done_rx),
        write_frames(sink, receiver, writer_done),
    );
    Ok(())
}

async fn read_frames<S>(
    mut source: futures_util::stream::SplitStream<WebSocketStream<S>>,
    logging: Arc<LoggingService>,
    outbound: mpsc::Sender<String>,
    mut writer_done: oneshot::Receiver<()>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let bus = logging.bus_ref();
    let mut updates = bus.subscribe_updates();
    let mut subscriptions = Subscriptions::default();
    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(MissedTickBehavior::Skip);
    heartbeat.tick().await;

    loop {
        let keep_running = tokio::select! {
            message = source.next() => handle_incoming(
                message,
                &logging,
                &outbound,
                &mut subscriptions,
            ).await,
            update = updates.recv() => match update {
                Ok(()) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    replay_subscriptions(&logging, &outbound, &mut subscriptions).await
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => false,
            },
            _ = heartbeat.tick() => enqueue_heartbeat(&outbound, &subscriptions),
            _ = &mut writer_done => false,
        };
        if !keep_running {
            break;
        }
    }
}

async fn handle_incoming(
    message: Option<Result<Message, tokio_tungstenite::tungstenite::Error>>,
    logging: &Arc<LoggingService>,
    outbound: &mpsc::Sender<String>,
    subscriptions: &mut Subscriptions,
) -> bool {
    let Some(message) = message else {
        return false;
    };
    let message = match message {
        Ok(message) => message,
        Err(_) => return false,
    };
    match message {
        Message::Text(text) => match parse_client(&text) {
            Ok(client) => apply_client_message(client, logging, outbound, subscriptions).await,
            Err(code) => enqueue_error(outbound, code),
        },
        Message::Binary(_) | Message::Frame(_) => {
            enqueue_error(outbound, ErrorCode::UnsupportedFrame)
        }
        Message::Close(_) => false,
        Message::Ping(_) | Message::Pong(_) => true,
    }
}

async fn apply_client_message(
    message: ClientMessage,
    logging: &Arc<LoggingService>,
    outbound: &mpsc::Sender<String>,
    subscriptions: &mut Subscriptions,
) -> bool {
    match message {
        ClientMessage::Subscribe { channel, cursor } => {
            subscriptions.subscribe(channel, cursor.unwrap_or_default());
            replay_channel(logging, outbound, subscriptions, channel).await
        }
        ClientMessage::Unsubscribe { channel } => {
            subscriptions.unsubscribe(channel);
            true
        }
        ClientMessage::ReplayCursor { channel, cursor } => {
            if !subscriptions.contains(channel) {
                return enqueue_error(outbound, ErrorCode::NotSubscribed);
            }
            subscriptions.set_cursor(channel, cursor);
            replay_channel(logging, outbound, subscriptions, channel).await
        }
    }
}

async fn replay_subscriptions(
    logging: &Arc<LoggingService>,
    outbound: &mpsc::Sender<String>,
    subscriptions: &mut Subscriptions,
) -> bool {
    for channel in subscriptions.channels() {
        if !replay_channel(logging, outbound, subscriptions, channel).await {
            return false;
        }
    }
    true
}

async fn replay_channel(
    logging: &Arc<LoggingService>,
    outbound: &mpsc::Sender<String>,
    subscriptions: &mut Subscriptions,
    channel: WebSocketChannel,
) -> bool {
    let cursor = subscriptions.cursor(channel).unwrap_or_default();
    let replay = logging
        .bus_ref()
        .replay_after(channel.replay_channel(), cursor);
    if let Some(earliest) = replay.earliest_sequence {
        let first_missing = cursor.saturating_add(1);
        if first_missing < earliest {
            let recovery = recovery_cursor(logging, channel, replay.entries.first()).await;
            if !enqueue(
                outbound,
                ServerMessage::Gap {
                    channel,
                    from_sequence: first_missing,
                    to_sequence: earliest - 1,
                    recovery,
                },
            ) {
                return false;
            }
            subscriptions.set_cursor(channel, earliest - 1);
        }
    }

    for entry in replay.entries {
        let Some(sequence) = entry.replay.map(|replay| replay.sequence) else {
            continue;
        };
        if sequence <= subscriptions.cursor(channel).unwrap_or_default() {
            continue;
        }
        let message = match event_from_entry(&entry) {
            Ok(message) => message,
            Err(code) => return enqueue_error(outbound, code),
        };
        if enqueue(outbound, message) {
            subscriptions.set_cursor(channel, sequence);
            tokio::task::yield_now().await;
            continue;
        }
        let recovery = recovery_cursor(logging, channel, Some(&entry)).await;
        if !enqueue(
            outbound,
            ServerMessage::Gap {
                channel,
                from_sequence: sequence,
                to_sequence: sequence,
                recovery,
            },
        ) {
            return false;
        }
        subscriptions.set_cursor(channel, sequence);
    }
    true
}

async fn recovery_cursor(
    logging: &Arc<LoggingService>,
    channel: WebSocketChannel,
    entry: Option<&BusEntry>,
) -> RestRecovery {
    if channel != WebSocketChannel::Requests {
        return RestRecovery::unavailable();
    }
    let Some(request_id) = entry.and_then(request_id) else {
        return RestRecovery::requests(None);
    };
    let Some(artifact_store) = logging.log_store_ref() else {
        return RestRecovery::requests(None);
    };
    let result =
        tokio::task::spawn_blocking(move || artifact_store.store_ref().query_request(&request_id))
            .await;
    match result {
        Ok(Ok(Some(record))) => RestRecovery::requests(Some(mesh_llm_log_store::encode_cursor(
            &record.created_at,
            &record.request_id,
        ))),
        Ok(Ok(None) | Err(_)) | Err(_) => RestRecovery::requests(None),
    }
}

fn enqueue_heartbeat(outbound: &mpsc::Sender<String>, subscriptions: &Subscriptions) -> bool {
    enqueue(
        outbound,
        ServerMessage::Heartbeat {
            cursors: subscriptions.cursors(),
        },
    )
}

fn enqueue_error(outbound: &mpsc::Sender<String>, code: ErrorCode) -> bool {
    enqueue(outbound, ServerMessage::Error { code })
}

fn enqueue(outbound: &mpsc::Sender<String>, message: ServerMessage) -> bool {
    let Ok(encoded) = serde_json::to_string(&message) else {
        return false;
    };
    if encoded.len() > MAX_SERVER_FRAME_BYTES {
        return false;
    }
    outbound.try_send(encoded).is_ok()
}

async fn write_frames<S>(
    mut sink: futures_util::stream::SplitSink<WebSocketStream<S>, Message>,
    mut receiver: mpsc::Receiver<String>,
    writer_done: oneshot::Sender<()>,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    while let Some(frame) = receiver.recv().await {
        let result = timeout(WRITE_TIMEOUT, sink.send(Message::Text(frame.into()))).await;
        if !matches!(result, Ok(Ok(()))) {
            break;
        }
    }
    let _ = timeout(WRITE_TIMEOUT, sink.send(Message::Close(None))).await;
    let _ = writer_done.send(());
}
