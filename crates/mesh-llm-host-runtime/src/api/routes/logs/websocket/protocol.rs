use mesh_llm_events::logging::replay::ReplayChannel;
use serde::{Deserialize, Serialize};

use crate::logging::BusEntry;

pub(super) const MAX_CLIENT_MESSAGE_BYTES: usize = 4 * 1024;
pub(super) const MAX_SERVER_FRAME_BYTES: usize = 16 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum WebSocketChannel {
    Requests,
    Operations,
    System,
}

impl WebSocketChannel {
    pub(super) const fn replay_channel(self) -> ReplayChannel {
        match self {
            Self::Requests => ReplayChannel::Requests,
            Self::Operations => ReplayChannel::Operations,
            Self::System => ReplayChannel::System,
        }
    }

    pub(super) const fn from_replay(channel: ReplayChannel) -> Self {
        match channel {
            ReplayChannel::Requests => Self::Requests,
            ReplayChannel::Operations => Self::Operations,
            ReplayChannel::System => Self::System,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ClientMessage {
    Subscribe {
        channel: WebSocketChannel,
        #[serde(default)]
        cursor: Option<u64>,
    },
    Unsubscribe {
        channel: WebSocketChannel,
    },
    ReplayCursor {
        channel: WebSocketChannel,
        cursor: u64,
    },
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ErrorCode {
    InvalidEvent,
    InvalidMessage,
    MessageTooLarge,
    NotSubscribed,
    UnsupportedFrame,
}

#[derive(Debug, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub(super) enum ServerMessage {
    Event {
        channel: WebSocketChannel,
        sequence: u64,
        request_id: String,
        occurred_at: String,
    },
    Heartbeat {
        cursors: Vec<ChannelCursor>,
    },
    Gap {
        channel: WebSocketChannel,
        from_sequence: u64,
        to_sequence: u64,
        recovery: RestRecovery,
    },
    Error {
        code: ErrorCode,
    },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct ChannelCursor {
    channel: WebSocketChannel,
    cursor: u64,
}

impl ChannelCursor {
    pub(super) const fn new(channel: WebSocketChannel, cursor: u64) -> Self {
        Self { channel, cursor }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct RestRecovery {
    endpoint: Option<&'static str>,
    cursor: Option<String>,
    available: bool,
}

impl RestRecovery {
    pub(super) fn requests(cursor: Option<String>) -> Self {
        Self {
            endpoint: Some("/api/logs/requests"),
            cursor,
            available: true,
        }
    }

    pub(super) const fn unavailable() -> Self {
        Self {
            endpoint: None,
            cursor: None,
            available: false,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BufferedEvent {
    request_id: String,
    channel: ReplayChannel,
    sequence: u64,
    occurred_at: String,
    payload: String,
}

pub(super) fn parse_client(text: &str) -> Result<ClientMessage, ErrorCode> {
    if text.len() > MAX_CLIENT_MESSAGE_BYTES {
        return Err(ErrorCode::MessageTooLarge);
    }
    serde_json::from_str(text).map_err(|_| ErrorCode::InvalidMessage)
}

pub(super) fn event_from_entry(entry: &BusEntry) -> Result<ServerMessage, ErrorCode> {
    let replay = entry.replay.ok_or(ErrorCode::InvalidEvent)?;
    let event = buffered_event(entry)?;
    if event.channel != replay.channel || event.sequence != replay.sequence {
        return Err(ErrorCode::InvalidEvent);
    }
    let _: serde_json::Value =
        serde_json::from_str(&event.payload).map_err(|_| ErrorCode::InvalidEvent)?;
    Ok(ServerMessage::Event {
        channel: WebSocketChannel::from_replay(replay.channel),
        sequence: replay.sequence,
        request_id: event.request_id,
        occurred_at: event.occurred_at,
    })
}

pub(super) fn request_id(entry: &BusEntry) -> Option<String> {
    buffered_event(entry).ok().map(|event| event.request_id)
}

fn buffered_event(entry: &BusEntry) -> Result<BufferedEvent, ErrorCode> {
    serde_json::from_str(&entry.payload).map_err(|_| ErrorCode::InvalidEvent)
}
