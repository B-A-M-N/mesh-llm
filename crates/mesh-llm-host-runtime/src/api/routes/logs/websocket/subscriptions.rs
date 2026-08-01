use super::protocol::{ChannelCursor, WebSocketChannel};

#[derive(Default)]
pub(super) struct Subscriptions {
    requests: Option<u64>,
    operations: Option<u64>,
    system: Option<u64>,
}

impl Subscriptions {
    pub(super) fn subscribe(&mut self, channel: WebSocketChannel, cursor: u64) {
        self.set(channel, Some(cursor));
    }

    pub(super) fn unsubscribe(&mut self, channel: WebSocketChannel) {
        self.set(channel, None);
    }

    pub(super) fn contains(&self, channel: WebSocketChannel) -> bool {
        self.cursor(channel).is_some()
    }

    pub(super) fn cursor(&self, channel: WebSocketChannel) -> Option<u64> {
        match channel {
            WebSocketChannel::Requests => self.requests,
            WebSocketChannel::Operations => self.operations,
            WebSocketChannel::System => self.system,
        }
    }

    pub(super) fn set_cursor(&mut self, channel: WebSocketChannel, cursor: u64) {
        self.set(channel, Some(cursor));
    }

    pub(super) fn channels(&self) -> Vec<WebSocketChannel> {
        [
            (WebSocketChannel::Requests, self.requests),
            (WebSocketChannel::Operations, self.operations),
            (WebSocketChannel::System, self.system),
        ]
        .into_iter()
        .filter_map(|(channel, cursor)| cursor.map(|_| channel))
        .collect()
    }

    pub(super) fn cursors(&self) -> Vec<ChannelCursor> {
        self.channels()
            .into_iter()
            .filter_map(|channel| {
                self.cursor(channel)
                    .map(|cursor| ChannelCursor::new(channel, cursor))
            })
            .collect()
    }

    fn set(&mut self, channel: WebSocketChannel, cursor: Option<u64>) {
        match channel {
            WebSocketChannel::Requests => self.requests = cursor,
            WebSocketChannel::Operations => self.operations = cursor,
            WebSocketChannel::System => self.system = cursor,
        }
    }
}
