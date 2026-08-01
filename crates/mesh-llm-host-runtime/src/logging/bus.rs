//! Bounded nonblocking replay bus for logging events.
//!
//! **Overflow policy: drop-oldest.** When the queue is full, the oldest entry is evicted to make room. This preserves recent context at the cost of losing aged entries. Drop counters track both dropped events and evicted entries separately via `AtomicU64`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use mesh_llm_events::logging::replay::{ReplayChannel, ReplaySequence};
use tokio::sync::broadcast;

/// Entry on the replay bus carrying a serialized event payload.
#[derive(Clone, Debug)]
pub struct BusEntry {
    /// Serialized JSON/event content to persist. The sink deserializes this into domain types (lifecycle events, summaries, etc.). Payloads are sanitized before enqueue via the privacy policy redactor.
    pub payload: String,

    /// Channel routing hint for downstream consumers (requests/operations/system). This helps workers route entries without parsing JSON.
    #[allow(dead_code)]
    pub channel_hint: u8, // 0=requests, 1=operations, 2=system — matches ReplayChannel discriminant

    /// Cursor used for WebSocket replay when this entry is a lifecycle event.
    pub replay: Option<ReplaySequence>,
}

/// Retained entries for one replay channel and the first sequence still available.
#[derive(Clone, Debug, Default)]
pub struct ReplayWindow {
    /// Oldest retained sequence for this channel, if the bounded bus contains one.
    pub earliest_sequence: Option<u64>,
    /// Entries newer than the requested cursor in channel order.
    pub entries: Vec<BusEntry>,
}

/// Bounded nonblocking replay bus with drop-oldest overflow policy.
///
/// When `push` is called and the queue is already at capacity, the oldest entry
/// is evicted (popped from the front) before the new entry is appended. This ensures
/// recent context survives under pressure while older entries are discarded.
#[derive(Debug)]
pub struct ReplayBus {
    capacity: usize,
    replay_entries: Mutex<VecDeque<BusEntry>>,
    persistence_entries: Mutex<VecDeque<BusEntry>>,
    updates: broadcast::Sender<()>,

    /// Number of events dropped because the queue was full and overflow policy applied.
    pub drops: Arc<AtomicU64>,

    /// Number of oldest entries evicted to make room for new ones (under drop-oldest).
    pub evictions: Arc<AtomicU64>,
}

impl ReplayBus {
    /// Create a new bus with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let c = capacity.max(1);
        Self {
            capacity: c,
            replay_entries: Mutex::new(VecDeque::with_capacity(c)),
            persistence_entries: Mutex::new(VecDeque::with_capacity(c)),
            updates: broadcast::channel(c).0,
            drops: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Push an entry onto the bus. If full, drop-oldest applies (evict front, push back).
    pub fn push(&self, payload: String) {
        self.push_entry(BusEntry {
            payload,
            channel_hint: 0,
            replay: None,
        });
    }

    /// Push with a channel hint for downstream routing.
    #[allow(dead_code)]
    pub fn push_with_hint(&self, payload: String, channel_hint: u8) {
        self.push_entry(BusEntry {
            payload,
            channel_hint,
            replay: None,
        });
    }

    /// Push a sequenced lifecycle event for replay without blocking producers.
    pub fn push_replay(&self, channel: ReplayChannel, sequence: u64, payload: String) {
        self.push_entry(BusEntry {
            payload,
            channel_hint: channel_hint(channel),
            replay: Some(ReplaySequence { channel, sequence }),
        });
    }

    /// Read the retained portion of one channel newer than `cursor` without consuming it.
    pub fn replay_after(&self, channel: ReplayChannel, cursor: u64) -> ReplayWindow {
        let entries = self.lock_replay_entries();
        let mut window = ReplayWindow::default();

        for entry in entries.iter() {
            let Some(replay) = entry.replay else {
                continue;
            };
            if replay.channel != channel {
                continue;
            }
            window.earliest_sequence.get_or_insert(replay.sequence);
            if replay.sequence > cursor {
                window.entries.push(entry.clone());
            }
        }

        window
    }

    fn push_entry(&self, entry: BusEntry) {
        let mut entries = self.lock_replay_entries();

        if entries.len() == self.capacity {
            // Drop oldest to make room.
            entries.pop_front();
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }

        entries.push_back(entry.clone());
        drop(entries);
        let mut persistence_entries = self.lock_persistence_entries();
        if persistence_entries.len() == self.capacity {
            persistence_entries.pop_front();
        }
        persistence_entries.push_back(entry);
        drop(persistence_entries);
        let _ = self.updates.send(());
    }

    /// Drain all entries from the bus for batch processing by the persistence worker.
    pub fn drain(&self) -> Vec<BusEntry> {
        let mut entries = self.lock_persistence_entries();
        entries.drain(..).collect()
    }

    /// Current number of buffered entries (for observability / tests).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.lock_replay_entries().len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wait until at least one entry is available (or the bus has been signalled).
    pub fn subscribe_updates(&self) -> broadcast::Receiver<()> {
        self.updates.subscribe()
    }

    /// Clone the drops counter for external observation.
    #[allow(dead_code)]
    pub fn drops_clone(&self) -> Arc<AtomicU64> {
        self.drops.clone()
    }

    /// Clone the evictions counter for external observation.
    #[allow(dead_code)]
    pub fn evictions_clone(&self) -> Arc<AtomicU64> {
        self.evictions.clone()
    }

    fn lock_replay_entries(&self) -> MutexGuard<'_, VecDeque<BusEntry>> {
        match self.replay_entries.lock() {
            Ok(entries) => entries,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn lock_persistence_entries(&self) -> MutexGuard<'_, VecDeque<BusEntry>> {
        match self.persistence_entries.lock() {
            Ok(entries) => entries,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

const fn channel_hint(channel: ReplayChannel) -> u8 {
    match channel {
        ReplayChannel::Requests => 0,
        ReplayChannel::Operations => 1,
        ReplayChannel::System => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_within_capacity_no_eviction() {
        let bus = ReplayBus::new(3);
        bus.push("a".into());
        bus.push("b".into());
        assert_eq!(bus.len(), 2);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn overflow_drops_oldest() {
        let bus = ReplayBus::new(2);
        bus.push("old".into());
        bus.push("keep".into());
        assert_eq!(bus.len(), 2);

        // Push third → evicts "old"
        bus.push("new".into());
        assert_eq!(bus.len(), 2);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 1);

        let entries = bus.drain();
        assert_eq!(entries.len(), 2);
        // First entry should be "keep", not "old" (oldest was dropped).
        assert_eq!(entries[0].payload, "keep");
        assert_eq!(entries[1].payload, "new");
    }

    #[test]
    fn persistence_drain_does_not_empty_replay_history() {
        let bus = ReplayBus::new(4);
        bus.push("x".into());
        bus.drain();
        assert_eq!(bus.len(), 1);
    }

    #[test]
    fn persistence_drain_preserves_replay_history() {
        let bus = ReplayBus::new(2);
        bus.push_replay(ReplayChannel::Requests, 1, "first".into());
        bus.drain();

        let replay = bus.replay_after(ReplayChannel::Requests, 0);
        assert_eq!(replay.entries.len(), 1);
        assert_eq!(replay.entries[0].replay.unwrap().sequence, 1);
    }

    #[tokio::test]
    async fn update_fanout_wakes_all_subscribers() {
        let bus = ReplayBus::new(2);
        let mut first = bus.subscribe_updates();
        let mut second = bus.subscribe_updates();
        bus.push_replay(ReplayChannel::Requests, 1, "first".into());

        assert!(first.recv().await.is_ok());
        assert!(second.recv().await.is_ok());
    }

    #[test]
    fn capacity_clamped_to_one() {
        let bus = ReplayBus::new(0);
        bus.push("a".into());
        assert_eq!(bus.len(), 1); // capacity.max(1) == 1
    }
}
