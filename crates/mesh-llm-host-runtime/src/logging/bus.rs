//! Bounded nonblocking replay bus for logging events.
//!
//! **Overflow policy: drop-oldest.** When the queue is full, the oldest entry is evicted to make room. This preserves recent context at the cost of losing aged entries. Drop counters track both dropped events and evicted entries separately via `AtomicU64`.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

/// Entry on the replay bus carrying a serialized event payload.
#[derive(Clone, Debug)]
pub struct BusEntry {
    /// Serialized JSON/event content to persist. The sink deserializes this into domain types (lifecycle events, summaries, etc.). Payloads are sanitized before enqueue via the privacy policy redactor.
    pub payload: String,

    /// Channel routing hint for downstream consumers (requests/operations/system). This helps workers route entries without parsing JSON.
    #[allow(dead_code)]
    pub channel_hint: u8, // 0=requests, 1=operations, 2=system — matches ReplayChannel discriminant
}

/// Bounded nonblocking replay bus with drop-oldest overflow policy.
///
/// When `push` is called and the queue is already at capacity, the oldest entry
/// is evicted (popped from the front) before the new entry is appended. This ensures
/// recent context survives under pressure while older entries are discarded.
#[derive(Debug)]
pub struct ReplayBus {
    capacity: usize,
    entries: Mutex<VecDeque<BusEntry>>,
    notify: Notify,

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
            entries: Mutex::new(VecDeque::with_capacity(c)),
            notify: Notify::new(),
            drops: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Push an entry onto the bus. If full, drop-oldest applies (evict front, push back).
    pub fn push(&self, payload: String) {
        self.push_with_hint(payload, 0);
    }

    /// Push with a channel hint for downstream routing.
    #[allow(dead_code)]
    pub fn push_with_hint(&self, payload: String, channel_hint: u8) {
        let mut entries = self.entries.lock().expect("bus mutex poisoned");

        if entries.len() == self.capacity {
            // Drop oldest to make room.
            entries.pop_front();
            self.evictions.fetch_add(1, Ordering::Relaxed);
        }

        entries.push_back(BusEntry {
            payload,
            channel_hint,
        });
        drop(entries);
        self.notify.notify_one();
    }

    /// Drain all entries from the bus for batch processing by the persistence worker.
    pub fn drain(&self) -> Vec<BusEntry> {
        let mut entries = self.entries.lock().expect("bus mutex poisoned");
        entries.drain(..).collect()
    }

    /// Current number of buffered entries (for observability / tests).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.lock().expect("bus mutex poisoned").len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wait until at least one entry is available (or the bus has been signalled).
    pub async fn notified(&self) {
        self.notify.notified().await;
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
    fn drain_is_empty_after() {
        let bus = ReplayBus::new(4);
        bus.push("x".into());
        bus.drain();
        assert!(bus.is_empty());
    }

    #[test]
    fn capacity_clamped_to_one() {
        let bus = ReplayBus::new(0);
        bus.push("a".into());
        assert_eq!(bus.len(), 1); // capacity.max(1) == 1
    }
}
