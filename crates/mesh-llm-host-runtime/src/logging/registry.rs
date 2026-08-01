//! Active/recent request registry for logging service.
//!
//! Tracks in-flight requests (active) and recently completed ones (recent). Both sets are bounded:
//! when capacity is exceeded, the oldest entry by `created_at` is evicted FIFO-style.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Summary record for a single request in the registry. Carries enough metadata to reconstruct
/// what happened without persisting full payloads by default.
#[derive(Clone, Debug)]
pub struct RequestSummaryEntry {
    /// UUID string identifying this request (from `RequestId::as_uuid()`).
    pub request_id: String,

    /// Current lifecycle state: "active", "completed", "failed", etc. Updated on terminal transition.
    pub state: String,

    /// ISO 8601 timestamp when the entry was first registered in active. Never changes after creation.
    pub created_at: String,

    /// ISO 8601 timestamp of the terminal transition (completed/failed/etc.). `None` while active.
    pub terminal_at: Option<String>,
}

/// Configuration controlling registry capacity bounds.
#[derive(Clone, Debug)]
pub struct RegistryConfig {
    /// Maximum number of entries in the active set before FIFO eviction applies. Default: 1024.
    pub max_active: usize,

    /// Maximum number of entries in the recent set before FIFO eviction applies. Default: 8192.
    pub max_recent: usize,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_active: 1024,
            max_recent: 8192,
        }
    }
}

/// Active/recent request registry with bounded capacity and FIFO eviction.
///
/// Thread-safe via internal Mutex guards on each set. Designed to be shared behind `Arc` across
/// the service facade (bus push path, terminal transition path, observability reads).
pub struct RequestRegistry {
    /// In-flight requests keyed by UUID string. Evicted oldest-first when exceeding max_active.
    active: Mutex<HashMap<String, RequestSummaryEntry>>,

    /// Recently completed/failed/dropped requests keyed by UUID string. Evicted oldest-first when exceeding max_recent.
    recent: Mutex<HashMap<String, RequestSummaryEntry>>,

    config: RegistryConfig,

    /// Total number of entries evicted from the active set (for observability).
    pub active_evictions: Arc<AtomicU64>,

    /// Total number of entries evicted from the recent set (for observability).
    #[allow(dead_code)]
    pub recent_evictions: Arc<AtomicU64>,
}

impl RequestRegistry {
    /// Create a new registry with the given configuration.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            active: Mutex::new(HashMap::with_capacity(config.max_active)),
            recent: Mutex::new(HashMap::with_capacity(config.max_recent)),
            config,
            active_evictions: Arc::new(AtomicU64::new(0)),
            recent_evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Insert an entry into the active set. If the set is already at capacity, evict the oldest
    /// entry (by `created_at` lexicographic comparison) and increment `active_evictions`.
    pub fn register_active(&self, entry: RequestSummaryEntry) {
        let mut map = self.active.lock().expect("registry mutex poisoned");

        // Evict if at capacity.
        if map.len() >= self.config.max_active {
            evict_oldest(&mut map);
            self.active_evictions.fetch_add(1, Ordering::Relaxed);
        }

        map.insert(entry.request_id.clone(), entry);
    }

    /// Look up an active entry by request ID. Returns a clone (not a reference) since the caller
    /// may need to hold it across the Mutex unlock boundary.
    pub fn get_active(&self, request_id: &str) -> Option<RequestSummaryEntry> {
        self.active
            .lock()
            .expect("registry mutex poisoned")
            .get(request_id)
            .cloned()
    }

    /// Remove an entry from active and insert into recent. If recent exceeds max_recent, evict the
    /// oldest entry and increment `recent_evictions`. The caller is expected to have already updated
    /// the entry's state/terminal_at fields before calling this method.
    pub fn move_to_recent(&self, entry: RequestSummaryEntry) {
        let rid = entry.request_id.clone();

        // Remove from active first (may not exist if it was evicted; that's fine).
        {
            let mut map = self.active.lock().expect("registry mutex poisoned");
            map.remove(&rid);
        }

        // Insert into recent. Evict oldest if at capacity.
        {
            let mut map = self.recent.lock().expect("registry mutex poisoned");
            if map.len() >= self.config.max_recent {
                evict_oldest(&mut map);
                self.recent_evictions.fetch_add(1, Ordering::Relaxed);
            }

            map.insert(entry.request_id.clone(), entry);
        }
    }

    /// Current number of entries in the active set.
    pub fn active_count(&self) -> usize {
        self.active.lock().expect("registry mutex poisoned").len()
    }

    /// Current number of entries in the recent set.
    pub fn recent_count(&self) -> usize {
        self.recent.lock().expect("registry mutex poisoned").len()
    }

    /// Look up a recent entry by request ID. Returns a clone (not a reference).
    pub fn get_recent(&self, request_id: &str) -> Option<RequestSummaryEntry> {
        self.recent
            .lock()
            .expect("registry mutex poisoned")
            .get(request_id)
            .cloned()
    }

    /// Clear both active and recent sets. Used for shutdown or test isolation.
    pub fn clear(&self) {
        let mut map = self.active.lock().expect("registry mutex poisoned");
        map.clear();
        drop(map);

        let mut map = self.recent.lock().expect("registry mutex poisoned");
        map.clear();
    }

    /// Returns true if both active and recent sets are empty.
    pub fn is_empty(&self) -> bool {
        let a = self
            .active
            .lock()
            .expect("registry mutex poisoned")
            .is_empty();
        let r = self
            .recent
            .lock()
            .expect("registry mutex poisoned")
            .is_empty();
        a && r
    }
}

/// Remove the entry with the lexicographically smallest `created_at` from the map.
/// No-op if the map is empty. Uses ISO 8601 timestamp ordering (lexicographic = chronological).
fn evict_oldest(map: &mut HashMap<String, RequestSummaryEntry>) {
    let oldest_key = map
        .iter()
        .min_by_key(|(_, entry)| &entry.created_at)
        .map(|(key, _)| key.clone());

    if let Some(key) = oldest_key {
        map.remove(&key);
    }
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
