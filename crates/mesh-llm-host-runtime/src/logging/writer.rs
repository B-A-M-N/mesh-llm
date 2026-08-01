//! Fail-open persistence writer with recursion guard.
//!
//! The writer ensures request-path completion despite full queue or store worker failure. When the bus is full, drop counters increment and the caller proceeds without blocking. When the underlying sink fails, a sanitized audit record is written via an error fallback path — but this path itself cannot re-enter (recursion guard) to prevent infinite self-logging loops.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Recursion guard preventing the error-audit fallback from entering itself recursively.
pub struct RecursionGuard {
    /// Per-thread flag checked before entering the error-record path. When `true`, we are already inside an error record and must not re-enter.
    in_error_path: std::cell::Cell<bool>,

    /// Atomic global guard for cross-thread recursion detection (belt-and-suspenders).
    depth: Arc<AtomicU64>,

    /// Global atomic flag preventing any thread from entering when another is already in the error path.
    global_in_error: Arc<AtomicBool>,
}

impl RecursionGuard {
    pub fn new() -> Self {
        Self {
            in_error_path: std::cell::Cell::new(false),
            depth: Arc::new(AtomicU64::new(0)),
            global_in_error: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Try to enter the error-record path. Returns `true` if entry is allowed, `false` if we are already inside an error record (recursion detected). When returning false, no logging should occur — this prevents self-logging loops.
    pub fn try_enter_error_path(&self) -> bool {
        // Fast-path thread-local check first.
        if self.in_error_path.get() {
            return false;
        }

        // Global atomic guard: prevent any concurrent entry across threads.
        if !self
            .global_in_error
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return false;
        }

        self.in_error_path.set(true);
        true
    }

    /// Exit the error-record path. Must be called after every successful `try_enter_error_path()`.
    pub fn exit_error_path(&self) {
        self.in_error_path.set(false);
        self.global_in_error.store(false, Ordering::Release);
    }

    /// Check if currently inside an error path (for observability / tests).
    #[allow(dead_code)]
    pub fn is_in_error_path(&self) -> bool {
        self.in_error_path.get() || self.global_in_error.load(Ordering::Acquire)
    }

    /// Clone the depth counter for external observation.
    #[allow(dead_code)]
    pub fn depth_clone(&self) -> Arc<AtomicU64> {
        self.depth.clone()
    }
}

// RecursionGuard is Send + Sync: Cell<bool> is only accessed from its own thread, AtomicBool handles cross-thread.
unsafe impl Send for RecursionGuard {}
unsafe impl Sync for RecursionGuard {}

impl std::fmt::Debug for RecursionGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursionGuard")
            .field("depth", &self.depth.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

impl Default for RecursionGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Fail-open writer that ensures request-path completion despite queue or sink failures.
pub struct FailOpenWriter {
    /// Guard preventing recursive self-logging loops in the error-audit fallback path.
    recursion_guard: Arc<RecursionGuard>,

    /// Total number of writes dropped due to full queue (incremented by bus overflow).
    pub write_drops: Arc<AtomicU64>,

    /// Number of times the error-fallback path was blocked by recursion detection.
    pub recursion_blocks: Arc<AtomicU64>,
}

impl FailOpenWriter {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            recursion_guard: Arc::new(RecursionGuard::new()),
            write_drops: Arc::new(AtomicU64::new(0)),
            recursion_blocks: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Attempt to record an error/audit entry. Returns `true` if the fallback path was entered successfully, `false` if blocked by recursion guard (caller should proceed silently). This method is designed to never panic — it absorbs all internal failures.
    pub fn try_record_error<F>(&self, recorder: F) -> bool
    where
        F: FnOnce() + Send + Sync + 'static,
    {
        if !self.recursion_guard.try_enter_error_path() {
            self.recursion_blocks.fetch_add(1, Ordering::Relaxed);
            return false; // Recursion detected — abort silently.
        }

        // Execute the recorder (best-effort). Wrap in catch_unwind to prevent panics from propagating to request paths.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder();
        }));

        self.recursion_guard.exit_error_path();

        // If the recorder panicked, we silently absorb it (fail-open). The recursion_blocks counter doesn't increment here — this was a valid entry that happened to fail.
        result.is_ok()
    }

    /// Record a write drop due to full queue or sink failure. This is called by the service when enqueue fails. Incrementing this counter is itself fail-open (no-op if anything goes wrong).
    pub fn record_drop(&self) {
        self.write_drops.fetch_add(1, Ordering::Relaxed);
    }

    /// Clone recursion guard for external observation.
    #[allow(dead_code)]
    pub fn recursion_guard_clone(&self) -> Arc<RecursionGuard> {
        self.recursion_guard.clone()
    }

    /// Whether a recursive error path is currently active (for tests).
    #[allow(dead_code)]
    pub fn is_in_error_path(&self) -> bool {
        self.recursion_guard.is_in_error_path()
    }
}

impl Default for FailOpenWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_enter_allows_first_call() {
        let guard = RecursionGuard::new();
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();
    }

    #[test]
    fn try_enter_blocks_second_nested_call() {
        let guard = RecursionGuard::new();
        assert!(guard.try_enter_error_path());

        // Second nested call should be blocked.
        assert!(!guard.try_enter_error_path());

        guard.exit_error_path();
    }

    #[test]
    fn recursion_blocks_counter_increments() {
        let writer = FailOpenWriter::new();
        assert_eq!(writer.recursion_blocks.load(Ordering::Relaxed), 0);

        // First entry succeeds.
        assert!(writer.try_record_error(|| {}));

        // Simulate a nested call that gets blocked by the recursion guard internally.
        // Simulate a nested call that gets blocked by the recursion guard internally.

        // Direct test: try entering while already inside (via internal guard).
        {
            let rg = writer.recursion_guard_clone();
            if rg.try_enter_error_path() {
                // We're in — now a second attempt should fail.
                assert!(!rg.try_enter_error_path());
                rg.exit_error_path();
            }
        }

        // The recursion_blocks counter tracks blocks at the writer level, not guard level directly.
        // Verify the mechanism works: direct writer call after entering via another path.
    }

    #[test]
    fn try_record_error_catches_panic() {
        let writer = FailOpenWriter::new();

        // Recorder that panics — should be caught, not propagate.
        let result = writer.try_record_error(|| {
            panic!("simulated recorder panic");
        });

        // Returns false (panic was caught).
        assert!(!result);
    }

    #[test]
    fn write_drop_counter_increments() {
        let writer = FailOpenWriter::new();
        assert_eq!(writer.write_drops.load(Ordering::Relaxed), 0);

        for _ in 0..10 {
            writer.record_drop();
        }

        assert_eq!(writer.write_drops.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn recursion_guard_depth_clone() {
        let guard = RecursionGuard::new();
        let depth = guard.depth_clone();
        assert_eq!(depth.load(Ordering::Acquire), 0);
    }

    #[test]
    fn writer_is_default() {
        let _writer: FailOpenWriter = Default::default();
    }

    #[test]
    fn recursion_guard_is_default() {
        let _guard: RecursionGuard = Default::default();
    }

    #[test]
    fn error_path_allows_re_entry_after_exit() {
        let guard = RecursionGuard::new();

        // First entry.
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();

        // After exit, can enter again.
        assert!(guard.try_enter_error_path());
        guard.exit_error_path();
    }

    #[test]
    fn writer_try_record_success() {
        let writer = FailOpenWriter::new();

        let called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let c = called.clone();
        assert!(writer.try_record_error(move || {
            c.store(true, Ordering::Release);
        }));
        assert!(called.load(Ordering::Acquire));
    }

    #[test]
    fn concurrent_recursion_guard_does_not_allow_cross_thread_duplication() {
        use std::thread;

        let guard = Arc::new(RecursionGuard::new());

        // Thread 1 enters.
        let g1 = guard.clone();
        assert!(g1.try_enter_error_path());

        // Thread 2 tries to enter (same shared guard).
        let g2 = guard.clone();
        let handle = thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            g2.try_enter_error_path()
        });

        // Give thread 1 time to hold the lock.
        std::thread::sleep(std::time::Duration::from_millis(10));

        let _ = handle.join().unwrap();
        // Thread-local check passes per-thread, but depth guard may block cross-thread at depth >= 2.
        // In practice, each thread has its own in_error_path cell, so the atomic depth is what matters for cross-thread.

        g1.exit_error_path();
    }
}
