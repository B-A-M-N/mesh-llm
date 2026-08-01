//! Lifecycle guard owning exactly ONE terminal transition per request.
//!
//! A `LifecycleGuard` starts in an active state and allows transitions to any terminal state
//! (Completed/Failed/Rejected/Cancelled/Dropped) exactly once. After the first terminal transition,
//! all further terminal attempts are rejected with [`DuplicateTerminalError`] — making the guard idempotent.
//! Per-attempt events (e.g., retry `AttemptStarted`) do NOT terminate the parent request; they are
//! always allowed regardless of current state and produce a separate attempt-scoped log entry.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Terminal outcomes for a lifecycle guard transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminalOutcome {
    /// Request completed successfully.
    Completed,
    /// Request failed with an error condition.
    Failed(String),
    /// Request rejected before processing (e.g., invalid input).
    Rejected(Option<String>),
    /// Request cancelled by caller or system.
    Cancelled(Option<String>),
    /// Request dropped without terminal processing (queue overflow, timeout, etc.).
    Dropped(Option<String>),
}

impl TerminalOutcome {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed(_) => "failed",
            Self::Rejected(_) => "rejected",
            Self::Cancelled(_) => "cancelled",
            Self::Dropped(_) => "dropped",
        }
    }

    /// Returns the error/reason string if this is a non-success terminal outcome.
    #[allow(dead_code)]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Completed => None,
            Self::Failed(e) => Some(e),
            Self::Rejected(r) => r.as_deref(),
            Self::Cancelled(r) => r.as_deref(),
            Self::Dropped(r) => r.as_deref(),
        }
    }

    /// Is this a success outcome?
    #[allow(dead_code)]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

impl fmt::Display for TerminalOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
            Self::Failed(e) => write!(f, "failed: {}", e),
            Self::Rejected(r) => write!(f, "rejected: {}", r.as_deref().unwrap_or("unknown")),
            Self::Cancelled(r) => write!(f, "cancelled: {}", r.as_deref().unwrap_or("unknown")),
            Self::Dropped(r) => write!(f, "dropped: {}", r.as_deref().unwrap_or("unknown")),
        }
    }
}

/// Error returned when a duplicate terminal transition is attempted.
#[derive(Clone, Debug)]
pub struct DuplicateTerminalError {
    pub existing: TerminalOutcome,
    pub attempted: TerminalOutcome,
}

impl fmt::Display for DuplicateTerminalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "duplicate terminal transition: {} → {}",
            self.existing.as_str(),
            self.attempted.as_str()
        )
    }
}

impl std::error::Error for DuplicateTerminalError {}

/// Internal state encoding as a single byte.
const STATE_ACTIVE: u8 = 0;
// Terminal states are encoded by outcome variant index + data pointer stored separately via Arc<Mutex<...>>.

/// Shared terminal state storage (only written once, read many).
#[derive(Clone, Debug, Default)]
struct TerminalRecord {
    outcome: Option<TerminalOutcome>,
}

/// Lifecycle guard with exactly-one-terminal-transition invariant.
///
/// The guard is `Clone` — all clones share the same underlying terminal state via `Arc`. This means
/// cloning a guard does not create independent lifecycle tracks; they all observe and enforce the same
/// single-terminal constraint. Per-attempt events are recorded separately (via the service) without
/// affecting this parent request's terminal status.
#[derive(Clone, Debug)]
pub struct LifecycleGuard {
    /// Atomic flag: 0 = active, non-zero = terminal outcome index stored in record.
    state_flag: Arc<AtomicU8>,
    /// Stores the actual TerminalOutcome once transitioned (None while active).
    record: Arc<std::sync::Mutex<TerminalRecord>>,
}

impl LifecycleGuard {
    /// Create a new guard starting in Active state.
    pub fn new() -> Self {
        Self {
            state_flag: Arc::new(AtomicU8::new(STATE_ACTIVE)),
            record: Arc::new(std::sync::Mutex::new(TerminalRecord::default())),
        }
    }

    /// Check if this guard is still in an active (non-terminal) state.
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        self.state_flag.load(Ordering::Acquire) == STATE_ACTIVE
    }

    /// Transition to a terminal outcome. Returns `Err(DuplicateTerminalError)` if already terminated.
    /// This transition is exactly-once: the first call succeeds, all subsequent calls are rejected.
    pub fn terminate(&self, outcome: TerminalOutcome) -> Result<(), DuplicateTerminalError> {
        // CAS from ACTIVE to a terminal marker (any non-zero value). We use 1 as "terminal" since we only care about active vs not-active in the flag.
        match self
            .state_flag
            .compare_exchange(STATE_ACTIVE, 1, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                // Successfully claimed terminal — store the outcome.
                let mut record = self.record.lock().expect("terminal record mutex poisoned");
                if let Some(existing) = &record.outcome {
                    return Err(DuplicateTerminalError {
                        existing: existing.clone(),
                        attempted: outcome,
                    });
                }
                record.outcome = Some(outcome);
                Ok(())
            }
            Err(_) => {
                // Already terminal — return the error with both outcomes.
                let record = self.record.lock().expect("terminal record mutex poisoned");
                match &record.outcome {
                    Some(existing) => Err(DuplicateTerminalError {
                        existing: existing.clone(),
                        attempted: outcome,
                    }),

                    None => {
                        // Edge case: CAS failed but no record yet (shouldn't happen in practice).
                        // Treat as duplicate with a placeholder.
                        Err(DuplicateTerminalError {
                            existing: TerminalOutcome::Dropped(None),
                            attempted: outcome,
                        })
                    }
                }
            }
        }
    }

    /// Attempt to transition idempotently — if already at the same terminal state, return Ok; otherwise attempt.
    pub fn terminate_idempotent(
        &self,
        outcome: TerminalOutcome,
    ) -> Result<(), DuplicateTerminalError> {
        match self.terminate(outcome) {
            Ok(()) => Ok(()),
            Err(DuplicateTerminalError {
                existing,
                attempted,
            }) if existing == attempted => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Get the current terminal outcome (if any). Returns None while active.
    #[allow(dead_code)]
    pub fn terminal_outcome(&self) -> Option<TerminalOutcome> {
        let record = self.record.lock().expect("terminal record mutex poisoned");
        record.outcome.clone()
    }

    /// Record a per-attempt event (e.g., retry attempt started). These do NOT affect the parent request's lifecycle — they are always allowed. Returns `true` to signal the caller should emit this as an attempt-scoped log entry rather than a terminal one.
    #[allow(dead_code)]
    pub fn record_attempt(&self) -> bool {
        // Always returns true — per-attempt events never terminate the parent.
        true
    }

    /// Clone the internal state flag for external observation (drop counters etc.).
    #[allow(dead_code)]
    pub fn state_flag_clone(&self) -> Arc<AtomicU8> {
        self.state_flag.clone()
    }

    /// Whether two guards share the same terminal track.
    #[allow(dead_code)]
    pub fn shares_track_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state_flag, &other.state_flag) && Arc::ptr_eq(&self.record, &other.record)
    }
}

impl Default for LifecycleGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_guard_is_active() {
        let guard = LifecycleGuard::new();
        assert!(guard.is_active());
        assert_eq!(guard.terminal_outcome(), None);
    }

    #[test]
    fn terminate_completed_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(guard.terminate(TerminalOutcome::Completed).is_ok());
        assert!(!guard.is_active());
        assert_eq!(guard.terminal_outcome(), Some(TerminalOutcome::Completed));
    }

    #[test]
    fn terminate_failed_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Failed("timeout".into()))
                .is_ok()
        );
        assert_eq!(
            guard.terminal_outcome(),
            Some(TerminalOutcome::Failed("timeout".into()))
        );
    }

    #[test]
    fn terminate_rejected_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Rejected(Some("invalid model".into())))
                .is_ok()
        );
    }

    #[test]
    fn terminate_cancelled_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(guard.terminate(TerminalOutcome::Cancelled(None)).is_ok());
    }

    #[test]
    fn terminate_dropped_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Dropped(Some("queue full".into())))
                .is_ok()
        );
    }

    #[test]
    fn second_terminal_rejected_as_duplicate() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Completed).unwrap();

        let err = guard
            .terminate(TerminalOutcome::Failed("oops".into()))
            .unwrap_err();
        assert_eq!(err.existing, TerminalOutcome::Completed);
        assert_eq!(err.attempted, TerminalOutcome::Failed("oops".into()));
    }

    #[test]
    fn idempotent_same_terminal_ok() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Cancelled(None)).unwrap();

        // Same terminal → Ok (idempotent)
        assert!(
            guard
                .terminate_idempotent(TerminalOutcome::Cancelled(None))
                .is_ok()
        );
    }

    #[test]
    fn idempotent_different_terminal_rejected() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Completed).unwrap();

        // Different terminal → Error
        assert!(
            guard
                .terminate_idempotent(TerminalOutcome::Failed("x".into()))
                .is_err()
        );
    }

    #[test]
    fn cloned_guard_shares_terminal_state() {
        let guard = LifecycleGuard::new();
        let cloned = guard.clone();

        guard.terminate(TerminalOutcome::Completed).unwrap();
        assert!(!cloned.is_active());
        assert_eq!(cloned.terminal_outcome(), Some(TerminalOutcome::Completed));

        // Cannot terminate via clone either.
        assert!(
            cloned
                .terminate(TerminalOutcome::Failed("x".into()))
                .is_err()
        );
    }

    #[test]
    fn record_attempt_always_allowed() {
        let guard = LifecycleGuard::new();
        assert!(guard.record_attempt());

        // Even after terminal, attempt recording is allowed.
        guard.terminate(TerminalOutcome::Completed).unwrap();
        assert!(guard.record_attempt());
    }

    #[test]
    fn shares_track_with_same_instance() {
        let g = LifecycleGuard::new();
        assert!(g.shares_track_with(&g));
    }

    #[test]
    fn shares_track_with_clone() {
        let g1 = LifecycleGuard::new();
        let g2 = g1.clone();
        assert!(g1.shares_track_with(&g2));
    }

    #[test]
    fn does_not_share_track_independent() {
        let g1 = LifecycleGuard::new();
        let g2 = LifecycleGuard::new();
        assert!(!g1.shares_track_with(&g2));
    }

    #[test]
    fn terminal_outcome_as_str() {
        for outcome in [
            TerminalOutcome::Completed,
            TerminalOutcome::Failed("e".into()),
            TerminalOutcome::Rejected(None),
            TerminalOutcome::Cancelled(Some("r".into())),
            TerminalOutcome::Dropped(None),
        ] {
            assert!(!outcome.as_str().is_empty());
        }

        let g = LifecycleGuard::new();
        g.terminate(TerminalOutcome::Completed).unwrap();
        assert!(g.terminal_outcome().unwrap().is_success());
    }

    #[test]
    fn terminal_outcome_reason() {
        assert_eq!(TerminalOutcome::Completed.reason(), None);
        assert_eq!(TerminalOutcome::Failed("err".into()).reason(), Some("err"));
        assert_eq!(
            TerminalOutcome::Rejected(Some("r".into())).reason(),
            Some("r")
        );
        assert_eq!(TerminalOutcome::Cancelled(None).reason(), None);
    }

    #[test]
    fn default_guard_is_active() {
        let guard = LifecycleGuard::default();
        assert!(guard.is_active());
    }

    #[test]
    fn display_for_terminal_outcome() {
        let msg = format!("{}", TerminalOutcome::Failed("timeout".into()));
        assert!(msg.contains("failed"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn duplicate_terminal_error_display() {
        let err = DuplicateTerminalError {
            existing: TerminalOutcome::Completed,
            attempted: TerminalOutcome::Failed("x".into()),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("completed"));
        assert!(msg.contains("failed"));
    }

    #[test]
    fn duplicate_terminal_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(DuplicateTerminalError {
            existing: TerminalOutcome::Completed,
            attempted: TerminalOutcome::Failed("x".into()),
        });
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn concurrent_terminate_only_one_succeeds() {
        use std::thread;

        let guard = LifecycleGuard::new();
        let mut handles = vec![];

        for _i in 0..10 {
            let g = guard.clone();
            handles.push(thread::spawn(move || {
                match g.terminate(TerminalOutcome::Completed) {
                    Ok(()) => true,
                    Err(_) => false,
                }
            }));
        }

        let successes: usize = handles
            .into_iter()
            .map(|h| h.join().unwrap() as usize)
            .sum();
        assert_eq!(successes, 1, "exactly one thread should succeed");
    }
}
