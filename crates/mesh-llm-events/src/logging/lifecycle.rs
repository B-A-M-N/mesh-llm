//! Lifecycle state machine with exactly-one-terminal-transition invariant.

use std::fmt;

/// Terminal or active lifecycle states for a request or attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LifecycleState {
    /// The entity is currently being processed.
    Active,
    /// Processing finished successfully.
    Completed,
    /// Processing ended with an error.
    Failed,
    /// Processing was rejected before beginning (e.g., invalid input).
    Rejected,
    /// Processing was cancelled by the caller or system.
    Cancelled,
}

impl LifecycleState {
    /// Returns `true` if this state is terminal (no further transitions allowed).
    pub fn is_terminal(&self) -> bool {
        !matches!(self, LifecycleState::Active)
    }

    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            LifecycleState::Active => "active",
            LifecycleState::Completed => "completed",
            LifecycleState::Failed => "failed",
            LifecycleState::Rejected => "rejected",
            LifecycleState::Cancelled => "cancelled",
        }
    }
}

/// Error returned when a lifecycle transition is invalid.
#[derive(Clone, Copy, Debug)]
pub struct LifecycleTransitionError {
    pub from: LifecycleState,
    pub to: LifecycleState,
}

impl fmt::Display for LifecycleTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid transition {} → {}", self.from.as_str(), self.to.as_str())
    }
}

impl std::error::Error for LifecycleTransitionError {}

/// Guard that owns the current lifecycle state and enforces exactly-one-terminal-transition.
///
/// A guard starts in `Active` (or a provided initial state). Once it transitions to any terminal
/// state, all further transition attempts are rejected with [`LifecycleTransitionError`].
#[derive(Clone, Debug)]
pub struct LifecycleGuard {
    current: LifecycleState,
}

impl LifecycleGuard {
    /// Create a new guard starting in the given state (typically `Active`).
    pub fn new(state: LifecycleState) -> Self {
        Self { current: state }
    }

    /// Returns a guard initialized to `Active`.
    #[allow(dead_code)]
    pub fn active() -> Self {
        Self::new(LifecycleState::Active)
    }

    /// Return the current lifecycle state.
    #[allow(dead_code)]
    pub fn state(&self) -> LifecycleState {
        self.current
    }

    /// Transition to a new state, returning an error if:
    /// - The current state is already terminal (no further transitions allowed), OR
    /// - Attempting the same non-terminal transition twice where idempotency does not apply.
    ///
    /// Note: transitioning from `Active` → `Active` is allowed (idempotent no-op).
    pub fn transition(&mut self, new_state: LifecycleState) -> Result<(), LifecycleTransitionError> {
        // Allow idempotent non-terminal transitions (e.g., Active → Active)
        if self.current == new_state && !self.current.is_terminal() {
            return Ok(());
        }

        // Reject any transition once terminal
        if self.current.is_terminal() {
            return Err(LifecycleTransitionError { from: self.current, to: new_state });
        }

        self.current = new_state;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_to_completed_works() {
        let mut guard = LifecycleGuard::active();
        assert!(guard.transition(LifecycleState::Completed).is_ok());
        assert_eq!(guard.state(), LifecycleState::Completed);
    }

    #[test]
    fn test_completed_to_failed_fails() {
        let mut guard = LifecycleGuard::active();
        guard.transition(LifecycleState::Completed).unwrap();
        let err = guard.transition(LifecycleState::Failed).unwrap_err();
        assert_eq!(err.from, LifecycleState::Completed);
        assert_eq!(err.to, LifecycleState::Failed);
    }

    #[test]
    fn test_second_terminal_rejected() {
        let mut guard = LifecycleGuard::active();
        guard.transition(LifecycleState::Completed).unwrap();
        let err = guard.transition(LifecycleState::Completed).unwrap_err();
        assert_eq!(err.from, LifecycleState::Completed);
    }

    #[test]
    fn test_active_to_active_idempotent() {
        let mut guard = LifecycleGuard::active();
        assert!(guard.transition(LifecycleState::Active).is_ok());
        assert_eq!(guard.state(), LifecycleState::Active);
    }

    #[test]
    fn test_terminal_states_are_terminal() {
        for state in [LifecycleState::Completed, LifecycleState::Failed,
                      LifecycleState::Rejected, LifecycleState::Cancelled] {
            assert!(state.is_terminal(), "{:?} should be terminal", state);
        }
        assert!(!LifecycleState::Active.is_terminal());
    }

    #[test]
    fn test_active_to_all_terminals_work() {
        for target in [LifecycleState::Completed, LifecycleState::Failed,
                       LifecycleState::Rejected, LifecycleState::Cancelled] {
            let mut guard = LifecycleGuard::active();
            assert!(guard.transition(target).is_ok(), "Active→{:?} should work", target);
        }
    }

    #[test]
    fn test_lifecycle_transition_error_display() {
        let err = LifecycleTransitionError { from: LifecycleState::Completed, to: LifecycleState::Failed };
        let msg = format!("{}", err);
        assert!(msg.contains("completed"));
        assert!(msg.contains("failed"));
    }

    #[test]
    fn test_guard_clone_preserves_state() {
        let mut guard1 = LifecycleGuard::active();
        guard1.transition(LifecycleState::Failed).unwrap();

        let guard2 = guard1.clone();
        assert_eq!(guard2.state(), LifecycleState::Failed);
    }

    #[test]
    fn test_new_guard_with_custom_state() {
        let mut guard = LifecycleGuard::new(LifecycleState::Cancelled);
        assert_eq!(guard.state(), LifecycleState::Cancelled);
        // Already terminal → any transition fails
        assert!(guard.transition(LifecycleState::Active).is_err());
    }

    #[test]
    fn test_state_as_str() {
        assert_eq!(LifecycleState::Active.as_str(), "active");
        assert_eq!(LifecycleState::Completed.as_str(), "completed");
        assert_eq!(LifecycleState::Failed.as_str(), "failed");
        assert_eq!(LifecycleState::Rejected.as_str(), "rejected");
        assert_eq!(LifecycleState::Cancelled.as_str(), "cancelled");
    }
}
