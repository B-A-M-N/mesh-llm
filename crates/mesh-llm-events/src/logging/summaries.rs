//! Request summary for lifecycle tracking and reporting.

use super::identifiers::RequestId;
use super::lifecycle::LifecycleState;

/// Compact view of a request's lifecycle state and routing metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSummary {
    pub request_id: RequestId,
    pub state: LifecycleState,
    /// ISO 8601 timestamp when the request was created.
    pub created_at: String,
    /// ISO 8601 timestamp when the request reached a terminal state (if applicable).
    #[allow(dead_code)]
    pub terminal_at: Option<String>,

    // Routing metadata.
    #[allow(dead_code)]
    pub route: Option<String>,
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub provider: Option<String>,
    #[allow(dead_code)]
    pub engine: Option<String>,

    // Outcome metadata.
    #[allow(dead_code)]
    pub status_code: Option<u16>,
    #[allow(dead_code)]
    pub error: Option<String>,

    // Nullable reserved identity fields.
    #[allow(dead_code)]
    pub tenant_id: Option<String>,
    #[allow(dead_code)]
    pub account_id: Option<String>,
    #[allow(dead_code)]
    pub user_id: Option<String>,
}

impl RequestSummary {
    /// Create a new summary in the Active state.
    #[allow(dead_code)]
    pub fn new(request_id: RequestId, created_at: String) -> Self {
        Self {
            request_id,
            state: LifecycleState::Active,
            created_at,
            terminal_at: None,
            route: None,
            model: None,
            provider: None,
            engine: None,
            status_code: None,
            error: None,
            tenant_id: None,
            account_id: None,
            user_id: None,
        }
    }

    /// Mark the request as completed. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn set_completed(&mut self) {
        self.state = LifecycleState::Completed;
    }

    /// Mark the request as failed with an error message.
    #[allow(dead_code)]
    pub fn set_failed(&mut self, error: String) {
        self.state = LifecycleState::Failed;
        self.error = Some(error);
    }

    /// Check if this summary is in a terminal state.
    #[allow(dead_code)]
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_summary_is_active() {
        let summary = RequestSummary::new(RequestId::new(), "2025-01-01T00:00:00Z".into());
        assert_eq!(summary.state, LifecycleState::Active);
        assert!(!summary.is_terminal());
    }

    #[test]
    fn test_set_completed() {
        let mut summary = RequestSummary::new(RequestId::new(), "2025-01-01T00:00:00Z".into());
        summary.set_completed();
        assert_eq!(summary.state, LifecycleState::Completed);
        assert!(summary.is_terminal());
    }

    #[test]
    fn test_set_failed() {
        let mut summary = RequestSummary::new(RequestId::new(), "2025-01-01T00:00:00Z".into());
        summary.set_failed("timeout".into());
        assert_eq!(summary.state, LifecycleState::Failed);
        assert_eq!(summary.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn test_summary_clone() {
        let mut a = RequestSummary::new(RequestId::new(), "2025-01-01T00:00:00Z".into());
        a.set_completed();

        let b = a.clone();
        assert_eq!(b.state, LifecycleState::Completed);
    }
}
