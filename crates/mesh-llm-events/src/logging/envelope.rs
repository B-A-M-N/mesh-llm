//! Canonical event envelope with versioned schema and identity context.

use serde::{Deserialize, Serialize};

use super::events::LifecycleEvent;
use super::identifiers::{EventId, RequestId};
use super::replay::ReplayChannel;

/// Current canonical logging schema version. Bump on additive changes to the envelope shape.
pub const SCHEMA_VERSION: u16 = 1;

/// Top-level event envelope carrying all metadata required for persistence and replay.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CanonicalEnvelope {
    pub schema_version: u16,
    pub event_id: EventId,
    pub request_id: RequestId,
    #[serde(rename = "channel")]
    pub channel: ReplayChannel,
    pub sequence: u64,
    /// ISO 8601 timestamp of when the event occurred.
    pub occurred_at: String,

    /// The lifecycle payload for this envelope.
    #[serde(flatten)]
    pub event: LifecycleEvent,

    /// Nullable reserved identity fields (omitted from JSON when None).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

impl CanonicalEnvelope {
    /// Create a new envelope with the given fields. Identity fields default to None.
    #[allow(dead_code)]
    pub fn new(
        event_id: EventId,
        request_id: RequestId,
        channel: ReplayChannel,
        sequence: u64,
        occurred_at: String,
        event: LifecycleEvent,
    ) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            event_id,
            request_id,
            channel,
            sequence,
            occurred_at,
            event,
            tenant_id: None,
            account_id: None,
            user_id: None,
            role: None,
        }
    }

    /// Set the tenant ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_tenant(mut self, tenant_id: String) -> Self {
        self.tenant_id = Some(tenant_id);
        self
    }

    /// Set the account ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_account(mut self, account_id: String) -> Self {
        self.account_id = Some(account_id);
        self
    }

    /// Set the user ID. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    /// Set the role. Returns `&mut self` for chaining.
    #[allow(dead_code)]
    pub fn with_role(mut self, role: String) -> Self {
        self.role = Some(role);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_version_constant() {
        assert_eq!(SCHEMA_VERSION, 1);
    }

    #[test]
    fn test_envelope_new_minimal() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );

        assert_eq!(env.schema_version, SCHEMA_VERSION);
        assert!(env.tenant_id.is_none());
    }

    #[test]
    fn test_envelope_with_identity() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Operations,
            1,
            "2025-06-15T12:30:00Z".into(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: None,
            },
        )
        .with_tenant("t-abc".into())
        .with_account("a-def".into());

        assert_eq!(env.tenant_id, Some("t-abc".into()));
        assert_eq!(env.account_id, Some("a-def".into()));
    }

    #[test]
    fn test_envelope_serde_roundtrip() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::System,
            42,
            "2025-07-01T10:00:00Z".into(),
            LifecycleEvent::Failed {
                error: "timeout".into(),
            },
        );

        let json = serde_json::to_string(&env).unwrap();
        let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, env);
    }

    #[test]
    fn test_envelope_identity_omitted_when_none() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );

        let json = serde_json::to_string(&env).unwrap();
        assert!(!json.contains("tenant_id"));
        assert!(!json.contains("account_id"));
        assert!(!json.contains("user_id"));
        assert!(!json.contains("role"));
    }

    #[test]
    fn test_envelope_identity_included_when_set() {
        let env = CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            0,
            "2025-01-01T00:00:00Z".into(),
            LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        )
        .with_user("u-xyz".into())
        .with_role("admin".into());

        let json = serde_json::to_string(&env).unwrap();
        assert!(json.contains("user_id"));
        assert!(json.contains("role"));
    }
}
