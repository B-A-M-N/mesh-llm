//! Cross-module acceptance tests for canonical logging contracts.

use super::envelope::CanonicalEnvelope;
use super::events::LifecycleEvent;
use super::identifiers::{EventId, RequestId};
use super::lifecycle::{LifecycleGuard, LifecycleState};
use super::replay::ReplayChannel;

#[test]
fn test_invalid_lifecycle_transition() {
    let mut guard = LifecycleGuard::active();
    assert!(guard.transition(LifecycleState::Completed).is_ok());
    assert!(guard.transition(LifecycleState::Failed).is_err());
}

#[test]
fn test_second_terminal_rejected() {
    let mut guard = LifecycleGuard::active();
    assert!(guard.transition(LifecycleState::Completed).is_ok());
    assert!(guard.transition(LifecycleState::Completed).is_err());
}

#[test]
fn test_serde_roundtrip_preserves_fields() {
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        7,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Admitted {
            model: Some("llama-3".into()),
            method: Some("POST".into()),
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.schema_version, env.schema_version);
    assert_eq!(parsed.channel, ReplayChannel::Requests);
    assert_eq!(parsed.sequence, 7);
    assert_eq!(parsed.event, env.event);
}

#[test]
fn test_absent_identity_fields_omitted() {
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::System,
        0,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: None,
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    assert!(!json.contains("tenant_id"));
    assert!(!json.contains("account_id"));
    assert!(!json.contains("user_id"));
    assert!(!json.contains("\"role\""));
}

#[test]
fn test_no_raw_invite_token_in_serialized_events() {
    // The contract types introduce no token-shaped fields: serialized output
    // never gains keys like "invite" or "token". Caller-supplied error strings
    // pass through verbatim (redaction is a later policy-layer concern).
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        1,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Failed {
            error: "upstream rejected the request".into(),
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.event, env.event);
    // No token-bearing key is introduced by the envelope itself.
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let obj = value.as_object().unwrap();
    assert!(!obj.contains_key("invite"));
    assert!(!obj.contains_key("token"));
    assert!(!obj.contains_key("invite_token"));
}
