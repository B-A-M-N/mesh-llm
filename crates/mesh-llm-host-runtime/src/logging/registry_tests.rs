use super::*;
use std::thread;

fn make_entry(id: &str, ts: u8) -> RequestSummaryEntry {
    RequestSummaryEntry {
        request_id: id.to_string(),
        state: "active".into(),
        created_at: format!("2025-01-01T00:00:{ts:02}Z"),
        terminal_at: None,
    }
}

#[test]
fn test_active_to_recent_movement() {
    let registry = RequestRegistry::new(RegistryConfig {
        max_active: 10,
        max_recent: 20,
    });
    let entry = make_entry("req-1", 0);
    registry.register_active(entry.clone());
    assert_eq!(registry.active_count(), 1);
    registry.move_to_recent(entry);
    assert_eq!(registry.active_count(), 0);
    assert_eq!(registry.recent_count(), 1);
}

#[test]
fn test_getters_return_clones_and_missing_is_none() {
    let registry = RequestRegistry::new(RegistryConfig::default());
    let active = make_entry("req-active", 5);
    registry.register_active(active);
    assert_eq!(
        registry.get_active("req-active").unwrap().request_id,
        "req-active"
    );
    let recent = make_entry("req-recent", 6);
    registry.register_active(recent.clone());
    registry.move_to_recent(recent);
    assert_eq!(
        registry.get_recent("req-recent").unwrap().request_id,
        "req-recent"
    );
    assert!(registry.get_active("missing").is_none());
    assert!(registry.get_recent("missing").is_none());
}

#[test]
fn test_active_eviction_when_over_capacity() {
    let registry = RequestRegistry::new(RegistryConfig {
        max_active: 3,
        max_recent: 10,
    });
    for index in 0..5u8 {
        registry.register_active(make_entry(&format!("req-{index}"), index));
    }
    assert_eq!(registry.active_count(), 3);
    assert_eq!(registry.active_evictions.load(Ordering::Relaxed), 2);
    assert!(registry.get_active("req-0").is_none());
    assert!(registry.get_active("req-4").is_some());
}

#[test]
fn test_recent_eviction_when_over_capacity() {
    let registry = RequestRegistry::new(RegistryConfig {
        max_active: 20,
        max_recent: 3,
    });
    for index in 0..5u8 {
        let entry = make_entry(&format!("req-{index}"), index);
        registry.register_active(entry.clone());
        registry.move_to_recent(entry);
    }
    assert_eq!(registry.recent_count(), 3);
    assert_eq!(registry.recent_evictions.load(Ordering::Relaxed), 2);
    assert!(registry.get_recent("req-0").is_none());
    assert!(registry.get_recent("req-4").is_some());
}

#[test]
fn test_eviction_removes_oldest_by_created_at() {
    let registry = RequestRegistry::new(RegistryConfig {
        max_active: 2,
        max_recent: 10,
    });
    registry.register_active(make_entry("z-last", 3));
    registry.register_active(make_entry("a-first", 1));
    registry.register_active(make_entry("m-mid", 2));
    assert!(registry.get_active("a-first").is_none());
}

#[test]
fn test_clear_and_empty_state() {
    let registry = RequestRegistry::new(RegistryConfig::default());
    assert!(registry.is_empty());
    registry.register_active(make_entry("req", 1));
    assert!(!registry.is_empty());
    registry.clear();
    assert!(registry.is_empty());
}

#[test]
fn test_no_leak_under_pressure() {
    let registry = RequestRegistry::new(RegistryConfig {
        max_active: 5,
        max_recent: 10,
    });
    for index in 0..200u8 {
        let entry = make_entry(&format!("req-{index}"), index % 60);
        registry.register_active(entry.clone());
        if index % 3 == 0 {
            registry.move_to_recent(entry);
        }
    }
    assert!(registry.active_count() <= registry.config.max_active);
    assert!(registry.recent_count() <= registry.config.max_recent);
}

#[test]
fn test_concurrent_register_active() {
    let registry = Arc::new(RequestRegistry::new(RegistryConfig {
        max_active: 100,
        max_recent: 200,
    }));
    let mut handles = Vec::new();
    for thread_index in 0..4u8 {
        let registry = Arc::clone(&registry);
        handles.push(thread::spawn(move || {
            for request_index in 0..50u8 {
                registry.register_active(RequestSummaryEntry {
                    request_id: format!("t{thread_index}-req-{request_index}"),
                    state: "active".into(),
                    created_at: format!(
                        "2025-01-01T00:{:02}:{:02}Z",
                        thread_index * 15,
                        request_index % 60
                    ),
                    terminal_at: None,
                });
            }
        }));
    }
    for handle in handles {
        handle.join().expect("thread panicked");
    }
    assert!(registry.active_count() <= registry.config.max_active);
}

#[test]
fn test_config_default_values() {
    let config = RegistryConfig::default();
    assert_eq!(config.max_active, 1024);
    assert_eq!(config.max_recent, 8192);
}

#[test]
fn test_terminal_at_preserved_through_move_to_recent() {
    let registry = RequestRegistry::new(RegistryConfig::default());
    let mut entry = make_entry("req-term", 7);
    registry.register_active(entry.clone());
    entry.state = "completed".into();
    entry.terminal_at = Some("2025-01-01T00:00:15Z".into());
    registry.move_to_recent(entry);
    let recent = registry.get_recent("req-term").unwrap();
    assert_eq!(recent.state, "completed");
    assert_eq!(recent.terminal_at.as_deref(), Some("2025-01-01T00:00:15Z"));
}
