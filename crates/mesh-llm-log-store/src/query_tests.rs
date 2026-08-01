use std::sync::Arc;

use crate::{Clock, LogStore, PageQuery, ProxyQuery, QuerySort, RequestOutcome, RequestQuery};

#[derive(Debug)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> String {
        "2026-08-01T00:00:00Z".to_string()
    }
}

#[test]
fn request_query_applies_all_filters() {
    let temp = tempfile::tempdir().expect("create query store root");
    let store = LogStore::open(temp.path(), Arc::new(FixedClock)).expect("open query store");
    store
        .insert_summary(
            "matching-request",
            Some("model-a"),
            Some("chat"),
            Some("provider-a"),
            Some("engine-a"),
            "2026-08-01T00:00:05Z",
            None,
            None,
            None,
        )
        .expect("insert matching summary");
    store
        .write_terminal_event(
            "matching-request",
            "matching-event",
            r#"{"type":"completed","status_code":201}"#,
            "completed",
            "2026-08-01T00:00:06Z",
        )
        .expect("complete matching summary");
    store
        .conn()
        .execute(
            "UPDATE summaries SET status_code = 201 WHERE request_id = ?",
            ["matching-request"],
        )
        .expect("set status code");

    let page = store
        .query_requests(&RequestQuery {
            limit: 10,
            cursor: None,
            from: Some("2026-08-01T00:00:00Z".to_string()),
            to: Some("2026-08-01T00:01:00Z".to_string()),
            route: Some("chat".to_string()),
            model: Some("model-a".to_string()),
            provider: Some("provider-a".to_string()),
            engine: Some("engine-a".to_string()),
            status_code: Some(201),
            outcome: Some(RequestOutcome::Completed),
            sort: QuerySort::Descending,
        })
        .expect("query requests");

    assert_eq!(page.items.len(), 1);
    assert_eq!(page.items[0].request_id, "matching-request");
}

#[test]
fn request_query_cursor_is_stable_across_same_timestamp_reopen() {
    let temp = tempfile::tempdir().expect("create pagination store root");
    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let store = LogStore::open(temp.path(), clock.clone()).expect("open pagination store");
    for request_id in ["request-a", "request-b", "request-c"] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                "2026-08-01T00:00:05Z",
                None,
                None,
                None,
            )
            .expect("insert paginated summary");
    }
    let first = store
        .query_requests(&RequestQuery {
            limit: 1,
            cursor: None,
            from: None,
            to: None,
            route: None,
            model: None,
            provider: None,
            engine: None,
            status_code: None,
            outcome: None,
            sort: QuerySort::Descending,
        })
        .expect("query first page");
    let cursor = first.next_cursor.expect("first page cursor");
    drop(store);

    let reopened = LogStore::reopen_at(temp.path(), clock).expect("reopen pagination store");
    let second = reopened
        .query_requests(&RequestQuery {
            limit: 1,
            cursor: Some(cursor),
            from: None,
            to: None,
            route: None,
            model: None,
            provider: None,
            engine: None,
            status_code: None,
            outcome: None,
            sort: QuerySort::Descending,
        })
        .expect("query second page after reopen");

    assert_eq!(first.items[0].request_id, "request-c");
    assert_eq!(second.items[0].request_id, "request-b");
}

#[test]
fn related_queries_return_typed_events_and_request_scoped_proxies() {
    let temp = tempfile::tempdir().expect("create related query store root");
    let store = LogStore::open(temp.path(), Arc::new(FixedClock)).expect("open related store");
    for request_id in ["request-a", "request-b"] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                "2026-08-01T00:00:05Z",
                None,
                None,
                None,
            )
            .expect("insert related summary");
    }
    store
        .insert_lifecycle_event(
            "request-a",
            "event-a",
            r#"{"type":"stream_chunk","tokens":3}"#,
            "2026-08-01T00:00:06Z",
        )
        .expect("insert lifecycle event");
    store
        .insert_proxy_record(
            "attempt-a",
            "request-a",
            "2026-08-01T00:00:07Z",
            "local-target",
            Some("provider-a"),
            Some("engine-a"),
            None,
            None,
            Some(200),
            None,
        )
        .expect("insert request-a proxy");
    store
        .insert_proxy_record(
            "attempt-b",
            "request-b",
            "2026-08-01T00:00:08Z",
            "other-target",
            None,
            None,
            None,
            None,
            Some(503),
            None,
        )
        .expect("insert request-b proxy");
    let page = PageQuery {
        limit: 10,
        cursor: None,
        sort: QuerySort::Ascending,
    };

    let events = store
        .query_events("request-a", &page)
        .expect("query lifecycle events");
    let proxies = store
        .query_proxy_records(&ProxyQuery {
            page,
            request_id: Some("request-a".to_string()),
            provider: Some("provider-a".to_string()),
            engine: Some("engine-a".to_string()),
            status_code: Some(200),
        })
        .expect("query proxy records");

    assert_eq!(
        events.items[0].payload_json,
        r#"{"type":"stream_chunk","tokens":3}"#
    );
    assert_eq!(proxies.items.len(), 1);
    assert_eq!(proxies.items[0].attempt_id, "attempt-a");
}
