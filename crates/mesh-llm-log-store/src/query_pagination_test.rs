use std::sync::Arc;

use crate::{Clock, LogStore, QuerySort, RequestQuery};

#[derive(Debug)]
struct FixedClock;

impl Clock for FixedClock {
    fn now(&self) -> String {
        "2026-08-01T00:00:00Z".to_string()
    }
}

fn insert_request(store: &LogStore, request_id: &str, created_at: &str) {
    store
        .insert_summary(
            request_id, None, None, None, None, created_at, None, None, None,
        )
        .expect("insert paginated request");
}

fn descending_page(cursor: Option<String>) -> RequestQuery {
    RequestQuery {
        limit: 2,
        cursor,
        from: None,
        to: None,
        route: None,
        model: None,
        provider: None,
        engine: None,
        status_code: None,
        outcome: None,
        sort: QuerySort::Descending,
    }
}

#[test]
fn request_cursor_survives_concurrent_inserts_and_reopen_without_skips() {
    let temp = tempfile::tempdir().expect("create pagination store root");
    let clock: Arc<dyn Clock> = Arc::new(FixedClock);
    let store = LogStore::open(temp.path(), clock.clone()).expect("open pagination store");
    for (request_id, created_at) in [
        ("request-a", "2026-08-01T00:00:01Z"),
        ("request-b", "2026-08-01T00:00:02Z"),
        ("request-c", "2026-08-01T00:00:03Z"),
        ("request-d", "2026-08-01T00:00:04Z"),
    ] {
        insert_request(&store, request_id, created_at);
    }

    let first = store
        .query_requests(&descending_page(None))
        .expect("query first page");
    let writer = LogStore::reopen_at(temp.path(), clock.clone()).expect("open concurrent writer");
    insert_request(&writer, "inserted-newer", "2026-08-01T00:00:05Z");
    insert_request(&writer, "request-bb", "2026-08-01T00:00:02Z");
    drop(writer);
    drop(store);

    let reopened = LogStore::reopen_at(temp.path(), clock).expect("reopen pagination store");
    let mut observed = first
        .items
        .into_iter()
        .map(|record| record.request_id)
        .collect::<Vec<_>>();
    let mut next_cursor = first.next_cursor;
    while let Some(cursor) = next_cursor {
        let page = reopened
            .query_requests(&descending_page(Some(cursor)))
            .expect("continue page after reopen");
        observed.extend(page.items.into_iter().map(|record| record.request_id));
        next_cursor = page.next_cursor;
    }

    assert_eq!(
        observed,
        [
            "request-d",
            "request-c",
            "request-bb",
            "request-b",
            "request-a",
        ]
    );
}
