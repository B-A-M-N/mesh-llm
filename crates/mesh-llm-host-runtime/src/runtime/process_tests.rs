//! Process-level tests for the logging service.
//! Covers startup cleanup idempotency, scheduled cleanup, artifact cascades, restart durability, and bounded shutdown.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use mesh_llm_log_store::{ArtifactFileStore, Clock, LogStoreError};
use tempfile::tempdir;
use tokio::time::timeout;

use crate::logging::foundation::LoggingFoundation;
use crate::logging::{Clock as ServiceClock, LoggingService, ServiceConfig, StoreBackedSink};

struct TestClock {
    counter: Arc<AtomicU64>,
}

impl TestClock {
    fn new() -> Self {
        Self {
            counter: Arc::new(AtomicU64::new(0)),
        }
    }

    fn advance(&self, seconds: u64) {
        self.counter.fetch_add(seconds, Ordering::Relaxed);
    }
}

impl Clock for TestClock {
    fn now(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("2025-01-01T00:00:{:02}Z", (n % 60) as u32)
    }
}

struct TestServiceClock(Arc<TestClock>);

impl ServiceClock for TestServiceClock {
    fn now(&self) -> String {
        self.0.now()
    }
}

fn temp_logging_root() -> tempfile::TempDir {
    tempdir().expect("create temp dir")
}

fn create_foundation(root: &std::path::Path) -> LoggingFoundation {
    LoggingFoundation::init(true, Some(&root.to_path_buf()))
}

type SharedService = Arc<tokio::sync::Mutex<LoggingService>>;

async fn create_service_with_clock(
    store_dir: std::path::PathBuf,
    artifact_dir: std::path::PathBuf,
    clock: Arc<TestClock>,
) -> Result<(SharedService, Arc<ArtifactFileStore>), LogStoreError> {
    let sink = StoreBackedSink::open_with_clock(store_dir, artifact_dir, clock.clone())?;
    let artifact_store = sink.artifact_store().clone();
    let service = LoggingService::new(
        ServiceConfig::default(),
        Arc::new(sink),
        Box::new(TestServiceClock(clock)),
    );
    Ok((Arc::new(tokio::sync::Mutex::new(service)), artifact_store))
}

async fn spawn_service(service: &SharedService) -> bool {
    let svc = Arc::clone(service);
    tokio::task::spawn_blocking(move || {
        // try_lock is safe here because we're in a dedicated blocking thread with no other holder.
        let inner = (*svc).try_lock().expect("lock service for spawn");
        inner.spawn()
    })
    .await
    .expect("spawn_blocking failed")
}

async fn shutdown_service(service: &SharedService) {
    let inner = service.lock().await;
    let _ = timeout(Duration::from_secs(5), inner.shutdown()).await;
}

#[tokio::test]
async fn startup_cleanup_idempotent() {
    let root = temp_logging_root();
    let foundation = create_foundation(root.path());
    assert!(foundation.is_healthy());

    let store_dir = foundation.store_dir().to_path_buf();
    let artifact_dir = foundation.artifact_dir().to_path_buf();

    let clock1 = Arc::new(TestClock::new());
    let (service1, _artifact_store1) =
        create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock1)
            .await
            .expect("create service");
    spawn_service(&service1).await;

    assert!(store_dir.exists());
    assert!(artifact_dir.exists());

    drop(service1);

    let clock2 = Arc::new(TestClock::new());
    let (service2, _artifact_store2) =
        create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock2)
            .await
            .expect("create service second time");
    spawn_service(&service2).await;

    assert!(store_dir.exists());
    assert!(artifact_dir.exists());

    shutdown_service(&service2).await;
}

#[tokio::test]
async fn scheduled_cleanup_uses_injected_time() {
    let root = temp_logging_root();
    let foundation = create_foundation(root.path());
    assert!(foundation.is_healthy());

    let store_dir = foundation.store_dir().to_path_buf();
    let artifact_dir = foundation.artifact_dir().to_path_buf();

    let clock = Arc::new(TestClock::new());
    let (service, artifact_store) =
        create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock.clone())
            .await
            .expect("create service");
    spawn_service(&service).await;

    let log_store = artifact_store.store_ref();

    clock.advance(7200);
    let old_time = clock.now(); // 00:00:00 (7200%60=0), counter now 7201
    log_store
        .insert_summary(
            "req-old-1",
            None,
            None,
            None,
            None,
            &old_time,
            None,
            None,
            None,
        )
        .expect("insert old summary");
    log_store
        .insert_lifecycle_event("req-old-1", "evt-1", r#"{"type":"started"}"#, &old_time)
        .expect("insert old event");

    let cutoff = clock.now(); // 00:00:01 (7201%60=1), counter now 7202
    clock.advance(3600); // counter 10802
    let recent_time = clock.now(); // 00:00:02 (10802%60=2), counter 10803
    log_store
        .insert_summary(
            "req-recent-1",
            None,
            None,
            None,
            None,
            &recent_time,
            None,
            None,
            None,
        )
        .expect("insert recent summary");
    let (deleted_count, artifact_ids) = log_store
        .cascade_cleanup_before(&cutoff)
        .expect("cascade cleanup");

    assert!(deleted_count > 0, "expected some deletions");
    assert!(
        artifact_ids.is_empty(),
        "no artifacts to delete in this test"
    );

    let recent = log_store.get_summary("req-recent-1").expect("get recent");
    assert!(recent.is_some(), "recent summary should exist");

    let old = log_store.get_summary("req-old-1").expect("get old");
    assert!(old.is_none(), "old summary should be deleted");

    shutdown_service(&service).await;
}

#[tokio::test]
async fn artifact_cascades_preserve_references() {
    let root = temp_logging_root();
    let foundation = create_foundation(root.path());
    assert!(foundation.is_healthy());

    let store_dir = foundation.store_dir().to_path_buf();
    let artifact_dir = foundation.artifact_dir().to_path_buf();

    let clock = Arc::new(TestClock::new());
    let (service, artifact_store) =
        create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock.clone())
            .await
            .expect("create service");
    spawn_service(&service).await;

    let log_store = artifact_store.store_ref();

    // req-referenced: OLD summary + a RECENT lifecycle event keeps it alive. Artifact preserved.
    clock.advance(7200);
    let old_time = clock.now(); // 00:00:00, counter now 7201
    let cutoff = clock.now(); // 00:00:01, counter now 7202
    clock.advance(3600); // counter 10802
    let recent_time = clock.now(); // 00:00:02, counter 10803

    log_store
        .insert_summary(
            "req-referenced",
            None,
            None,
            None,
            None,
            &old_time,
            None,
            None,
            None,
        )
        .expect("insert referenced summary");
    log_store
        .insert_lifecycle_event(
            "req-referenced",
            "evt-recent",
            r#"{"type":"admitted"}"#,
            &recent_time,
        )
        .expect("insert recent event");

    // req-orphaned: OLD summary, NO lifecycle events -> orphan-deleted; artifact cascades.
    log_store
        .insert_summary(
            "req-orphaned",
            None,
            None,
            None,
            None,
            &old_time,
            None,
            None,
            None,
        )
        .expect("insert orphaned summary");

    let content = b"test artifact content";
    artifact_store
        .write_artifact(
            "art-referenced",
            "req-referenced",
            "request",
            &old_time,
            content,
            Some("text/plain"),
            1,
            false,
            false,
            1024,
            10240,
        )
        .expect("write referenced artifact");
    artifact_store
        .write_artifact(
            "art-orphaned",
            "req-orphaned",
            "response",
            &old_time,
            content,
            Some("text/plain"),
            1,
            false,
            false,
            1024,
            10240,
        )
        .expect("write orphaned artifact");

    let (deleted_count, artifact_ids) = log_store
        .cascade_cleanup_before(&cutoff)
        .expect("cascade cleanup");

    assert!(deleted_count > 0, "expected deletions");
    assert!(
        artifact_ids.contains(&"art-orphaned".to_string()),
        "orphaned artifact should be in deletion list"
    );
    assert!(
        !artifact_ids.contains(&"art-referenced".to_string()),
        "referenced artifact should NOT be in deletion list"
    );

    artifact_store.delete_artifact_files(&artifact_ids);

    let referenced_path = artifact_dir.join("req-referenced").join("art-referenced");
    assert!(
        referenced_path.exists(),
        "referenced artifact file should exist"
    );

    let orphaned_path = artifact_dir.join("req-orphaned").join("art-orphaned");
    assert!(
        !orphaned_path.exists(),
        "orphaned artifact file should be deleted"
    );

    shutdown_service(&service).await;
}

#[tokio::test]
async fn restart_keeps_durable_summaries() {
    let root = temp_logging_root();
    let foundation = create_foundation(root.path());
    assert!(foundation.is_healthy());

    let store_dir = foundation.store_dir().to_path_buf();
    let artifact_dir = foundation.artifact_dir().to_path_buf();

    let clock = Arc::new(TestClock::new());
    let time1 = clock.now();

    {
        let (service, _artifact_store) =
            create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock.clone())
                .await
                .expect("create service");
        spawn_service(&service).await;

        let log_store = _artifact_store.store_ref();
        log_store
            .insert_summary(
                "req-persistent",
                None,
                None,
                None,
                None,
                &time1,
                None,
                None,
                None,
            )
            .expect("insert summary");
        log_store
            .insert_lifecycle_event("req-persistent", "evt-1", r#"{"type":"completed"}"#, &time1)
            .expect("insert event");

        shutdown_service(&service).await;
    }

    {
        let (service, _artifact_store) =
            create_service_with_clock(store_dir.clone(), artifact_dir.clone(), clock.clone())
                .await
                .expect("create service on restart");
        spawn_service(&service).await;

        let log_store = _artifact_store.store_ref();

        let summary = log_store
            .get_summary("req-persistent")
            .expect("get summary");
        assert!(summary.is_some(), "summary should persist across restart");
        assert_eq!(summary.unwrap().request_id, "req-persistent");

        let events = log_store
            .list_events_for_summary("req-persistent")
            .expect("list events");
        assert_eq!(events.len(), 1, "event should persist across restart");
        assert_eq!(events[0].event_id, "evt-1");

        shutdown_service(&service).await;
    }
}

#[tokio::test]
async fn bounded_shutdown_drain() {
    let root = temp_logging_root();
    let foundation = create_foundation(root.path());
    assert!(foundation.is_healthy());

    let store_dir = foundation.store_dir().to_path_buf();
    let artifact_dir = foundation.artifact_dir().to_path_buf();

    let (service, _artifact_store) = create_service_with_clock(
        store_dir.clone(),
        artifact_dir.clone(),
        Arc::new(TestClock::new()),
    )
    .await
    .expect("create service");
    spawn_service(&service).await;

    use mesh_llm_events::logging::identifiers::RequestId;
    use mesh_llm_events::logging::replay::ReplayChannel;

    {
        let svc = service.lock().await;

        for i in 0..100 {
            let req_id = RequestId::new();
            let payload = format!(r#"{{"seq":{}}}"#, i);
            svc.enqueue_event(req_id, ReplayChannel::Requests, payload)
                .expect("enqueue");
        }
    }

    shutdown_service(&service).await;
}

#[tokio::test]
async fn logging_foundation_fail_open_on_unwritable_root() {
    let root = temp_logging_root();
    let unwritable = root.path().join("unwritable");
    std::fs::create_dir_all(&unwritable).expect("create dir");

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&unwritable).unwrap().permissions();
        perms.set_mode(0o000);
        std::fs::set_permissions(&unwritable, perms).expect("set permissions");

        let foundation = LoggingFoundation::init(true, Some(&unwritable));
        assert!(
            !foundation.is_healthy(),
            "should fail open on unwritable root"
        );

        let mut perms = std::fs::metadata(&unwritable).unwrap().permissions();
        perms.set_mode(0o700);
        std::fs::set_permissions(&unwritable, perms).ok();
    }

    #[cfg(not(unix))]
    {
        let foundation = LoggingFoundation::init(true, Some(&unwritable));
        assert!(foundation.is_healthy());
    }
}
