use super::logs_api_seed::{ACTIVE_REQUEST_ID, COMPLETED_REQUEST_ID, seed_log_api_store};
use super::*;

#[derive(Clone)]
struct LogApiClock {
    tick: Arc<std::sync::atomic::AtomicU64>,
}

impl mesh_llm_log_store::Clock for LogApiClock {
    fn now(&self) -> String {
        self.timestamp()
    }
}

impl crate::logging::Clock for LogApiClock {
    fn now(&self) -> String {
        self.timestamp()
    }
}

impl LogApiClock {
    fn timestamp(&self) -> String {
        let second = self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("2026-08-01T00:00:{second:02}Z")
    }
}

pub(super) struct LogApiFixture {
    pub(super) state: MeshApi,
    pub(super) artifact_store: Arc<mesh_llm_log_store::ArtifactFileStore>,
    _temp: tempfile::TempDir,
}

pub(super) async fn build_log_api_fixture() -> LogApiFixture {
    build_log_api_fixture_with_service_config(crate::logging::ServiceConfig::default()).await
}

pub(super) async fn build_log_api_fixture_with_service_config(
    service_config: crate::logging::ServiceConfig,
) -> LogApiFixture {
    let temp = tempfile::tempdir().expect("create log API root");
    let (state, artifact_store) =
        build_log_api_state_with_service_config(temp.path(), service_config).await;
    LogApiFixture {
        state,
        artifact_store,
        _temp: temp,
    }
}

pub(super) async fn build_log_api_state(
    root: &std::path::Path,
) -> (MeshApi, Arc<mesh_llm_log_store::ArtifactFileStore>) {
    build_log_api_state_with_service_config(root, crate::logging::ServiceConfig::default()).await
}

pub(super) async fn build_log_api_state_with_service_config(
    root: &std::path::Path,
    service_config: crate::logging::ServiceConfig,
) -> (MeshApi, Arc<mesh_llm_log_store::ArtifactFileStore>) {
    let clock = LogApiClock {
        tick: Arc::new(std::sync::atomic::AtomicU64::new(30)),
    };
    let sink = crate::logging::StoreBackedSink::open_with_clock(
        root.join("store"),
        root.join("artifacts"),
        Arc::new(clock.clone()),
    )
    .expect("open log API store");
    let artifact_store = sink.artifact_store().clone();
    let service = Arc::new(crate::logging::LoggingService::new(
        service_config,
        Arc::new(sink),
        Some(artifact_store.clone()),
        Box::new(clock),
    ));
    if artifact_store
        .store_ref()
        .query_request(COMPLETED_REQUEST_ID)
        .expect("query seed state")
        .is_none()
    {
        seed_log_api_store(&service, &artifact_store);
    } else {
        let active_uuid = uuid::Uuid::parse_str(ACTIVE_REQUEST_ID).expect("active request UUID");
        service.register_request(mesh_llm_events::logging::identifiers::RequestId::from(
            active_uuid,
        ));
    }
    let state = build_test_mesh_api().await;
    state.inner.lock().await.logging_service = Some(service);
    (state, artifact_store)
}

pub(super) async fn log_api_get(state: MeshApi, path: &str) -> String {
    let (addr, server) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
    )
    .await;
    server.await.unwrap().unwrap();
    response
}
