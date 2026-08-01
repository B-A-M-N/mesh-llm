pub(super) const ACTIVE_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000001";
pub(super) const COMPLETED_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000002";
pub(super) const RETRIED_REQUEST_ID: &str = "00000000-0000-4000-8000-000000000003";
pub(super) const REDACTED_ARTIFACT_ID: &str = "00000000-0000-4000-8000-000000000011";
pub(super) const PRIVATE_ARTIFACT_ID: &str = "00000000-0000-4000-8000-000000000012";
pub(super) const MISSING_ARTIFACT_ID: &str = "00000000-0000-4000-8000-000000000013";

pub(super) fn seed_log_api_store(
    service: &crate::logging::LoggingService,
    artifacts: &mesh_llm_log_store::ArtifactFileStore,
) {
    let store = artifacts.store_ref();
    for (request_id, model, route, provider, engine, created_at) in [
        (
            ACTIVE_REQUEST_ID,
            "model-active",
            "responses",
            "provider-active",
            "engine-active",
            "2026-08-01T00:00:30Z",
        ),
        (
            COMPLETED_REQUEST_ID,
            "model-a",
            "chat",
            "provider-a",
            "engine-a",
            "2026-08-01T00:00:20Z",
        ),
        (
            RETRIED_REQUEST_ID,
            "/Users/private/model.gguf?token=secret",
            "responses",
            "provider-b",
            "engine-b",
            "2026-08-01T00:00:20Z",
        ),
    ] {
        store
            .insert_summary(
                request_id,
                Some(model),
                Some(route),
                Some(provider),
                Some(engine),
                created_at,
                None,
                None,
                None,
            )
            .expect("seed summary");
    }
    store
        .write_terminal_event(
            COMPLETED_REQUEST_ID,
            "00000000-0000-4000-8000-000000000021",
            r#"{"type":"completed","status_code":201,"duration_ms":12}"#,
            "completed",
            "2026-08-01T00:00:21Z",
        )
        .expect("complete request");
    store
        .insert_lifecycle_event(
            RETRIED_REQUEST_ID,
            "00000000-0000-4000-8000-000000000022",
            r#"{"type":"attempt_failed","attempt_id":null,"error":"sanitized"}"#,
            "2026-08-01T00:00:22Z",
        )
        .expect("record failed proxy attempt");
    store
        .write_terminal_event(
            RETRIED_REQUEST_ID,
            "00000000-0000-4000-8000-000000000023",
            r#"{"type":"completed","status_code":200,"duration_ms":20}"#,
            "completed",
            "2026-08-01T00:00:23Z",
        )
        .expect("complete retried request");
    store
        .conn()
        .execute(
            "UPDATE summaries SET status_code = 201 WHERE request_id = ?",
            [COMPLETED_REQUEST_ID],
        )
        .expect("set completed status");
    store
        .conn()
        .execute(
            "UPDATE summaries SET status_code = 200 WHERE request_id = ?",
            [RETRIED_REQUEST_ID],
        )
        .expect("set failed status");
    let active_uuid = uuid::Uuid::parse_str(ACTIVE_REQUEST_ID).expect("active request UUID");
    service.register_request(mesh_llm_events::logging::identifiers::RequestId::from(
        active_uuid,
    ));
    seed_log_api_artifacts(artifacts);
    seed_log_api_proxies(store);
}

fn seed_log_api_artifacts(artifacts: &mesh_llm_log_store::ArtifactFileStore) {
    artifacts
        .write_artifact(
            REDACTED_ARTIFACT_ID,
            COMPLETED_REQUEST_ID,
            "response",
            "2026-08-01T00:00:23Z",
            b"redacted-content",
            Some("text/plain"),
            1,
            true,
            false,
            1024,
            4096,
        )
        .expect("write redacted artifact");
    artifacts
        .write_artifact(
            PRIVATE_ARTIFACT_ID,
            COMPLETED_REQUEST_ID,
            "request",
            "2026-08-01T00:00:24Z",
            b"private-content",
            Some("text/plain"),
            1,
            false,
            false,
            1024,
            4096,
        )
        .expect("write private artifact");
    artifacts
        .store_ref()
        .insert_artifact_pointer(
            MISSING_ARTIFACT_ID,
            COMPLETED_REQUEST_ID,
            "2026-08-01T00:00:25Z",
            "trace",
            None,
        )
        .expect("insert missing artifact pointer");
}

fn seed_log_api_proxies(store: &mesh_llm_log_store::LogStore) {
    store
        .insert_proxy_record(
            "00000000-0000-4000-8000-000000000031",
            COMPLETED_REQUEST_ID,
            "2026-08-01T00:00:26Z",
            "http://user:secret@127.0.0.1:9337/private?token=secret",
            Some("provider-a"),
            Some("engine-a"),
            None,
            None,
            Some(201),
            None,
        )
        .expect("insert completed proxy");
    store
        .insert_proxy_record(
            "00000000-0000-4000-8000-000000000032",
            RETRIED_REQUEST_ID,
            "2026-08-01T00:00:27Z",
            "/private/local/path?token=secret",
            Some("https://user:secret@provider.example/private?token=secret"),
            Some("/Users/private/engine?api_key=secret"),
            None,
            None,
            Some(503),
            None,
        )
        .expect("insert failed proxy");
}
