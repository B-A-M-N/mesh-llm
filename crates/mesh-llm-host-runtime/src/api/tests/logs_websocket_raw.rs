use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::time::Duration;

use super::logs_api_fixture::{
    build_log_api_fixture, build_log_api_fixture_with_service_config, log_api_get,
};
use super::logs_api_seed::ACTIVE_REQUEST_ID;
use super::logs_websocket_support::*;

#[tokio::test]
async fn websocket_fans_out_a_live_event_to_each_ready_raw_client() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    let (address, server) = spawn_two_connection_management_server(fixture.state).await;
    let mut first_socket = upgrade_log_stream(address).await;
    let mut second_socket = upgrade_log_stream(address).await;

    publish_request_event(&service, 1);
    subscribe(&mut first_socket, "requests", 0).await;
    subscribe(&mut second_socket, "requests", 0).await;
    assert_eq!(receive_frame(&mut first_socket).await["sequence"], 1);
    assert_eq!(receive_frame(&mut second_socket).await["sequence"], 1);

    publish_request_event(&service, 2);
    let (first, second) = tokio::time::timeout(Duration::from_secs(1), async {
        tokio::join!(
            receive_frame(&mut first_socket),
            receive_frame(&mut second_socket),
        )
    })
    .await
    .expect("both ready clients receive the live event within the bound");

    assert_eq!(first["type"], "event");
    assert_eq!(second["type"], "event");
    assert_eq!(first["sequence"], 2);
    assert_eq!(second["sequence"], 2);

    drop(first_socket);
    drop(second_socket);
    tokio::time::timeout(Duration::from_secs(1), server)
        .await
        .expect("both raw handlers exit within the bound")
        .expect("raw management server task joins")
        .expect("both raw management handlers exit cleanly");
}

#[tokio::test]
async fn websocket_reports_truthful_eviction_recovery_for_every_channel() {
    let fixture = build_log_api_fixture_with_service_config(crate::logging::ServiceConfig {
        queue_capacity: 3,
        registry_config: crate::logging::RegistryConfig::default(),
    })
    .await;
    let service = log_service(&fixture.state).await;

    for sequence in 1..=2 {
        publish_request_event(&service, sequence);
        service
            .enqueue_event(
                active_request_id(),
                ReplayChannel::Operations,
                serde_json::json!({ "sequence": sequence }).to_string(),
            )
            .expect("bounded replay bus accepts operation event");
        service
            .enqueue_event(
                active_request_id(),
                ReplayChannel::System,
                serde_json::json!({ "sequence": sequence }).to_string(),
            )
            .expect("bounded replay bus accepts system event");
    }

    let (mut socket, server) = connect_log_stream(fixture.state.clone()).await;
    let mut requests_recovery_cursor = None;
    for (channel, durable_recovery_available) in
        [("requests", true), ("operations", false), ("system", false)]
    {
        subscribe(&mut socket, channel, 0).await;
        let gap = receive_frame(&mut socket).await;
        assert_eq!(gap["type"], "gap");
        assert_eq!(gap["channel"], channel);
        assert_eq!(gap["fromSequence"], 1);
        assert_eq!(gap["toSequence"], 1);

        match durable_recovery_available {
            true => {
                assert_eq!(gap["recovery"]["available"], true);
                assert_eq!(gap["recovery"]["endpoint"], "/api/logs/requests");
                let cursor = gap["recovery"]["cursor"]
                    .as_str()
                    .expect("request gap provides an opaque REST cursor")
                    .to_owned();
                assert_eq!(
                    cursor,
                    "v1:MjAyNi0wOC0wMVQwMDowMDozMFp8MDAwMDAwMDAtMDAwMC00MDAwLTgwMDAtMDAwMDAwMDAwMDAx"
                );
                requests_recovery_cursor = Some(cursor);
            }
            false => {
                assert_eq!(gap["recovery"]["available"], false);
                assert!(gap["recovery"]["endpoint"].is_null());
                assert!(gap["recovery"]["cursor"].is_null());
            }
        }

        let event = receive_frame(&mut socket).await;
        assert_eq!(event["type"], "event");
        assert_eq!(event["channel"], channel);
        assert_eq!(event["sequence"], 2);
    }
    close_socket(socket, server).await;

    let cursor = requests_recovery_cursor.expect("requests channel records recovery cursor");
    let response = log_api_get(
        fixture.state,
        &format!("/api/logs/requests?cursor={}", urlencoding::encode(&cursor)),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 200 OK"));
}

#[tokio::test]
async fn websocket_frames_exclude_the_privacy_corpus_at_the_service_enqueue_seam() {
    let fixture = build_log_api_fixture().await;
    let service = log_service(&fixture.state).await;
    let (mut socket, server) = connect_log_stream(fixture.state).await;

    publish_request_event(&service, 1);
    subscribe(&mut socket, "requests", 0).await;
    assert_eq!(receive_frame(&mut socket).await["sequence"], 1);

    let forbidden_values = [
        "raw-body-DO-NOT-LOG",
        "prompt-DO-NOT-LOG",
        "/Users/alice/private/model.gguf",
        "query-secret-DO-NOT-LOG",
        "alice:credential-DO-NOT-LOG",
        "token-DO-NOT-LOG",
        "credential-DO-NOT-LOG",
    ];
    let corpus = serde_json::json!({
        "body": forbidden_values[0],
        "prompt": forbidden_values[1],
        "local_path": forbidden_values[2],
        "url": format!("https://{}@logs.invalid/v1?token={}", forbidden_values[4], forbidden_values[3]),
        "userinfo": forbidden_values[4],
        "token": forbidden_values[5],
        "credential": forbidden_values[6],
    })
    .to_string();
    service
        .enqueue_event(active_request_id(), ReplayChannel::Requests, corpus)
        .expect("bounded replay bus accepts privacy corpus");

    let serialized_frame = receive_text_frame(&mut socket).await;
    let frame: serde_json::Value =
        serde_json::from_str(&serialized_frame).expect("server frame is valid JSON");
    assert_eq!(frame["type"], "event");
    assert_eq!(frame["channel"], "requests");
    assert_eq!(frame["sequence"], 2);
    assert_eq!(frame["requestId"], ACTIVE_REQUEST_ID);
    assert!(frame["occurredAt"].as_str().is_some());
    for forbidden in forbidden_values {
        assert!(
            !serialized_frame.contains(forbidden),
            "serialized frame leaked {forbidden}"
        );
    }

    close_socket(socket, server).await;
}
