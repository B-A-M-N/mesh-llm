use super::logs_api_fixture::build_log_api_state;
use super::*;

#[tokio::test]
#[ignore = "manual curl evidence harness"]
async fn manual_logs_api_server() {
    let root = std::env::var_os("MESH_LOG_QA_ROOT").expect("MESH_LOG_QA_ROOT");
    let port_file = std::env::var_os("MESH_LOG_QA_PORT_FILE").expect("MESH_LOG_QA_PORT_FILE");
    let requests = std::env::var("MESH_LOG_QA_REQUESTS")
        .expect("MESH_LOG_QA_REQUESTS")
        .parse::<usize>()
        .expect("request count");
    let (state, _artifact_store) = build_log_api_state(std::path::Path::new(&root)).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind manual log API server");
    let address = listener.local_addr().expect("manual server address");
    std::fs::write(port_file, address.port().to_string()).expect("write manual server port");
    for _ in 0..requests {
        let (stream, _) = listener.accept().await.expect("accept manual request");
        handle_request(stream, &state)
            .await
            .expect("handle manual request");
    }
}
