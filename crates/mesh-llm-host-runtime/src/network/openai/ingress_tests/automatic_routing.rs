//! Mode selection for the automatic routing directive on the host path.
//!
//! These cover the decision `resolve_auto_routed_model` makes *before* any
//! model is contacted: whether a request that asked for automatic routing
//! convenes a committee or is served by one capability-selected model.

use crate::inference::election;
use crate::mesh;
use crate::network::affinity;
use crate::network::openai::automatic;
use crate::network::openai::transport as proxy;
use mesh_llm_events::logging::identifiers::RequestId;

use super::super::ingress::{AutoRouteResolution, resolve_auto_routed_model};

/// A served model with the given capabilities, ready for the media filter.
fn descriptor(model: &str, vision: bool, audio: bool) -> mesh::ServedModelDescriptor {
    use crate::models::CapabilityLevel;
    mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model.to_string(),
            ..Default::default()
        },
        // Runtime-verified, so `supports_*_runtime()` accepts these.
        capabilities_known: true,
        capabilities: crate::models::ModelCapabilities {
            multimodal: vision || audio,
            vision: if vision {
                CapabilityLevel::Supported
            } else {
                CapabilityLevel::None
            },
            audio: if audio {
                CapabilityLevel::Supported
            } else {
                CapabilityLevel::None
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn request_with_body(model: Option<&str>, body: &serde_json::Value) -> proxy::BufferedHttpRequest {
    let body = serde_json::to_vec(body).expect("serialize body");
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes()
    .into_iter()
    .chain(body.iter().copied())
    .collect::<Vec<u8>>();
    proxy::BufferedHttpRequest {
        raw,
        method: "POST".to_owned(),
        path: "/v1/chat/completions".to_owned(),
        client_path: "/v1/chat/completions".to_owned(),
        request_id: RequestId::default(),
        body_json: None,
        body_json_attempted: false,
        body_bytes: None,
        body_len_bytes: body.len(),
        completion_tokens: None,
        stream: None,
        model_name: model.map(str::to_owned),
        request_object_request_ids: Vec::new(),
        response_adapter: proxy::ResponseAdapter::OpenAiChatCompletionsJson,
        correlation_id: None,
    }
}

fn text_body(model: Option<&str>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "messages": [{ "role": "user", "content": "hello" }],
    });
    if let Some(model) = model {
        body["model"] = serde_json::json!(model);
    }
    body
}

fn image_body(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": "what is in this image?" },
                { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
            ],
        }],
    })
}

/// Node serving `models`, each locally callable.
async fn node_serving(models: &[&str]) -> (mesh::Node, election::ModelTargets) {
    let node = mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .expect("test node");
    node.set_hosted_models(models.iter().map(|m| (*m).to_string()).collect())
        .await;
    let mut targets = election::ModelTargets::default();
    for (index, model) in models.iter().enumerate() {
        targets.targets.insert(
            (*model).to_string(),
            vec![election::InferenceTarget::Local(9000 + index as u16)],
        );
    }
    (node, targets)
}

async fn resolve(
    model: Option<&str>,
    body: &serde_json::Value,
    node: &mesh::Node,
    targets: &election::ModelTargets,
    descriptors: &[mesh::ServedModelDescriptor],
) -> AutoRouteResolution {
    let mut request = request_with_body(model, body);
    let affinity = affinity::AffinityRouter::new();
    resolve_auto_routed_model(
        node,
        &mut request,
        targets,
        None,
        descriptors,
        None,
        &affinity,
    )
    .await
}

#[tokio::test]
async fn plain_text_directive_stays_on_the_committee() {
    let (node, targets) = node_serving(&["vision-model", "text-model"]).await;
    let descriptors = vec![
        descriptor("vision-model", true, false),
        descriptor("text-model", false, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &text_body(Some(automatic::DIRECTIVE)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    // The directive must survive resolution so the MoA gateway picks it up.
    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some(automatic::DIRECTIVE)),
        AutoRouteResolution::MediaUnsupported => panic!("text request is not a media failure"),
    }
}

#[tokio::test]
async fn image_request_resolves_to_a_vision_capable_model() {
    // The defect this pins: `model=mesh` with an image used to skip the media
    // filter entirely and reach MoA, whose text extraction drops the image and
    // answers the text half as if no image were sent.
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(
            effective_model.as_deref(),
            Some("vision-model"),
            "an image request must resolve to the vision-capable model, not the directive"
        ),
        AutoRouteResolution::MediaUnsupported => {
            panic!("a vision-capable model is served, so this must not fail")
        }
    }
}

#[tokio::test]
async fn image_request_with_no_capable_model_is_reported_unsupported() {
    // Honest failure beats a confident answer to the text half.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &image_body(automatic::DIRECTIVE),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    assert!(
        matches!(resolution, AutoRouteResolution::MediaUnsupported),
        "no served model can satisfy the image, so the request must be refused"
    );
}

#[tokio::test]
async fn deprecated_alias_behaves_exactly_like_the_directive() {
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let via_alias = resolve(
        Some(automatic::DEPRECATED_ALIAS),
        &text_body(Some(automatic::DEPRECATED_ALIAS)),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    // `auto` is the same directive, so it must also reach the committee rather
    // than resolving to a single model as it did historically.
    match via_alias {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some(automatic::DIRECTIVE)),
        AutoRouteResolution::MediaUnsupported => panic!("text request is not a media failure"),
    }
}

#[tokio::test]
async fn streaming_directive_resolves_to_a_single_model() {
    // A committee cannot stream: workers are called non-streaming and the SSE
    // is synthesised afterwards. A client asking to stream gets one model.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];
    let mut body = text_body(Some(automatic::DIRECTIVE));
    body["stream"] = serde_json::json!(true);

    let resolution = resolve(
        Some(automatic::DIRECTIVE),
        &body,
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => {
            let model = effective_model.expect("a streaming request must resolve to a model");
            assert_ne!(
                model,
                automatic::DIRECTIVE,
                "a streaming request must not stay on the committee"
            );
            assert!(
                model == "text-model" || model == "other-text-model",
                "must resolve to a served model, got {model}"
            );
        }
        AutoRouteResolution::MediaUnsupported => panic!("no media in this request"),
    }
}

#[tokio::test]
async fn model_less_request_resolves_to_a_single_model() {
    // A client that named nothing never opted into committee cost.
    let (node, targets) = node_serving(&["text-model", "other-text-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("other-text-model", false, false),
    ];

    let resolution = resolve(None, &text_body(None), &node, &targets, &descriptors).await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => {
            let model = effective_model.expect("must resolve to a concrete model");
            assert_ne!(
                model,
                automatic::DIRECTIVE,
                "a model-less request must not silently convene a committee"
            );
        }
        AutoRouteResolution::MediaUnsupported => panic!("no media in this request"),
    }
}

#[tokio::test]
async fn an_explicitly_named_model_is_never_reinterpreted() {
    let (node, targets) = node_serving(&["text-model", "vision-model"]).await;
    let descriptors = vec![
        descriptor("text-model", false, false),
        descriptor("vision-model", true, false),
    ];

    let resolution = resolve(
        Some("text-model"),
        &text_body(Some("text-model")),
        &node,
        &targets,
        &descriptors,
    )
    .await;

    match resolution {
        AutoRouteResolution::Continue {
            effective_model, ..
        } => assert_eq!(effective_model.as_deref(), Some("text-model")),
        AutoRouteResolution::MediaUnsupported => panic!("explicit routing is untouched"),
    }
}
