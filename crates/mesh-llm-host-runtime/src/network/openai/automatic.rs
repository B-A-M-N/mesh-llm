//! The automatic routing directive and its serving modes.
//!
//! Mesh exposes exactly one automatic directive. A client that does not want
//! to name a model sends [`DIRECTIVE`] and the mesh decides how to serve the
//! request as well as it can. `auto` is accepted as a deprecated alias so
//! existing OpenAI-compatible clients keep working.
//!
//! The directive resolves to one of two modes. Both are first-class: neither
//! is a failure path.
//!
//! * [`ServingMode::Committee`] — fan out to a Mixture-of-Agents committee.
//!   Chosen when the request permits it and the mesh can actually field one.
//! * [`ServingMode::SingleModel`] — serve from one capability-selected model.
//!   Chosen when the request states something a committee cannot honour.
//!
//! Mode selection reads *declarations the client made*, not guesses about
//! intent:
//!
//! * **Media.** MoA aggregation compares and synthesises drafts as strings, so
//!   a committee has no defined semantics for image or audio input — and the
//!   text-extraction step drops non-text content blocks outright. A media
//!   request must reach a model whose runtime advertises the modality.
//! * **Streaming.** Committee workers are called non-streaming because the
//!   arbiter needs complete drafts to detect divergence, so committee "streams"
//!   are synthesised after the fact. A client asking for `stream: true` gets a
//!   single model, which streams tokens for real.
//!
//! Committee-plus-streaming is deliberately not reachable by sending
//! `stream: true`; it would need its own opt-in.

use serde_json::Value;

use crate::network::router;

/// The one automatic routing directive clients should send.
pub(crate) const DIRECTIVE: &str = mesh_mixture_of_agents::VIRTUAL_MODEL_NAME;

/// Accepted spelling of [`DIRECTIVE`] retained for compatibility.
///
/// Historically `auto` selected a single "good" model while `mesh` convened a
/// committee. Those were never real alternatives — the committee path already
/// served a single model whenever one was all the mesh could field — so the
/// two names described one intent and are now one directive.
pub(crate) const DEPRECATED_ALIAS: &str = "auto";

/// How the mesh will serve a request that used the automatic directive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServingMode {
    /// Fan out to a committee and aggregate the drafts.
    Committee,
    /// Serve from a single capability-selected model.
    SingleModel(SingleModelReason),
}

/// Why a request that asked for automatic routing is being served by one model.
///
/// Carried so logs, tests, and the management API can distinguish a deliberate
/// mode choice from a mesh that simply could not field a committee.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SingleModelReason {
    /// The request carries image, audio, or file content.
    MediaInput,
    /// The client asked for a streamed response.
    StreamRequested,
    /// The request named no model, so it never opted into committee serving.
    ModelUnspecified,
}

impl SingleModelReason {
    /// Stable identifier for logs and route observers.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::MediaInput => "media_input",
            Self::StreamRequested => "stream_requested",
            Self::ModelUnspecified => "model_unspecified",
        }
    }
}

/// True when `model` names the automatic directive under any accepted spelling.
///
/// A request with no `model` at all is also automatic, but that is the caller's
/// observation to make — this answers only about a name that was supplied.
pub(crate) fn is_directive(model: &str) -> bool {
    model == DIRECTIVE || model == DEPRECATED_ALIAS
}

/// Warn once per request when a client used the deprecated spelling.
///
/// Logged rather than rejected: the alias keeps working, and the operator needs
/// to know a client is still sending it before it is eventually removed.
pub(crate) fn warn_if_deprecated_alias(model: Option<&str>) {
    if model == Some(DEPRECATED_ALIAS) {
        tracing::warn!(
            "request used deprecated model \"{DEPRECATED_ALIAS}\"; \
             send \"{DIRECTIVE}\" instead (\"{DEPRECATED_ALIAS}\" is an alias \
             and will be removed in a future release)"
        );
    }
}

/// Choose the serving mode for a request that is being routed automatically.
///
/// `model` is the name the client sent, or `None` when it sent no `model` at
/// all. Callers must only reach here once they have established the request is
/// automatic; an explicitly named model never has a mode.
///
/// A request that named no model did not opt into committee serving, so it is
/// served by a single model rather than silently paying committee latency and
/// cost. Naming the directive is the opt-in.
pub(crate) fn serving_mode(model: Option<&str>, body: &Value) -> ServingMode {
    if model.is_none() {
        return ServingMode::SingleModel(SingleModelReason::ModelUnspecified);
    }
    // Media first: a media request that is also streaming still needs a
    // modality-capable model, and reporting the media reason is the more
    // useful diagnostic of the two.
    if router::media_requirements(body).has_media {
        return ServingMode::SingleModel(SingleModelReason::MediaInput);
    }
    if requests_streaming(body) {
        return ServingMode::SingleModel(SingleModelReason::StreamRequested);
    }
    ServingMode::Committee
}

/// True when the body asks for a streamed response.
///
/// Only a literal JSON `true` counts. OpenAI clients send a bool here, and
/// coercing strings or numbers would let an unrelated field silently change
/// routing.
fn requests_streaming(body: &Value) -> bool {
    body.get("stream") == Some(&Value::Bool(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text_request() -> Value {
        json!({
            "model": DIRECTIVE,
            "messages": [{ "role": "user", "content": "hello" }],
        })
    }

    #[test]
    fn both_spellings_are_the_directive() {
        assert!(is_directive(DIRECTIVE));
        assert!(is_directive(DEPRECATED_ALIAS));
    }

    #[test]
    fn a_named_model_is_not_the_directive() {
        assert!(!is_directive("Qwen3-8B"));
        // Guard against prefix/substring matching: these are real model names.
        assert!(!is_directive("mesh-router-8B"));
        assert!(!is_directive("autocoder-3B"));
    }

    #[test]
    fn plain_text_request_convenes_a_committee() {
        assert_eq!(
            serving_mode(Some(DIRECTIVE), &text_request()),
            ServingMode::Committee
        );
    }

    #[test]
    fn model_less_request_takes_a_single_model() {
        // No `model` field means the client never opted into committee
        // serving, so it must not silently pay committee latency and cost.
        let mut body = text_request();
        body.as_object_mut().unwrap().remove("model");
        assert_eq!(
            serving_mode(None, &body),
            ServingMode::SingleModel(SingleModelReason::ModelUnspecified)
        );
    }

    #[test]
    fn deprecated_alias_convenes_a_committee_too() {
        // `auto` and `mesh` are one directive; they must not diverge in mode.
        let body = text_request();
        assert_eq!(
            serving_mode(Some(DEPRECATED_ALIAS), &body),
            serving_mode(Some(DIRECTIVE), &body)
        );
    }

    #[test]
    fn streaming_request_takes_a_single_model() {
        let mut body = text_request();
        body["stream"] = json!(true);
        assert_eq!(
            serving_mode(Some(DIRECTIVE), &body),
            ServingMode::SingleModel(SingleModelReason::StreamRequested)
        );
    }

    #[test]
    fn stream_false_still_convenes_a_committee() {
        let mut body = text_request();
        body["stream"] = json!(false);
        assert_eq!(serving_mode(Some(DIRECTIVE), &body), ServingMode::Committee);
    }

    #[test]
    fn non_bool_stream_does_not_change_routing() {
        // A stringly-typed `stream` is not a streaming declaration; treating it
        // as one would let a malformed field silently disable the committee.
        for value in [json!("true"), json!(1), json!(null)] {
            let mut body = text_request();
            body["stream"] = value;
            assert_eq!(serving_mode(Some(DIRECTIVE), &body), ServingMode::Committee);
        }
    }

    #[test]
    fn image_request_takes_a_single_model() {
        let body = json!({
            "model": DIRECTIVE,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": "what is this?" },
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(Some(DIRECTIVE), &body),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }

    #[test]
    fn audio_request_takes_a_single_model() {
        // Audio-only input has `needs_vision == false`, so gating on vision
        // alone would leave it in the committee and silently drop the audio.
        let body = json!({
            "model": DIRECTIVE,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": "AAAA", "format": "wav" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(Some(DIRECTIVE), &body),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }

    #[test]
    fn media_outranks_streaming() {
        let body = json!({
            "model": DIRECTIVE,
            "stream": true,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                ],
            }],
        });
        assert_eq!(
            serving_mode(Some(DIRECTIVE), &body),
            ServingMode::SingleModel(SingleModelReason::MediaInput)
        );
    }
}
