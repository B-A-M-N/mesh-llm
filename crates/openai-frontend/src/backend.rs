use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use async_trait::async_trait;
use futures_core::Stream;
use tokio::sync::Notify;

use crate::{
    chat::{ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse},
    completions::{CompletionChunk, CompletionRequest, CompletionResponse},
    errors::OpenAiError,
    models::ModelObject,
};

pub type ChatCompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<ChatCompletionChunk>> + Send + 'static>>;
pub type CompletionStream =
    Pin<Box<dyn Stream<Item = OpenAiResult<CompletionChunk>> + Send + 'static>>;

pub type OpenAiResult<T> = Result<T, OpenAiError>;

#[derive(Debug, Default)]
struct CancellationState {
    cancelled: AtomicBool,
    notify: Notify,
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if !self.state.cancelled.swap(true, Ordering::AcqRel) {
            self.state.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.state.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiRequestContext {
    cancellation: CancellationToken,
    request_id: Option<String>,
    started_at: Instant,
    stream_usage_observation: bool,
    trusted_agent_session: bool,
}

impl OpenAiRequestContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_request_id(request_id: impl Into<String>) -> Self {
        Self {
            request_id: Some(request_id.into()),
            ..Self::default()
        }
    }

    /// Enables internal stream-usage observation for an HTTP adapter. This is
    /// deliberately separate from the client's `include_usage` wire option so
    /// direct backend consumers retain their existing stream semantics.
    pub fn with_stream_usage_observation(mut self) -> Self {
        self.stream_usage_observation = true;
        self
    }

    pub fn observes_stream_usage(&self) -> bool {
        self.stream_usage_observation
    }

    pub(crate) fn with_trusted_agent_session(mut self) -> Self {
        self.trusted_agent_session = true;
        self
    }

    /// Returns whether the request entered through the configured trusted
    /// agent-session header. Direct backend calls cannot set this marker.
    pub fn has_trusted_agent_session(&self) -> bool {
        self.trusted_agent_session
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    pub fn cancellation_token(&self) -> CancellationToken {
        self.cancellation.clone()
    }

    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }
}

impl Default for OpenAiRequestContext {
    fn default() -> Self {
        Self {
            cancellation: CancellationToken::new(),
            request_id: None,
            started_at: Instant::now(),
            stream_usage_observation: false,
            trusted_agent_session: false,
        }
    }
}

#[async_trait]
pub trait OpenAiBackend: Send + Sync + 'static {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>>;

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse>;

    async fn chat_completion_with_context(
        &self,
        request: ChatCompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_completion(request).await
    }

    async fn chat_completion_stream(
        &self,
        request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream>;

    async fn completion(&self, _request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        Err(OpenAiError::unsupported(
            "/v1/completions is not supported by this backend",
        ))
    }

    async fn completion_with_context(
        &self,
        request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        self.completion(request).await
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        Err(OpenAiError::unsupported(
            "/v1/completions streaming is not supported by this backend",
        ))
    }
}

pub(crate) type SharedBackend = Arc<dyn OpenAiBackend>;
