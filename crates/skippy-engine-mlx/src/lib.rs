//! MLX (Metal) serving engine for mesh-llm.
//!
//! Serves HF safetensors tensor models over mesh-llm's real OpenAI-compatible
//! frontend (`openai_frontend::OpenAiBackend`), goose-style, on Apple Silicon.
//!
//! All MLX-touching code is gated behind BOTH the `mlx` cargo feature AND
//! `target_os = "macos"`. On any other target, or without the feature, this
//! crate compiles to an empty shell so it never burdens non-Apple builds.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod backend;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod engine;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod stage;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use backend::MlxBackend;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use engine::{ChatTurn, GenerateRequest, MlxEngine, MlxEngineConfig};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use stage::{MlxComputeDtype, MlxStageEngine, MlxStageEngineConfig, MlxWeightQuantization};

/// True when this build actually contains the MLX engine.
pub const fn mlx_available() -> bool {
    cfg!(all(feature = "mlx", target_os = "macos"))
}
