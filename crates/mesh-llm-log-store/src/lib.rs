//! SQLite persistence for mesh-llm canonical logging pipeline.

#[cfg(test)]
mod artifacts_tests;
#[cfg(test)]
mod tests;

mod artifacts;
mod cursor;
mod error;
mod migrations;
mod repositories;
mod store;

// Re-export primary types at crate root.
pub use artifacts::{ArtifactContent, ArtifactFileStore, ArtifactStatus, ArtifactWriteReceipt};
pub use cursor::{decode_cursor, encode_cursor};
pub use error::LogStoreError;
pub use store::{Clock, LogStore, SystemClock as RealClock};
