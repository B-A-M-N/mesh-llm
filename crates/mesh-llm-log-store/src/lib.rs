//! SQLite persistence for mesh-llm canonical logging pipeline.

#[cfg(test)]
mod tests;

mod cursor;
mod error;
mod migrations;
mod repositories;
mod store;

// Re-export primary types at crate root.
pub use cursor::{decode_cursor, encode_cursor};
pub use error::LogStoreError;
pub use store::{Clock, LogStore, SystemClock as RealClock};
