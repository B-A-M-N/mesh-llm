//! Typed error types for log-store operations.

use std::fmt;

#[derive(Debug)]
pub enum LogStoreError {
    /// SQLite-level failure (connection, schema, etc.)
    Sqlite(rusqlite::Error),

    /// Schema migration failed to apply cleanly.
    MigrationFailed(String),

    /// Cursor decode/encode error.
    CursorMalformed(String),

    /// Insert failed due to a conflict.
    InsertFailed(String),

    /// Duplicate terminal event for summary + event_type pair.
    DuplicateTerminalEvent {
        summary_id: String,
        event_type: String,
    },

    /// Entity already exists (unique constraint).
    AlreadyExists { entity: String },

    /// General query failure.
    QueryFailed(String),

    /// I/O error on the store path.
    IoError(std::io::Error),
}

impl fmt::Display for LogStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sqlite(e) => write!(f, "sqlite error: {}", e),
            Self::MigrationFailed(msg) => write!(f, "migration failed: {}", msg),
            Self::CursorMalformed(msg) => write!(f, "cursor malformed: {}", msg),
            Self::InsertFailed(msg) => write!(f, "insert failed: {}", msg),
            Self::DuplicateTerminalEvent {
                summary_id,
                event_type,
            } => {
                write!(
                    f,
                    "duplicate terminal event for summary={} type={}",
                    summary_id, event_type
                )
            }
            Self::AlreadyExists { entity } => write!(f, "{} already exists", entity),
            Self::QueryFailed(msg) => write!(f, "query failed: {}", msg),
            Self::IoError(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for LogStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sqlite(e) => Some(e),
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<rusqlite::Error> for LogStoreError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sqlite(e)
    }
}

impl From<std::io::Error> for LogStoreError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}
