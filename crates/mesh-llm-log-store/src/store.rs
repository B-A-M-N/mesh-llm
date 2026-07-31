//! LogStore owning the SQLite connection and lifecycle.

use crate::error::LogStoreError;
use rusqlite::{Connection, Transaction};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Clock abstraction for deterministic timestamps in tests.
pub trait Clock: Send + Sync {
    fn now(&self) -> String;
}

#[derive(Debug, Clone)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> String {
        let dt = chrono::Utc::now();
        format!("{}", dt.format("%Y-%m-%dT%H:%M:%SZ"))
    }
}

pub struct LogStore {
    conn: Mutex<Connection>,
    clock: std::sync::Arc<dyn Clock>,
    #[cfg_attr(not(test), allow(unused))]
    db_path: PathBuf,
}

impl LogStore {
    pub fn open(
        root_path: impl AsRef<Path>,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Result<Self, LogStoreError> {
        let root = root_path.as_ref();
        std::fs::create_dir_all(root)?;

        let db_path = root.join("log_store.db");
        let conn = Connection::open(&db_path).map_err(|e| {
            LogStoreError::IoError(std::io::Error::other(format!("sqlite open: {}", e)))
        })?;

        let pragmas = "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;
            PRAGMA busy_timeout = 30000;
        ";
        conn.execute_batch(pragmas).map_err(LogStoreError::Sqlite)?;

        crate::migrations::apply_migrations(&conn)
            .map_err(|e| LogStoreError::MigrationFailed(e.to_string()))?;

        Ok(Self {
            conn: Mutex::new(conn),
            clock,
            db_path,
        })
    }

    pub fn reopen_at(
        root_path: impl AsRef<Path>,
        clock: std::sync::Arc<dyn Clock>,
    ) -> Result<Self, LogStoreError> {
        Self::open(root_path, clock)
    }

    pub fn txn<T>(
        &self,
        f: impl FnOnce(&Transaction) -> Result<T, LogStoreError>,
    ) -> Result<T, LogStoreError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| LogStoreError::Sqlite(rusqlite::Error::ExecuteReturnedResults))?;

        let tx = conn.transaction().map_err(LogStoreError::Sqlite)?;
        let result = f(&tx);
        if result.is_ok() {
            tx.commit().map_err(LogStoreError::Sqlite)?;
        }
        result
    }

    pub fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().expect("connection mutex poisoned")
    }

    pub fn now(&self) -> String {
        self.clock.now()
    }

    #[cfg(test)]
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn schema_version(&self) -> u32 {
        self.conn()
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0) as u32
    }

    #[cfg(test)]
    pub fn reopen(&self, clock: std::sync::Arc<dyn Clock>) -> Result<Self, LogStoreError> {
        let parent = self.db_path.parent().ok_or_else(|| {
            LogStoreError::IoError(std::io::Error::other("no parent dir for db path"))
        })?;

        Self::open(parent, clock)
    }
}
