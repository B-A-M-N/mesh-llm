//! File-backed artifact storage with transactional DB pointers.
//! Artifact content lives on disk; SQLite rows track metadata + checksums.

use crate::error::LogStoreError;
use crate::store::{Clock, LogStore};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Redaction hook type applied before writing artifact content.
type RedactFn = Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

/// Receipt returned after a successful artifact write (no filesystem paths exposed).
#[derive(Debug, Clone)]
pub struct ArtifactWriteReceipt {
    pub artifact_id: String,
    pub bytes: usize,
    pub checksum: String, // lowercase hex sha256 of stored bytes
    pub version: u32,
    pub media_kind: Option<String>,
    pub redacted: bool,
    pub truncated: bool,
}

/// Artifact content returned by read (no filesystem paths exposed).
#[derive(Debug)]
pub struct ArtifactContent {
    pub artifact_id: String,
    pub bytes: Vec<u8>,
    pub checksum: String, // lowercase hex sha256 of stored bytes
    pub version: u32,
    pub media_kind: Option<String>,
    pub redacted: bool,
    pub truncated: bool,
    #[allow(dead_code)]
    pub kind: String,
}

/// Status enum for artifact health checks.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactStatus {
    Ok { checksum: String },
    Missing,
    Corrupt,
}

/// File-backed store paired with a LogStore for transactional pointer rows.
pub struct ArtifactFileStore {
    root: PathBuf, // canonicalised at open time
    #[allow(dead_code)]
    clock: Arc<dyn Clock>, // reserved for stored_at timestamps in future work
    store: LogStore, // shared DB connection (guarded by Mutex inside)
    redact: Option<RedactFn>,
}

// ─── Path helpers ──────────────────────

/// Reject path segments containing / \ .. NUL or standalone ".".
fn sanitize_segment(segment: &str) -> Result<(), LogStoreError> {
    if segment.is_empty() || segment == "." || segment == ".." {
        return Err(LogStoreError::PathUnsafe {
            segment: segment.to_string(),
        });
    }
    for c in segment.chars() {
        match c {
            '/' | '\\' | '\0' => {
                return Err(LogStoreError::PathUnsafe {
                    segment: segment.to_string(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Build the canonical file path for an artifact from request_id + artifact_id.
fn artifact_path(
    root: &Path,
    request_id: &str,
    artifact_id: &str,
) -> Result<PathBuf, LogStoreError> {
    sanitize_segment(request_id)?;
    sanitize_segment(artifact_id)?;

    let p = root.join(request_id).join(artifact_id);
    // Ensure the resolved path stays under root.
    if !p.starts_with(root) {
        return Err(LogStoreError::PathUnsafe {
            segment: artifact_id.to_string(),
        });
    }
    Ok(p)
}

/// Truncate content to byte_limit at a UTF-8 char boundary. Returns (truncated_bytes, was_truncated).
fn truncate_content(content: &[u8], byte_limit: usize) -> (Vec<u8>, bool) {
    if content.len() <= byte_limit {
        return (content.to_vec(), false);
    }

    // UTF-8 safe truncation: find last valid char boundary within limit.
    let mut truncated = &content[..byte_limit];
    while !truncated.is_empty() {
        match std::str::from_utf8(truncated) {
            Ok(_) => break,
            Err(e) => {
                truncated = &truncated[..truncated.len().saturating_sub(e.error_len().unwrap_or(1))]
            }
        }
    }

    (truncated.to_vec(), true)
}

// ─── ArtifactFileStore implementation ──────────────

impl ArtifactFileStore {
    /// Open artifact storage at `artifact_root`, creating dirs with owner-only permissions.
    pub fn open(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: LogStore,
    ) -> Result<Self, LogStoreError> {
        // Create root dir (owner-only).
        fs::create_dir_all(&artifact_root)?;
        set_owner_only_dir(&artifact_root)?;

        let canonical = artifact_root
            .canonicalize()
            .unwrap_or_else(|_| artifact_root.clone());

        let tmp = canonical.join("tmp");
        fs::create_dir_all(&tmp)?;
        set_owner_only_dir(&tmp)?;

        // Windows privacy check (best-effort).
        #[cfg(windows)]
        {
            if !check_windows_privacy(&canonical) {
                return Err(LogStoreError::PrivacyNotGuaranteed);
            }
        }

        let s = Self {
            root: canonical,
            clock,
            store,
            redact: None,
        };

        // Run startup recovery.
        s.recover_startup();

        Ok(s)
    }

    /// Open with a redaction hook applied before every write.
    pub fn open_with_redact(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: LogStore,
        redact_fn: RedactFn,
    ) -> Result<Self, LogStoreError> {
        fs::create_dir_all(&artifact_root)?;
        set_owner_only_dir(&artifact_root)?;

        let canonical = artifact_root
            .canonicalize()
            .unwrap_or_else(|_| artifact_root.clone());

        let tmp = canonical.join("tmp");
        fs::create_dir_all(&tmp)?;
        set_owner_only_dir(&tmp)?;

        #[cfg(windows)]
        {
            if !check_windows_privacy(&canonical) {
                return Err(LogStoreError::PrivacyNotGuaranteed);
            }
        }

        let s = Self {
            root: canonical,
            clock,
            store,
            redact: Some(redact_fn),
        };

        s.recover_startup();
        Ok(s)
    }

    /// Set a redaction hook on an existing store (useful for tests).
    #[cfg(test)]
    pub fn set_redact(&mut self, f: RedactFn) {
        self.redact = Some(f);
    }

    // ─── Write ──────────────

    /// Write artifact content to disk with a transactional DB pointer.
    /// Rejects writes exceeding byte_limit or aggregate_limit before creating any file.
    #[allow(clippy::too_many_arguments)]
    pub fn write_artifact(
        &self,
        artifact_id: &str,
        request_id: &str,
        kind: &str,
        occurred_at: &str,
        content: &[u8],
        media_kind: Option<&str>,
        version: u32,
        redacted_flag: bool,
        truncated_flag: bool,
        byte_limit: usize,
        aggregate_limit: usize,
    ) -> Result<ArtifactWriteReceipt, LogStoreError> {
        // Validate IDs.
        sanitize_segment(artifact_id)?;
        sanitize_segment(request_id)?;

        // Check for existing pointer before any disk work.
        let exists: bool = self
            .store
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_pointers WHERE artifact_id = ?)",
                rusqlite::params![artifact_id],
                |r| r.get::<_, i32>(0),
            )
            .map(|v| v != 0)
            .map_err(LogStoreError::Sqlite)?;

        if exists {
            return Err(LogStoreError::AlreadyExists {
                entity: format!("artifact_pointer {}", artifact_id),
            });
        }

        // Apply redaction hook if configured (before any size check).
        let mut processed = content.to_vec();
        let mut final_redacted = redacted_flag;
        if let Some(ref fn_) = self.redact {
            processed = fn_(&processed);
            final_redacted = true;
        }

        // Truncate to byte_limit if needed (UTF-8 safe). Reject if still over limit after truncation.
        let (stored, was_truncated) = if processed.len() > byte_limit {
            let result = truncate_content(&processed, byte_limit);
            if result.0.len() > byte_limit {
                return Err(LogStoreError::ArtifactLimitExceeded {
                    artifact_id: artifact_id.to_string(),
                    limit_bytes: byte_limit,
                    kind: "byte".to_string(),
                });
            }
            result
        } else {
            (processed.clone(), false)
        };

        let final_truncated = truncated_flag || was_truncated;

        // Check aggregate limit for this request (existing bytes + new bytes).
        let existing_bytes = self.store.sum_artifact_bytes_for_request(request_id)?;
        if existing_bytes + stored.len() as i64 > aggregate_limit as i64 {
            return Err(LogStoreError::ArtifactLimitExceeded {
                artifact_id: artifact_id.to_string(),
                limit_bytes: aggregate_limit,
                kind: "aggregate".to_string(),
            });
        }

        // 5. Compute checksum over final stored bytes.
        let mut hasher = Sha256::new();
        hasher.update(&stored);
        let checksum_hex = hex::encode(hasher.finalize());

        // 6. Atomic write: tmp/<artifact_id>.part → rename to <request_id>/<artifact_id>.
        let final_path = artifact_path(&self.root, request_id, artifact_id)?;
        let parent_dir = final_path.parent().expect("path always has parent");

        // Check symlink on target parent dir.
        if let Ok(meta) = fs::symlink_metadata(parent_dir)
            && meta.file_type().is_symlink()
        {
            return Err(LogStoreError::PathUnsafe {
                segment: request_id.to_string(),
            });
        }

        // Create parent dir (owner-only).
        fs::create_dir_all(parent_dir)?;
        set_owner_only_dir(parent_dir)?;

        let tmp_dir = self.root.join("tmp");
        fs::create_dir_all(&tmp_dir).map_err(|e| {
            LogStoreError::IoError(io::Error::other(format!("create tmp dir: {}", e)))
        })?;

        #[cfg(unix)]
        set_owner_only_dir(&tmp_dir).ok();

        let tmp_path = tmp_dir.join(format!("{}.part", artifact_id));

        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                // Recreate with mode. If file exists from previous crash, overwrite.
                if tmp_path.exists() {
                    fs::remove_file(&tmp_path).ok();
                }

                let mut opts = OpenOptions::new();
                opts.mode(0o600).write(true).create_new(true);
                let mut f = opts.open(&tmp_path).map_err(|e| {
                    LogStoreError::IoError(io::Error::other(format!("open temp: {}", e)))
                })?;
                f.write_all(&stored).map_err(LogStoreError::IoError)?;
                f.sync_all().map_err(LogStoreError::IoError)?;
            }

            #[cfg(not(unix))]
            {
                let mut opts = OpenOptions::new();
                opts.write(true).create_new(true);
                let mut f = opts.open(&tmp_path).map_err(|e| {
                    LogStoreError::IoError(io::Error::other(format!("open temp: {}", e)))
                })?;
                f.write_all(&stored).map_err(LogStoreError::IoError)?;
                f.sync_all().map_err(LogStoreError::IoError)?;
            }
        }

        // Rename atomically.
        fs::rename(&tmp_path, &final_path)
            .map_err(|e| LogStoreError::IoError(io::Error::other(format!("rename: {}", e))))?;

        #[cfg(unix)]
        {
            set_owner_only_file(&final_path)?;
        }

        // On non-unix, best-effort permissions (no-op on Windows with std-only).
        #[cfg(not(unix))]
        {
            let _ = set_owner_only_file(&final_path);
        }

        // Transactional DB INSERT + UPDATE of pointer row. File is already on disk; if txn fails, clean up the file.
        match self.store.txn(|tx| {
            tx.execute(
                "INSERT INTO artifact_pointers \
                 (artifact_id, request_id, occurred_at, kind) VALUES (?, ?, ?, ?)",
                rusqlite::params![artifact_id, request_id, occurred_at, kind],
            )
            .map_err(LogStoreError::Sqlite)?;

            tx.execute(
                "UPDATE artifact_pointers \
                 SET media_kind = ?, checksum = ?, bytes = ?, version = ?, \
                     redacted = ?, truncated = ? \
                 WHERE artifact_id = ?",
                rusqlite::params![
                    media_kind,
                    &checksum_hex,
                    stored.len() as i64,
                    version as i32,
                    final_redacted as i32,
                    final_truncated as i32,
                    artifact_id
                ],
            )
            .map_err(LogStoreError::Sqlite)?;

            Ok(())
        }) {
            Ok(()) => {}
            Err(e) => {
                // Best-effort cleanup: remove the file we just wrote since txn failed.
                let _ = fs::remove_file(&final_path);
                return Err(e);
            }
        }

        Ok(ArtifactWriteReceipt {
            artifact_id: artifact_id.to_string(),
            bytes: stored.len(),
            checksum: checksum_hex,
            version,
            media_kind: media_kind.map(String::from),
            redacted: final_redacted,
            truncated: final_truncated,
        })
    }

    // ─── Read ──────────────

    pub fn read_artifact(&self, artifact_id: &str) -> Result<ArtifactContent, LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = self
            .store
            .get_artifact_pointer(artifact_id)?
            .ok_or_else(|| LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            })?;

        // Check DB flags for missing/corrupt.
        if row.media_kind.is_none() && row.checksum.is_none() && row.bytes == 0 {
            // Pointer exists but file was never written (pre-v2 or recovery gap).
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        }

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;

        if !path.exists() {
            // Mark as missing in DB.
            let _ = self.store.update_artifact_pointer_missing(artifact_id);
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        }

        // Check symlink on file path itself.
        if let Ok(meta) = fs::symlink_metadata(&path)
            && meta.file_type().is_symlink()
        {
            return Err(LogStoreError::PathUnsafe {
                segment: artifact_id.to_string(),
            });
        }

        let data = std::fs::read(&path)
            .map_err(|e| LogStoreError::IoError(io::Error::other(format!("read file: {}", e))))?;

        // Verify checksum.
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let computed = hex::encode(hasher.finalize());

        let expected_checksum = row.checksum.as_deref().unwrap_or("");
        if !expected_checksum.is_empty() && computed != expected_checksum {
            // Mark as corrupt in DB.
            let _ = self.store.update_artifact_pointer_corrupt(artifact_id);
            return Err(LogStoreError::ArtifactCorrupt {
                artifact_id: artifact_id.to_string(),
            });
        }

        Ok(ArtifactContent {
            artifact_id: row.artifact_id,
            bytes: data,
            checksum: computed,
            version: row.version as u32,
            media_kind: row.media_kind,
            redacted: row.redacted,
            truncated: row.truncated,
            kind: row.kind,
        })
    }

    // ─── Delete single artifact ──────────────

    pub fn delete_artifact(&self, artifact_id: &str) -> Result<(), LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = self
            .store
            .get_artifact_pointer(artifact_id)?
            .ok_or_else(|| LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            })?;

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;

        // Delete file + DB row in one transaction.
        self.store.txn(|tx| {
            tx.execute(
                "DELETE FROM artifact_pointers WHERE artifact_id = ?",
                rusqlite::params![artifact_id],
            )
            .map_err(LogStoreError::Sqlite)?;
            Ok(()) as Result<(), LogStoreError>
        })?;

        // Delete file after txn commit (best-effort).
        let _ = fs::remove_file(&path);

        Ok(())
    }

    // ─── Delete all artifacts for a request ──────────────

    pub fn delete_artifacts_for_request(&self, request_id: &str) -> Result<u64, LogStoreError> {
        sanitize_segment(request_id)?;

        let rows = self.store.list_artifact_pointers_for_request(request_id)?;
        let count = rows.len() as u64;

        // Delete files first (best-effort), then DB rows in a transaction.
        for row in &rows {
            let path = artifact_path(&self.root, request_id, &row.artifact_id);
            if let Ok(p) = path {
                let _ = fs::remove_file(&p);
            }
        }

        self.store
            .delete_artifact_pointer_rows_for_request(request_id)?;

        // Clean up empty request directory.
        let req_dir = self.root.join(request_id);
        if req_dir.exists() && is_empty_dir(&req_dir) {
            let _ = fs::remove_dir(&req_dir);
        }

        Ok(count)
    }

    // ─── Status check ──────────────

    pub fn status(&self, artifact_id: &str) -> Result<ArtifactStatus, LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = match self.store.get_artifact_pointer(artifact_id)? {
            Some(r) => r,
            None => return Ok(ArtifactStatus::Missing),
        };

        // If DB flags it as missing or corrupt.
        if row.checksum.is_none() && row.bytes == 0 {
            return Ok(ArtifactStatus::Missing);
        }

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;

        if !path.exists() {
            return Ok(ArtifactStatus::Missing);
        }

        // Verify checksum.
        match std::fs::read(&path) {
            Ok(data) => {
                let mut hasher = Sha256::new();
                hasher.update(&data);
                let computed = hex::encode(hasher.finalize());

                if let Some(ref expected) = row.checksum
                    && !expected.is_empty()
                    && computed != *expected
                {
                    return Ok(ArtifactStatus::Corrupt);
                }

                Ok(ArtifactStatus::Ok { checksum: computed })
            }
            Err(_) => Ok(ArtifactStatus::Missing),
        }
    }

    // ─── Startup recovery ──────────────

    /// Idempotent startup recovery. Called from `open()`.
    pub fn recover_startup(&self) {
        self.cleanup_orphan_temps();
        self.remove_unreferenced_files();
        self.mark_missing_pointers();
    }

    fn cleanup_orphan_temps(&self) {
        let tmp = self.root.join("tmp");
        if !tmp.exists() || !tmp.is_dir() {
            return;
        }

        if let Ok(entries) = fs::read_dir(&tmp) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "part") {
                    let _ = fs::remove_file(&path);
                } else {
                    // Non-.part files in tmp/ are also stale.
                    let _ = fs::remove_file(&path);
                }
            }
        }

        // Remove empty tmp dir if possible (it will be recreated on next write).
        let _ = fs::remove_dir(&tmp);
    }

    fn remove_unreferenced_files(&self) {
        // Scan all files under root/ (excluding tmp/) and check they have a pointer row.
        for entry in walk_top_level_dirs(&self.root) {
            if !entry.is_file() {
                continue;
            }

            let filename = match entry.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };

            // Skip tmp/ contents.
            if let Ok(rel) = entry.strip_prefix(&self.root)
                && rel.starts_with("tmp")
            {
                continue;
            }

            // Check if this artifact_id has a pointer row.
            match self.store.get_artifact_pointer(filename) {
                Ok(Some(_)) => {} // Referenced — keep it.
                _ => {
                    // Unreferenced file — delete it.
                    let _ = fs::remove_file(&entry);

                    // Clean up parent dir if empty.
                    if let Some(parent) = entry.parent() {
                        let _ = clean_empty_dir_up(parent, &self.root);
                    }
                }
            }
        }
    }

    fn mark_missing_pointers(&self) {
        let mut after_cursor: Option<String> = None;

        loop {
            match self
                .store
                .list_artifact_pointers(100, after_cursor.as_deref())
            {
                Ok(page) if page.items.is_empty() => break,
                Ok(page) => {
                    for row in &page.items {
                        let path = artifact_path(&self.root, &row.request_id, &row.artifact_id);
                        if let Ok(p) = path
                            && !p.exists()
                        {
                            let _ = self.store.update_artifact_pointer_missing(&row.artifact_id);

                            if let Some(parent) = p.parent() {
                                let _ = clean_empty_dir_up(parent, &self.root);
                            }
                        }
                    }

                    match page.next_cursor {
                        Some(c) => after_cursor = Some(c),
                        None => break,
                    }
                }
                Err(_) => break, // Stop on error — recovery is best-effort.
            }
        }
    }

    /// Delete artifact files for a list of IDs (called after cascade_cleanup_before commits).
    pub fn delete_artifact_files(&self, artifact_ids: &[String]) {
        for id in artifact_ids {
            if let Ok(Some(row)) = self.store.get_artifact_pointer(id) {
                let path = artifact_path(&self.root, &row.request_id, id);
                if let Ok(p) = path {
                    let _ = fs::remove_file(&p);

                    // Clean up parent dir.
                    if let Some(parent) = p.parent() {
                        let _ = clean_empty_dir_up(parent, &self.root);
                    }
                }
            } else {
                // Row already deleted by cascade — try to find and remove the file anyway.
                self.remove_file_for_id(id);
            }
        }

        // Also check request dirs are empty after deletion.
        let _ = self.cleanup_empty_request_dirs();
    }

    fn remove_file_for_id(&self, artifact_id: &str) {
        // Scan root for a file matching this artifact_id in any request dir.
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries.flatten() {
                if !entry.path().is_dir() || entry.file_name().to_str() == Some("tmp") {
                    continue;
                }

                if let Ok(sub_entries) = fs::read_dir(entry.path()) {
                    for sub_entry in sub_entries.flatten() {
                        if sub_entry.file_name().to_str() == Some(artifact_id) {
                            let _ = fs::remove_file(sub_entry.path());
                            // Clean up empty dir.
                            let _ = clean_empty_dir_up(&entry.path(), &self.root);
                            return;
                        }
                    }
                }
            }
        }
    }

    fn cleanup_empty_request_dirs(&self) -> Result<(), LogStoreError> {
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                if !entry.path().is_dir() || entry.file_name().to_str() == Some("tmp") {
                    continue;
                }

                let req_id = match entry.file_name().to_str() {
                    Some(r) => r.to_owned(),
                    None => continue,
                };

                let count: i64 = self
                    .store
                    .conn()
                    .query_row(
                        "SELECT COUNT(*) FROM artifact_pointers WHERE request_id = ?",
                        rusqlite::params![req_id],
                        |r| r.get::<_, i64>(0),
                    )
                    .map_err(LogStoreError::Sqlite)?;

                if count == 0 && is_empty_dir(&entry.path()) {
                    let _ = fs::remove_dir(entry.path());
                }
            }
        }

        Ok(())
    }

    /// Expose root path for tests only.
    #[cfg(test)]
    pub fn root_path(&self) -> &Path {
        &self.root
    }

    /// Get the LogStore reference for repository operations (production sinks, tests).
    pub fn store_ref(&self) -> &LogStore {
        &self.store
    }
}

// ─── Helpers ──────────────

fn is_empty_dir(path: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(path) {
        entries.count() == 0
    } else {
        false
    }
}

/// Remove `dir` and walk up removing empty parent dirs, stopping at `stop_at`.
fn clean_empty_dir_up(dir: &Path, stop_at: &Path) -> io::Result<()> {
    let mut current = dir.to_path_buf();
    loop {
        if !current.starts_with(stop_at) || current == *stop_at {
            break;
        }

        match fs::remove_dir(&current) {
            Ok(()) => {}     // removed — try parent next.
            Err(_) => break, // not empty or other error — stop walking up.
        }

        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Ok(())
}

/// Walk all files under root (non-recursively into subdirs one level deep).
fn walk_top_level_dirs(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();

    if !root.is_dir() {
        return result;
    }

    // Skip tmp/.
    for entry in fs::read_dir(root).ok().into_iter().flatten() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.file_name().is_some_and(|n| n == "tmp") {
            continue;
        }

        if path.is_file() {
            result.push(path);
            continue;
        }

        // Recurse into subdirectories (request dirs contain artifact files).
        for sub in walk_top_level_dirs(&path) {
            result.push(sub);
        }
    }

    result
}

// ─── Platform permission helpers ──────────────

#[cfg(unix)]
fn set_owner_only_dir(path: &Path) -> Result<(), LogStoreError> {
    let meta = fs::metadata(path)?;
    let mut perms = meta.permissions();
    perms.set_mode(0o700);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(windows)]
fn set_owner_only_dir(_path: &Path) -> Result<(), LogStoreError> {
    // Best-effort on Windows — std has limited ACL APIs.
    Ok(())
}

#[cfg(unix)]
fn set_owner_only_file(path: &Path) -> Result<(), LogStoreError> {
    let meta = fs::metadata(path)?;
    let mut perms = meta.permissions();
    perms.set_mode(0o600);
    fs::set_permissions(path, perms)?;
    Ok(())
}

#[cfg(windows)]
fn set_owner_only_file(_path: &Path) -> Result<(), LogStoreError> {
    // Best-effort on Windows.
    Ok(())
}

/// Best-effort Windows privacy check. Returns true if we can proceed (or it's not a real risk).
#[cfg(windows)]
fn check_windows_privacy(_root: &Path) -> bool {
    // Pure std has no ACL APIs for checking Everyone/Users permissions.
    // Return false to trigger PrivacyNotGuaranteed on Windows in production,
    // but allow it during CI/testing (env var override).
    std::env::var("MESH_LLM_ALLOW_WEAK_PRIVACY").is_ok() || false
}
