//! Artifact storage tests — real tempdir filesystem, no in-memory shortcuts.

use crate::artifacts::{ArtifactFileStore, ArtifactStatus};
use crate::error::LogStoreError;
use crate::store::{Clock as ClockTrait, LogStore};
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug)]
struct TestClock {
    instant: AtomicU64,
}

impl Default for TestClock {
    fn default() -> Self {
        Self {
            instant: AtomicU64::new(0),
        }
    }
}

impl ClockTrait for TestClock {
    fn now(&self) -> String {
        let n = self.instant.fetch_add(1, Ordering::Relaxed);
        format!("2025-01-01T00:00:{:02}Z", n % 60)
    }
}

fn expected_checksum(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

// ════════════════════════════════
// 1. Write-then-read roundtrip
// ════════════════════════════════

#[test]
fn artifact_write_then_read_roundtrip() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    let content = b"hello artifact world";
    let receipt = afs
        .write_artifact(
            "art-1",
            "req-1",
            "log",
            &clock.now(),
            content,
            None::<&str>,
            1,
            false,
            false,
            4096,
            8192,
        )
        .unwrap();

    // Receipt has no filesystem paths.
    assert_eq!(receipt.artifact_id, "art-1");
    assert_eq!(receipt.bytes, content.len());
    assert_eq!(receipt.checksum, expected_checksum(content));
    assert!(!receipt.checksum.contains('/')); // no path chars in checksum

    let art_content = afs.read_artifact("art-1").unwrap();
    assert_eq!(art_content.bytes, content);
    assert_eq!(art_content.checksum, receipt.checksum);
}

// ════════════════════════════════
// 2. Atomic write — no partial final file on failure
// ════════════════════════════════

#[test]
fn artifact_atomic_write_no_partial_final_file() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    // Pre-create a .part file in tmp/ and an incomplete final file.
    let old_part = afs_root.path().join("tmp").join("art-1.part");
    fs::create_dir_all(old_part.parent().unwrap()).unwrap();
    fs::write(&old_part, b"incomplete data").unwrap();

    let fake_final = afs_root.path().join("req-1").join("art-1");
    fs::create_dir_all(fake_final.parent().unwrap()).unwrap();
    fs::write(&fake_final, b"stale partial content").unwrap();

    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Recovery should have removed the orphan .part file.
    assert!(!old_part.exists());

    // Write new artifact — final path only has complete data after rename.
    let content = b"complete new content";
    afs.write_artifact(
        "art-1",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Final file has only the complete new content.
    let final_data = fs::read(&fake_final).unwrap();
    assert_eq!(final_data.as_slice(), content);

    // No .part left behind.
    assert!(!old_part.exists());
}

// ════════════════════════════════
// 3. Redaction applied before write
// ════════════════════════════════

#[test]
fn artifact_redaction_applied_before_write() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let mut afs =
        ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Set redaction hook: upper-case everything.
    afs.set_redact(Arc::new(|data: &[u8]| -> Vec<u8> {
        String::from_utf8_lossy(data).to_uppercase().into_bytes()
    }));

    let secret = b"password=supersecret123";
    let receipt = afs
        .write_artifact(
            "art-secret",
            "req-1",
            "log",
            &clock.now(),
            secret,
            None::<&str>,
            1,
            false,
            false,
            4096,
            8192,
        )
        .unwrap();

    // Stored content should be upper-cased (redacted).
    let art_content = afs.read_artifact("art-secret").unwrap();
    assert_eq!(art_content.bytes.as_slice(), b"PASSWORD=SUPERSECRET123");
    assert!(receipt.redacted);

    // Raw secret is NOT in the stored bytes.
    assert!(!String::from_utf8_lossy(&art_content.bytes).contains("supersecret"));
}

// ════════════════════════════════
// 4. Truncation respects byte_limit (UTF-8 safe)
// ════════════════════════════════

#[test]
fn artifact_truncation_respects_byte_limit() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Multi-byte chars at boundary: "é" is 2 bytes. Content > byte_limit triggers truncation.
    let content = "hello é world this is a long message that exceeds the limit".as_bytes();
    let byte_limit = 20;

    let receipt = afs
        .write_artifact(
            "art-trunc",
            "req-1",
            "log",
            &clock.now(),
            content,
            None::<&str>,
            1,
            false,
            false,
            byte_limit,
            8192,
        )
        .unwrap();

    assert!(receipt.truncated);
    assert_eq!(receipt.bytes, byte_limit);

    // Stored bytes are valid UTF-8 (char-boundary truncation).
    let art_content = afs.read_artifact("art-trunc").unwrap();
    assert!(std::str::from_utf8(&art_content.bytes).is_ok());
}

// ════════════════════════════════
// 5. Individual and aggregate limits rejected without partial files
// ════════════════════════════════

#[test]
fn artifact_individual_and_aggregate_limits_rejected_without_partial_files() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // byte_limit exceeded → truncated to fit, not rejected (truncation always applies).
    let big_content = vec![0u8; 1024];
    {
        let receipt = afs
            .write_artifact(
                "art-big",
                "req-1",
                "log",
                &clock.now(),
                &big_content,
                None::<&str>,
                1,
                false,
                false,
                64,
                8192,
            )
            .unwrap();
        assert_eq!(receipt.bytes, 64); // truncated to byte_limit
        assert!(receipt.truncated);

        // Clean up for aggregate test below.
        afs.delete_artifact("art-big").ok();
    }

    // Aggregate limit exceeded on second write (two writes exceed total).
    let content_a = b"first artifact data";
    afs.write_artifact(
        "art-a",
        "req-1",
        "log",
        &clock.now(),
        content_a,
        None::<&str>,
        1,
        false,
        false,
        4096,
        32, // aggregate_limit=32
    )
    .unwrap();

    let content_b = b"second artifact data";
    let result = afs.write_artifact(
        "art-b",
        "req-1",
        "log",
        &clock.now(),
        content_b,
        None::<&str>,
        1,
        false,
        false,
        4096,
        32, // aggregate_limit=32 already exceeded by art-a (17 bytes) + art-b (20 bytes)
    );

    match result {
        Err(LogStoreError::ArtifactLimitExceeded { kind, .. }) => assert_eq!(kind, "aggregate"),
        other => panic!(
            "expected ArtifactLimitExceeded(aggregate), got: {:?}",
            other
        ),
    }

    // art-b file should not exist.
    let art_b_path = afs_root.path().join("req-1").join("art-b");
    assert!(!art_b_path.exists());
}

// ════════════════════════════════
// 6. Path confinement rejects unsafe segments
// ════════════════════════════════

#[test]
fn artifact_path_confinement_rejects_unsafe_segments() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // ../ traversal in artifact_id.
    let result = afs.write_artifact(
        "../etc/passwd",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // / in artifact_id.
    let result = afs.write_artifact(
        "foo/bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // \ in artifact_id.
    let result = afs.write_artifact(
        "foo\\bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // NUL in artifact_id.
    let result = afs.write_artifact(
        "foo\x00bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // Absolute path in request_id.
    let result = afs.write_artifact(
        "art-ok",
        "/etc/passwd",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // ".." as artifact_id.
    let result = afs.write_artifact(
        "..",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // "." as artifact_id.
    let result = afs.write_artifact(
        ".",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // Verify no file was created outside root.
    let etc = afs_root.path().join("..").join("etc");
    assert!(!etc.exists());
}

// ════════════════════════════════
// 7. Symlink rejected
// ════════════════════════════════

#[test]
fn artifact_symlink_rejected() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert a summary for the real request.
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");

    // Create a symlink inside artifact root pointing outside.
    let target_dir = tempfile::tempdir().expect("symlink target dir");
    let link_path = afs_root.path().join("req-1");
    #[cfg(unix)]
    std::os::unix::fs::symlink(target_dir.path(), &link_path).unwrap();

    #[cfg(unix)]
    {
        let afs =
            ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

        // Attempt to write through the symlinked dir.
        let result = afs.write_artifact(
            "art-sym",
            "req-1",
            "log",
            &clock.now(),
            b"nope",
            None::<&str>,
            1,
            false,
            false,
            4096,
            8192,
        );

        // Should be rejected because the parent dir is a symlink.
        assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

        // No file written to target_dir.
        assert_eq!(fs::read_dir(target_dir.path()).unwrap().count(), 0);
    }
}

// ════════════════════════════════
// 8. Checksum verification: corrupt and missing
// ════════════════════════════════

#[test]
fn artifact_checksum_verification_corrupt_and_missing() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write a valid artifact.
    let content = b"valid content here";
    afs.write_artifact(
        "art-ok",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Status should be Ok.
    let status = afs.status("art-ok").unwrap();
    assert!(matches!(status, ArtifactStatus::Ok { .. }));

    // Corrupt the file on disk (modify one byte).
    let art_path = afs_root.path().join("req-1").join("art-ok");
    let mut data = fs::read(&art_path).unwrap();
    data[0] ^= 0xFF; // flip first byte
    fs::write(&art_path, &data).unwrap();

    // Read should return ArtifactCorrupt.
    match afs.read_artifact("art-ok") {
        Err(LogStoreError::ArtifactCorrupt { .. }) => {} // expected
        other => panic!("expected ArtifactCorrupt, got: {:?}", other),
    }

    // Status reports Corrupt.
    assert_eq!(afs.status("art-ok").unwrap(), ArtifactStatus::Corrupt);

    // Now delete the file entirely (simulate missing).
    fs::remove_file(&art_path).unwrap();

    match afs.read_artifact("art-ok") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected
        other => panic!("expected ArtifactMissing, got: {:?}", other),
    }

    assert_eq!(afs.status("art-ok").unwrap(), ArtifactStatus::Missing);
}

// ════════════════════════════════
// 9. Delete removes file and row
// ════════════════════════════════

#[test]
fn artifact_delete_removes_file_and_row() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write two artifacts for req-1.
    afs.write_artifact(
        "art-d1",
        "req-1",
        "log",
        &clock.now(),
        b"delete me 1",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();
    afs.write_artifact(
        "art-d2",
        "req-1",
        "log",
        &clock.now(),
        b"delete me 2",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Delete single artifact.
    afs.delete_artifact("art-d1").unwrap();

    let art_path = afs_root.path().join("req-1").join("art-d1");
    assert!(!art_path.exists());

    match afs.read_artifact("art-d1") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected — row is gone
        other => panic!("expected ArtifactMissing after delete, got: {:?}", other),
    }

    // art-d2 still exists.
    assert!(afs_root.path().join("req-1").join("art-d2").exists());

    // Delete all artifacts for req-1.
    let count = afs.delete_artifacts_for_request("req-1").unwrap();
    assert_eq!(count, 1); // only art-d2 remains to be deleted

    assert!(!afs_root.path().join("req-1").exists());
}

// ════════════════════════════════
// 10. Cascade cleanup deletes artifact files
// ════════════════════════════════

#[test]
fn artifact_cascade_cleanup_deletes_artifact_files() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert summaries + artifacts for Jan and Mar.
    store
        .insert_summary(
            "req-jan",
            None,
            None,
            None,
            None,
            "2025-01-15T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
        rusqlite::params!["ev-jan-1", "req-jan", "2025-01-15T00:00:00Z", r#"{"type":"admitted"}"#],
    ).unwrap();

    store
        .insert_summary(
            "req-mar",
            None,
            None,
            None,
            None,
            "2025-03-15T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
        rusqlite::params!["ev-mar-1", "req-mar", "2025-03-15T00:00:00Z", r#"{"type":"admitted"}"#],
    ).unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write artifacts for both requests.
    afs.write_artifact(
        "art-jan",
        "req-jan",
        "log",
        "2025-01-15T00:00:00Z",
        b"jan data",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Need a new store reference to write for req-mar (same store).
    afs.write_artifact(
        "art-mar",
        "req-mar",
        "log",
        "2025-03-15T00:00:00Z",
        b"mar data",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Verify both files exist.
    assert!(afs_root.path().join("req-jan").join("art-jan").exists());
    assert!(afs_root.path().join("req-mar").join("art-mar").exists());

    // Cascade cleanup before Feb (removes Jan entries).
    let store_ref = afs.store_ref();
    let (_, artifact_ids) = store_ref
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .unwrap();

    // Delete the files that cascade removed from DB.
    afs.delete_artifact_files(&artifact_ids);

    // Jan file should be gone, Mar file survives.
    assert!(!afs_root.path().join("req-jan").join("art-jan").exists());
    assert!(afs_root.path().join("req-mar").join("art-mar").exists());

    // DB row for art-jan is gone too (CASCADE).
    match afs.read_artifact("art-jan") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected
        other => panic!("expected ArtifactMissing after cascade, got: {:?}", other),
    }
}

// ════════════════════════════════
// 11. Startup recovery removes orphans
// ════════════════════════════════

#[test]
fn artifact_startup_recovery_removes_orphans() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Create orphan .part file in tmp/ and an unreferenced artifact file.
    let afs_root = tempfile::tempdir().expect("artifact root");
    let tmp_dir = afs_root.path().join("tmp");
    fs::create_dir_all(&tmp_dir).unwrap();
    fs::write(tmp_dir.join("orphan.part"), b"stale temp data").unwrap();

    // Create an unreferenced artifact file (no DB pointer row).
    let req_dir = afs_root.path().join("req-ghost");
    fs::create_dir_all(&req_dir).unwrap();
    fs::write(req_dir.join("art-ghost"), b"unreferenced data").unwrap();

    // Open store — recovery should clean up.
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let _afs =
        ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Orphan .part removed.
    assert!(!tmp_dir.join("orphan.part").exists());

    // Unreferenced file removed.
    assert!(!req_dir.join("art-ghost").exists());
}

// ════════════════════════════════
// 12. Reopen preserves artifacts
// ════════════════════════════════

#[test]
fn artifact_reopen_preserves_artifacts() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Open store, write artifact.
    let store1 = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store1
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs1 =
        ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store1).unwrap();

    let content = b"persistent artifact data";
    afs1.write_artifact(
        "art-persist",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Drop artifact store (which owns the LogStore via Arc).
    drop(afs1);

    // Reopen at same paths.
    let store2 = LogStore::reopen_at(tmp.path(), clock.clone()).unwrap();
    let afs2 = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock, store2).unwrap();

    // Read succeeds with correct content.
    let art_content = afs2.read_artifact("art-persist").unwrap();
    assert_eq!(art_content.bytes.as_slice(), content);
}

// ════════════════════════════════
// 13. Unix modes 0700/0600
// ════════════════════════════════

#[cfg(unix)]
#[test]
fn artifact_unix_modes_0700_0600() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write an artifact.
    afs.write_artifact(
        "art-perm",
        "req-1",
        "log",
        &clock.now(),
        b"permission test",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Root dir should be mode 0700.
    let root_meta = fs::metadata(afs_root.path()).unwrap();
    assert_eq!(root_meta.permissions().mode() & 0o777, 0o700);

    // Request subdirectory should be mode 0700.
    let req_dir = afs_root.path().join("req-1");
    let dir_meta = fs::metadata(&req_dir).unwrap();
    assert_eq!(dir_meta.permissions().mode() & 0o777, 0o700);

    // Artifact file should be mode 0600.
    let art_path = req_dir.join("art-perm");
    let file_meta = fs::metadata(&art_path).unwrap();
    assert_eq!(file_meta.permissions().mode() & 0o777, 0o600);
}

// ════════════════════════════════
// 14. Duplicate write rejected (AlreadyExists)
// ════════════════════════════════

#[test]
fn artifact_duplicate_write_rejected() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // First write succeeds.
    afs.write_artifact(
        "art-dup",
        "req-1",
        "log",
        &clock.now(),
        b"first",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Second write with same artifact_id → AlreadyExists.
    let result = afs.write_artifact(
        "art-dup",
        "req-1",
        "log",
        &clock.now(),
        b"second attempt",
        None::<&str>,
        2,
        false,
        false,
        4096,
        8192,
    );

    match result {
        Err(LogStoreError::AlreadyExists { .. }) => {} // expected
        other => panic!(
            "expected AlreadyExists on duplicate write, got: {:?}",
            other
        ),
    }

    // Original file still intact (no orphan).
    let art_path = afs_root.path().join("req-1").join("art-dup");
    assert!(art_path.exists());
    assert_eq!(fs::read(&art_path).unwrap(), b"first");
}

// ════════════════════════════════
// 15. Transactional rollback removes file on FK failure
// ════════════════════════════════

#[test]
fn artifact_transactional_rollback_removes_file() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert a summary for req-1 but NOT for nonexistent-request.
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Attempt to write artifact for nonexistent request_id → FK violation.
    let result = afs.write_artifact(
        "art-fk",
        "nonexistent-request",
        "log",
        &clock.now(),
        b"fk fail",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );

    // Should fail with a SQLite error (FK constraint).
    assert!(result.is_err());

    // No orphan file should exist.
    let req_dir = afs_root.path().join("nonexistent-request");
    if req_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&req_dir).unwrap().collect();
        assert_eq!(
            entries.len(),
            0,
            "no artifact files should exist after FK failure"
        );
    }

    // No .part left in tmp/.
    let tmp_path = afs_root.path().join("tmp").join("art-fk.part");
    assert!(!tmp_path.exists());
}
