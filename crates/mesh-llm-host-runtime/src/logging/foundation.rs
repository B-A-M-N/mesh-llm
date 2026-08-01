//! Logging foundation: app-state root resolution, store/artifact initialization, health status.
//!
//! Wires validated logging config into host initialization without broadly instrumenting producers yet.
//! On startup (when enabled), creates the application-state layout expected by mesh-llm-log-store.
//! Follows fail-open policy: if the root is unwritable, disable logging and continue serving.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

/// Resolved application-state layout for the logging subsystem.
pub struct LoggingFoundation {
    /// Whether logging was successfully initialized (true) or failed-open/disabled (false).
    healthy: AtomicBool,
    /// The resolved root directory for all logging application state.
    app_state_root: PathBuf,
    /// Path to the SQLite-backed log store directory (`<root>/store/`).
    store_dir: PathBuf,
    /// Path to the artifact file storage root (`<root>/artifacts/`).
    artifact_dir: PathBuf,
}

impl LoggingFoundation {
    /// Resolve and initialize the logging foundation from config.
    ///
    /// If `enabled` is false, returns a disabled (unhealthy) instance that creates NO files.
    /// If initialization fails (unwritable root), returns an unhealthy instance with a sanitized diagnostic.
    pub fn init(enabled: bool, application_state_root: Option<&PathBuf>) -> Self {
        if !enabled {
            return Self::disabled();
        }

        let app_state_root = resolve_app_state_root(application_state_root);

        // Attempt to create the store and artifact directories (idempotent).
        let store_dir = app_state_root.join("store");
        let artifact_dir = app_state_root.join("artifacts");

        if !try_create_dirs(&app_state_root, &store_dir, &artifact_dir) {
            tracing::warn!(
                root = %sanitize_path(&app_state_root),
                "Failed to create logging application-state directories; disabling logging (fail-open)"
            );
            return Self {
                healthy: AtomicBool::new(false),
                app_state_root,
                store_dir,
                artifact_dir,
            };
        }

        tracing::info!(
            root = %sanitize_path(&app_state_root),
            "Logging application-state layout initialized"
        );

        Self {
            healthy: AtomicBool::new(true),
            app_state_root,
            store_dir,
            artifact_dir,
        }
    }

    /// Create a disabled foundation (logging.enabled = false). No files are created.
    pub fn disabled() -> Self {
        let dummy_path = PathBuf::from("/disabled");
        Self {
            healthy: AtomicBool::new(false),
            app_state_root: dummy_path.clone(),
            store_dir: dummy_path.join("store"),
            artifact_dir: dummy_path.join("artifacts"),
        }
    }

    /// Whether logging is initialized and operational.
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    /// The resolved application-state root directory.
    pub fn app_state_root(&self) -> &Path {
        &self.app_state_root
    }

    /// Path to the log store directory (contains `log_store.db`).
    pub fn store_dir(&self) -> &Path {
        &self.store_dir
    }

    /// Path to the artifact file storage root.
    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }

    /// Reinitialize after a restart-required config change (e.g., application_state_root changed).
    /// Returns a new LoggingFoundation if successful, None otherwise. Callers replace their handle with the returned value.
    pub fn reinit(new_root: Option<&PathBuf>) -> Option<Self> {
        let new_app_state_root = resolve_app_state_root(new_root);

        let store_dir = new_app_state_root.join("store");
        let artifact_dir = new_app_state_root.join("artifacts");

        if try_create_dirs(&new_app_state_root, &store_dir, &artifact_dir) {
            Some(Self {
                healthy: AtomicBool::new(true),
                app_state_root: new_app_state_root,
                store_dir,
                artifact_dir,
            })
        } else {
            None
        }
    }

    /// Sanitized health summary for diagnostics (no sensitive paths leaked).
    pub fn health_summary(&self) -> String {
        if !self.is_healthy() {
            return "logging disabled or unhealthy".to_string();
        }
        format!("logging healthy at {}", sanitize_path(&self.app_state_root))
    }

    /// Check whether the store directory exists on disk (for idempotent startup verification).
    #[cfg(test)]
    pub fn store_dir_exists_on_disk(&self) -> bool {
        self.store_dir.exists() && self.store_dir.is_dir()
    }

    /// Check whether the artifact directory exists on disk.
    #[cfg(test)]
    pub fn artifact_dir_exists_on_disk(&self) -> bool {
        self.artifact_dir.exists() && self.artifact_dir.is_dir()
    }
}

/// Resolve the application-state root from config or platform defaults.
fn resolve_app_state_root(config_path: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = config_path {
        return path.clone();
    }

    // Follow existing mesh-llm conventions for app data directories:
    // 1. MESH_LLM_DATA_DIR env var (highest priority, used by model-hf crate)
    // 2. $HOME/.mesh-llm/logging (default fallback)
    if let Some(env_path) = std::env::var_os("MESH_LLM_DATA_DIR") {
        return PathBuf::from(env_path).join("logging");
    }

    dirs::home_dir()
        .map(|home| home.join(".mesh-llm").join("logging"))
        .unwrap_or_else(|| PathBuf::from("/tmp/mesh-llm/logging"))
}

/// Attempt to create the app_state_root, store_dir, and artifact_dir. Returns true if all succeed.
fn try_create_dirs(root: &Path, store: &Path, artifacts: &Path) -> bool {
    for dir in [root, store, artifacts] {
        match std::fs::create_dir_all(dir) {
            Ok(()) => {}
            Err(e) => {
                tracing::debug!(path = %sanitize_path(dir), error = %e, "failed to create directory");
                return false;
            }
        }
    }

    // Verify the root is actually writable by attempting a quick write test.
    let test_file = root.join(".write_test");
    if std::fs::write(&test_file, "").is_err() {
        tracing::debug!(path = %sanitize_path(root), "root directory exists but is not writable");
        return false;
    }

    // Clean up the test file.
    let _ = std::fs::remove_file(&test_file);
    true
}

/// Sanitize a path for logging (replace home dir with ~, strip sensitive segments).
fn sanitize_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir()
        && let Ok(stripped) = path.strip_prefix(&home)
    {
        return format!("~/{}", stripped.display());
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    fn temp_root() -> PathBuf {
        static NEXT_ROOT: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "mesh-llm-log-foundation-{}-{}",
            std::process::id(),
            NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn init_enabled_creates_layout() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));
        assert!(foundation.is_healthy());
        assert_eq!(foundation.app_state_root(), &root);
        assert_eq!(foundation.store_dir(), &root.join("store"));
        assert_eq!(foundation.artifact_dir(), &root.join("artifacts"));
        assert!(foundation.store_dir_exists_on_disk());
        assert!(foundation.artifact_dir_exists_on_disk());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn init_disabled_creates_no_files() {
        let root = temp_root().join("disabled-test");
        // Root doesn't exist yet — disabled should NOT create it.
        assert!(!root.exists());

        let foundation = LoggingFoundation::init(false, Some(&root));
        assert!(!foundation.is_healthy());
        assert!(
            !root.exists(),
            "disabled logging must not create any directories"
        );

        // Cleanup (noop since nothing was created).
    }

    #[test]
    fn init_unwritable_root_fails_open() {
        let root = temp_root();
        std::fs::write(&root, "not a directory").unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));

        assert!(!foundation.is_healthy(), "a file root must fail open");
        std::fs::remove_file(root).unwrap();
    }

    #[test]
    fn init_idempotent_same_root() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let f1 = LoggingFoundation::init(true, Some(&root));
        assert!(f1.is_healthy());

        // Second initialization against the same root is safe (idempotent).
        let f2 = LoggingFoundation::init(true, Some(&root));
        assert!(f2.is_healthy());
        assert_eq!(f2.app_state_root(), &root);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn resolve_default_falls_back_to_home() {
        // Call the private resolver directly — avoids creating real dirs under $HOME.
        let resolved = resolve_app_state_root(None);

        if let Some(home) = dirs::home_dir() {
            assert!(resolved.starts_with(&home));
            assert!(resolved.ends_with(".mesh-llm/logging"));
        } else {
            // Fallback to /tmp when home is unavailable (rare in tests).
            assert_eq!(resolved, PathBuf::from("/tmp/mesh-llm/logging"));
        }
    }

    #[test]
    fn health_summary_healthy() {
        let root = temp_root();
        std::fs::create_dir_all(&root).unwrap();

        let foundation = LoggingFoundation::init(true, Some(&root));
        let summary = foundation.health_summary();
        assert!(summary.contains("logging healthy"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn health_summary_disabled() {
        let root = temp_root().join("disabled-summary");
        let foundation = LoggingFoundation::init(false, Some(&root));
        assert_eq!(foundation.health_summary(), "logging disabled or unhealthy");
    }

    #[test]
    fn sanitize_path_replaces_home_with_tilde() {
        if let Some(home) = dirs::home_dir() {
            let path = home.join("foo").join("bar");
            assert_eq!(sanitize_path(&path), "~/foo/bar");
        } else {
            // Fallback: just check it doesn't panic.
            let _ = sanitize_path(Path::new("/some/path"));
        }
    }

    #[test]
    fn sanitize_path_returns_display_for_non_home() {
        assert_eq!(sanitize_path(Path::new("/var/log/test")), "/var/log/test");
    }
}
