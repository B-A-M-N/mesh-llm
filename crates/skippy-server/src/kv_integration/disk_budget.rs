//! Node-level budget policy for the KV disk tier.
//!
//! # Why this is not just a number
//!
//! `PrefixDiskTier` enforces a byte budget with LRU eviction, but the budget
//! it is handed used to be *per stage directory*. Entries are namespaced per
//! stage identity so two stages never share a directory, which is right for
//! identity and wrong for accounting: a node serving three models with an
//! 8 GiB setting was actually allowed 24 GiB. Nobody configuring a cache size
//! expects it to be multiplied by however many models happen to be loaded.
//!
//! This module makes the configured number mean what it says. A single node
//! total is resolved once, and each stage reserves a share of it. When the
//! pool is exhausted, later stages get nothing rather than the node quietly
//! exceeding its budget.
//!
//! # Why free space matters
//!
//! An absolute byte budget has no relationship to the disk it sits on. 8 GiB
//! is trivial on a workstation and fatal on a nearly-full laptop, and filling
//! a user's boot disk to serve a cache is not a recoverable kind of mistake.
//! So the default is a share of *free* space with a fixed cap, and there is a
//! floor below which the tier declines to open at all.
//!
//! The share is deliberately generous. This cache exists for repeat users
//! returning to a large agent prefix after a gap; a miss costs seconds of GPU
//! prefill, while the disk it occupies sits on a machine already storing
//! multi-gigabyte model weights. Under-caching is the more expensive error
//! here, so the policy is tuned for hit rate rather than frugality.

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

/// Upper bound on the node-level budget regardless of how large the disk is.
///
/// A share of free space alone would hand a 100 GiB budget to a machine with
/// a 1 TB empty disk, which is far more than any realistic prefix working set
/// and more than a user would expect a cache to take without asking.
const MAX_NODE_BUDGET_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// Fraction of free space the default budget may occupy, as a percentage.
///
/// Generous by cache standards, per the retention goal above.
const FREE_SPACE_PERCENT: u64 = 20;

/// Refuse to open the tier when free space is below this.
///
/// Declining to cache costs prefill time. Filling the disk costs the user
/// their machine, so the asymmetry justifies a hard floor.
const MIN_FREE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Number of stages the node total is divided across.
///
/// The pool is claimed at open time and never rebalanced, so a divisor is
/// needed up front. Four covers a node serving several models or holding
/// several split stages; beyond that, later stages draw on whatever the pool
/// has left.
const STAGE_SHARES: u64 = 4;

/// Remaining unreserved bytes of the node budget.
static POOL: OnceLock<Mutex<u64>> = OnceLock::new();

/// How the node-level budget was arrived at, for logging and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NodeBudget {
    /// Operator set an explicit size; free space is not consulted.
    Explicit(u64),
    /// Derived from free space on the cache filesystem.
    Derived(u64),
    /// Free space is below the floor; the tier must not open.
    InsufficientSpace { free_bytes: u64 },
    /// Not enabled.
    Disabled,
}

impl NodeBudget {
    pub(super) fn bytes(self) -> Option<u64> {
        match self {
            Self::Explicit(bytes) | Self::Derived(bytes) => Some(bytes),
            Self::InsufficientSpace { .. } | Self::Disabled => None,
        }
    }
}

/// Resolve the node-level budget from an explicit setting and free space.
///
/// Kept pure so the policy is testable without touching a filesystem.
pub(super) fn resolve_node_budget(
    explicit_bytes: Option<u64>,
    enabled: bool,
    free_bytes: Option<u64>,
) -> NodeBudget {
    if let Some(bytes) = explicit_bytes {
        // An explicit size is an instruction, not a hint: an operator who
        // sizes the cache has taken responsibility for the disk. Still refuse
        // if the filesystem cannot hold it.
        return match free_bytes {
            Some(free) if free < bytes.saturating_add(MIN_FREE_BYTES) => {
                NodeBudget::InsufficientSpace { free_bytes: free }
            }
            _ => NodeBudget::Explicit(bytes),
        };
    }
    if !enabled {
        return NodeBudget::Disabled;
    }
    let Some(free) = free_bytes else {
        // Without a free-space reading there is no safe derived default, and
        // guessing is exactly the failure mode this module exists to prevent.
        return NodeBudget::InsufficientSpace { free_bytes: 0 };
    };
    if free < MIN_FREE_BYTES {
        return NodeBudget::InsufficientSpace { free_bytes: free };
    }
    // Multiply first: dividing by 100 first discards up to 99 bytes per
    // percent, which is noise here but makes the policy awkward to reason
    // about and to test exactly.
    let share = free.saturating_mul(FREE_SPACE_PERCENT) / 100;
    NodeBudget::Derived(share.min(MAX_NODE_BUDGET_BYTES))
}

/// Claim this stage's share of the node budget.
///
/// Returns the bytes this stage's tier may use, or `None` when the pool is
/// exhausted. Claims are permanent for the life of the process: the tier owns
/// its directory until shutdown, so returning bytes would mean shrinking a
/// live tier from outside it.
pub(super) fn claim_stage_share(node_budget: u64) -> Option<u64> {
    let pool = POOL.get_or_init(|| Mutex::new(node_budget));
    let mut remaining = pool.lock().expect("KV disk budget pool poisoned");
    let share = (node_budget / STAGE_SHARES).max(1);
    let claimed = share.min(*remaining);
    if claimed == 0 {
        return None;
    }
    *remaining -= claimed;
    Some(claimed)
}

/// The nearest ancestor of `path` that exists on disk.
///
/// The cache directory is created by the tier itself, so on a first run
/// neither it nor its configured parent exists yet. `statvfs` fails on a
/// missing path, and the budget policy treats an unreadable filesystem as
/// "do not enable" -- correct in general, but it meant the tier could never
/// open the very first time. Walking up to an existing ancestor measures the
/// same filesystem the cache will live on, since the missing components will
/// be created under it.
pub(super) fn existing_ancestor(path: &Path) -> PathBuf {
    let mut current = path;
    loop {
        if current.exists() {
            return current.to_path_buf();
        }
        match current.parent() {
            Some(parent) => current = parent,
            // A path with no existing ancestor at all: fall back to the root,
            // which lets `statvfs` decide rather than guessing here.
            None => return PathBuf::from("/"),
        }
    }
}

/// Free bytes available on the filesystem holding `path`.
///
/// Returns `None` when the platform query fails, which callers must treat as
/// "do not enable" rather than "assume plenty".
pub(super) fn free_space_bytes(path: &Path) -> Option<u64> {
    #[cfg(unix)]
    {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};

        let c_path = CString::new(path.as_os_str().as_bytes()).ok()?;
        // SAFETY: `c_path` is a valid NUL-terminated string for the duration
        // of the call, and `stats` is a correctly sized zeroed struct that
        // statvfs fully initializes on success.
        let mut stats: libc::statvfs = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::statvfs(c_path.as_ptr(), &mut stats) };
        if rc != 0 {
            return None;
        }
        // `f_bavail` is blocks available to unprivileged users, which is the
        // number that matters here -- `f_bfree` includes reserved blocks a
        // normal process cannot actually use.
        let block_size = if stats.f_frsize > 0 {
            stats.f_frsize as u64
        } else {
            stats.f_bsize as u64
        };
        Some(stats.f_bavail as u64 * block_size)
    }
    #[cfg(not(unix))]
    {
        // The tier already refuses to open on platforms without advisory
        // directory locking, so there is no path that reaches this today.
        let _ = path;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn derived_budget_is_a_share_of_free_space() {
        let budget = resolve_node_budget(None, true, Some(200 * GIB));
        assert_eq!(budget, NodeBudget::Derived(40 * GIB));
    }

    /// The reason free space is consulted at all: the same setting must not
    /// behave identically on a roomy workstation and a nearly-full laptop.
    #[test]
    fn derived_budget_scales_down_on_a_small_disk() {
        let budget = resolve_node_budget(None, true, Some(40 * GIB));
        assert_eq!(budget, NodeBudget::Derived(8 * GIB));
    }

    #[test]
    fn derived_budget_is_capped_on_a_very_large_disk() {
        let budget = resolve_node_budget(None, true, Some(4000 * GIB));
        assert_eq!(budget, NodeBudget::Derived(MAX_NODE_BUDGET_BYTES));
    }

    #[test]
    fn tier_declines_below_the_free_space_floor() {
        assert!(matches!(
            resolve_node_budget(None, true, Some(4 * GIB)),
            NodeBudget::InsufficientSpace { .. }
        ));
    }

    /// First run: the cache directory and its parent do not exist yet, so
    /// free space must be measured on an ancestor that does. Probing a
    /// missing path made `statvfs` fail and disabled the tier permanently on
    /// any node that had never created the directory.
    #[test]
    fn free_space_is_measured_on_an_existing_ancestor() {
        let temp = std::env::temp_dir();
        let missing = temp
            .join("skippy-kv-budget-probe-does-not-exist")
            .join("nested")
            .join("deeper");
        assert!(!missing.exists());

        let ancestor = existing_ancestor(&missing);

        assert!(ancestor.exists());
        assert!(free_space_bytes(&ancestor).is_some_and(|bytes| bytes > 0));
    }

    /// An unreadable filesystem must not be read as "plenty of room".
    #[test]
    fn unknown_free_space_declines_rather_than_assuming() {
        assert!(matches!(
            resolve_node_budget(None, true, None),
            NodeBudget::InsufficientSpace { .. }
        ));
    }

    #[test]
    fn explicit_budget_is_honoured_when_the_disk_can_hold_it() {
        assert_eq!(
            resolve_node_budget(Some(8 * GIB), true, Some(500 * GIB)),
            NodeBudget::Explicit(8 * GIB)
        );
    }

    /// An explicit size is an instruction, but not a licence to fill the disk.
    #[test]
    fn explicit_budget_still_refuses_to_fill_the_disk() {
        assert!(matches!(
            resolve_node_budget(Some(100 * GIB), true, Some(20 * GIB)),
            NodeBudget::InsufficientSpace { .. }
        ));
    }

    #[test]
    fn disabled_without_an_explicit_size() {
        assert_eq!(
            resolve_node_budget(None, false, Some(500 * GIB)),
            NodeBudget::Disabled
        );
        assert_eq!(
            resolve_node_budget(None, false, Some(500 * GIB)).bytes(),
            None
        );
    }
}
