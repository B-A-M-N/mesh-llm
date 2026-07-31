//! Versioned canonical logging contracts and lifecycle invariants.
//!
//! This module defines the semantic event types used by the logging system.
//! `OutputEvent` remains a presentation adapter and is never persisted raw.

pub mod identifiers;
pub mod envelope;
pub mod lifecycle;
pub mod summaries;
pub mod events;
pub mod artifacts;
pub mod proxy;
pub mod replay;

#[cfg(test)]
mod tests;
