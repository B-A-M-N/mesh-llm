//! Operator logging privacy policy and redaction infrastructure.
//! Centralizes all credential/token/header/query/body/stack sanitization before any log event is serialized or persisted.
//!
//! Note: `#[allow(dead_code)]` is intentional — this module defines the full policy surface
//! that will be wired into the runtime logging pipeline in subsequent tasks.

#![allow(dead_code)]

pub mod policy;
