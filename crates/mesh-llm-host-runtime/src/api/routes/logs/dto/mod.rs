mod artifacts;
mod events;
mod proxy;
mod requests;

pub(super) use artifacts::{ArtifactDto, artifact_state};
pub(super) use events::EventDto;
pub(super) use proxy::ProxyDto;
pub(super) use requests::{PageDto, RequestDto};

fn safe_metadata(value: &str) -> String {
    let trimmed = value.trim();
    let path_shaped = trimmed.starts_with('/')
        || trimmed.starts_with("~/")
        || trimmed.as_bytes().get(1) == Some(&b':')
        || trimmed.contains('\\');
    if path_shaped || trimmed.contains('?') || trimmed.contains("://") {
        return "[REDACTED]".to_string();
    }
    crate::logging::policy::apply_redaction(trimmed).0
}
