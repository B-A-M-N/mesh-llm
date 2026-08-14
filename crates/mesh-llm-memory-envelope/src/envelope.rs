use std::ffi::CStr;
use std::os::raw::c_int;
use std::time::Instant;

/// Timeline memory sample captured at each phase boundary.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TimelineSample {
    pub label: &'static str,
    pub system: super::probe::SystemMemory,
    pub process: super::probe::ProcessMemory,
}

/// Full memory envelope result for a single model load + forward.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryEnvelopeResultV2 {
    pub timeline: Vec<TimelineSample>,
    pub forward_passed: bool,
    pub model_load_ms: u64,
    pub first_forward_ms: u64,
}

impl MemoryEnvelopeResultV2 {
    pub fn system_delta_bytes(&self) -> Option<(u64, u64)> {
        let t0 = &self.timeline.first()?;
        let t_last = &self.timeline.last()?;
        let rss_delta = t_last.process.rss_bytes.saturating_sub(t0.process.rss_bytes);
        let memavail_delta = t0.system.mem_available_bytes.saturating_sub(t_last.system.mem_available_bytes);
        Some((rss_delta, memavail_delta))
    }
}

impl std::fmt::Display for MemoryEnvelopeResultV2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for sample in &self.timeline {
            writeln!(
                f,
                "Tl[{}] rss={}MB memavail={}MB swap={}MB minflt={} majflt={}",
                sample.label,
                sample.process.rss_bytes / 1024 / 1024,
                sample.system.mem_available_bytes / 1024 / 1024,
                (sample.system.swap_total_bytes - sample.system.swap_free_bytes) / 1024 / 1024,
                sample.process.minor_faults,
                sample.process.major_faults,
            )?;
        }
        write!(f, "forward_ok={}", self.forward_passed)
    }
}
