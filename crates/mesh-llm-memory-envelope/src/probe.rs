use std::fs;

/// Snapshot of system memory state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SystemMemory {
    pub mem_available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
}

/// Snapshot of this process's memory state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProcessMemory {
    pub rss_bytes: u64,
    pub minor_faults: u64,
    pub major_faults: u64,
}

/// Sample system + process memory. Returns None if any probe fails.
pub fn sample() -> Option<(SystemMemory, ProcessMemory)> {
    let sys = system_memory().ok()?;
    let proc = process_memory().ok()?;
    Some((sys, proc))
}

fn system_memory() -> Result<SystemMemory, anyhow::Error> {
    let text = fs::read_to_string("/proc/meminfo")?;
    let mut avail = None;
    let mut swap_total = None;
    let mut swap_free = None;

    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let key = parts
            .next()
            .ok_or_else(|| anyhow::anyhow!("malformed line"))?;
        let val: u64 = parts
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| anyhow::anyhow!("malformed value"))?;
        match key.trim_end_matches(':') {
            "MemAvailable" => avail = Some(val * 1024),
            "SwapTotal" => swap_total = Some(val * 1024),
            "SwapFree" => swap_free = Some(val * 1024),
            _ => {}
        }
    }

    Ok(SystemMemory {
        mem_available_bytes: avail.ok_or_else(|| anyhow::anyhow!("MemAvailable not found"))?,
        swap_total_bytes: swap_total.unwrap_or(0),
        swap_free_bytes: swap_free.unwrap_or(0),
    })
}

fn process_memory() -> Result<ProcessMemory, anyhow::Error> {
    let status = fs::read_to_string("/proc/self/status")?;
    let mut rss = None;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            rss = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
                .map(|kb| kb * 1024);
        }
    }

    // getrusage gives minor/major fault counters
    let mut rusage = unsafe { std::mem::zeroed::<libc::rusage>() };
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut rusage) };
    let (minflt, majflt) = if rc == 0 {
        (Some(rusage.ru_minflt as u64), Some(rusage.ru_majflt as u64))
    } else {
        (None, None)
    };

    Ok(ProcessMemory {
        rss_bytes: rss.unwrap_or(0),
        minor_faults: minflt.unwrap_or(0),
        major_faults: majflt.unwrap_or(0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_system_memory() {
        let (sys, proc) = sample().expect("probe should succeed on Linux");
        assert!(sys.mem_available_bytes > 0, "MemAvailable should be non-zero");
        assert!(proc.rss_bytes > 0, "RSS should be non-zero");
    }
}
