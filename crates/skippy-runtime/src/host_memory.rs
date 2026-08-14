use anyhow::Result;
use std::fs;

pub struct HostMemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

pub fn probe_host_memory() -> Result<HostMemoryInfo> {
    let contents = fs::read_to_string("/proc/meminfo")?;
    let mut total = None;
    let mut available = None;
    for line in contents.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            available = parse_meminfo_kb(line);
        }
    }
    Ok(HostMemoryInfo {
        total_bytes: total.ok_or_else(|| anyhow::anyhow!("MemTotal not found"))? * 1024,
        available_bytes: available.ok_or_else(|| anyhow::anyhow!("MemAvailable not found"))? * 1024,
    })
}

fn parse_meminfo_kb(line: &str) -> Option<u64> {
    line.split_whitespace().nth(1)?.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_meminfo_line() {
        assert_eq!(parse_meminfo_kb("MemTotal:       32768000 kB"), Some(32768000));
        assert_eq!(parse_meminfo_kb("MemAvailable:   25600000 kB"), Some(25600000));
        assert_eq!(parse_meminfo_kb("invalid"), None);
    }
}