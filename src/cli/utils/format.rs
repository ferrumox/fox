use std::time::SystemTime;

/// Format byte count as human-readable string (GB / MB / B).
pub fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a `SystemTime` as a human-readable relative age string.
pub fn format_age(modified: SystemTime) -> String {
    let elapsed = SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default();
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{} seconds ago", secs)
    } else if secs < 3600 {
        let m = secs / 60;
        if m == 1 {
            "1 minute ago".to_string()
        } else {
            format!("{} minutes ago", m)
        }
    } else if secs < 86400 {
        let h = secs / 3600;
        if h == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{} hours ago", h)
        }
    } else {
        let d = secs / 86400;
        if d == 1 {
            "1 day ago".to_string()
        } else {
            format!("{} days ago", d)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(999), "999 B");
        assert_eq!(format_size(999_999), "999999 B");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1_000_000), "1.0 MB");
        assert_eq!(format_size(5_500_000), "5.5 MB");
        assert_eq!(format_size(999_999_999), "1000.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
        assert_eq!(format_size(7_300_000_000), "7.3 GB");
    }

    #[test]
    fn test_format_age_seconds() {
        let t = SystemTime::now() - Duration::from_secs(30);
        assert_eq!(format_age(t), "30 seconds ago");
    }

    #[test]
    fn test_format_age_one_minute() {
        let t = SystemTime::now() - Duration::from_secs(60);
        assert_eq!(format_age(t), "1 minute ago");
    }

    #[test]
    fn test_format_age_minutes() {
        let t = SystemTime::now() - Duration::from_secs(5 * 60);
        assert_eq!(format_age(t), "5 minutes ago");
    }

    #[test]
    fn test_format_age_one_hour() {
        let t = SystemTime::now() - Duration::from_secs(3600);
        assert_eq!(format_age(t), "1 hour ago");
    }

    #[test]
    fn test_format_age_hours() {
        let t = SystemTime::now() - Duration::from_secs(3 * 3600);
        assert_eq!(format_age(t), "3 hours ago");
    }

    #[test]
    fn test_format_age_one_day() {
        let t = SystemTime::now() - Duration::from_secs(86400);
        assert_eq!(format_age(t), "1 day ago");
    }

    #[test]
    fn test_format_age_days() {
        let t = SystemTime::now() - Duration::from_secs(3 * 86400);
        assert_eq!(format_age(t), "3 days ago");
    }
}
