// Model digest helpers and file metadata utilities.

use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::UNIX_EPOCH;

/// Return the digest reported for a model file, as `"sha256:<hex>"`.
///
/// This hashes the file's *identity* — name, size and mtime — not its contents.
/// Hashing the contents is what this used to do, and it made `GET /api/tags`
/// read every byte of every GGUF in the models directory before it could
/// answer: ~50 s and a pegged core for a 27 GB directory, repeated in full for
/// every concurrent request, since the cache was only written once a hash
/// finished. A listing endpoint cannot afford that.
///
/// Nothing in fox resolves a model by digest — it is an opaque identifier for
/// Ollama clients — so the property that actually matters is that it stays
/// stable while the file does and changes when the file is replaced. The
/// identity triple gives exactly that, for free.
pub fn metadata_digest(path: &Path, meta: &std::fs::Metadata) -> String {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);

    let mut hasher = Sha256::new();
    hasher.update(format!("{name}:{}:{mtime}", meta.len()).as_bytes());
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

/// Format a file's `modified_at` timestamp as a minimal RFC 3339 UTC string.
pub fn modified_at_rfc3339(meta: &std::fs::Metadata) -> String {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| rfc3339_utc(d.as_secs()))
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}

/// Format seconds since the Unix epoch as `YYYY-MM-DDThh:mm:ssZ`.
///
/// The date half is Howard Hinnant's `civil_from_days`, which is exact for every
/// day this can represent. It replaces an approximation — `year = 1970 + days/365`,
/// `month = day_of_year/30 + 1` — that ignored leap years and assumed 30-day
/// months, so it drifted a day per leap year and several days within each year:
/// a file touched on 2025-08-04 was reported as 2025-08-20.
fn rfc3339_utc(secs: u64) -> String {
    let sec = secs % 60;
    let min = (secs / 60) % 60;
    let hour = (secs / 3600) % 24;

    // Shift the epoch to 0000-03-01 so leap days land at the end of the cycle.
    let z = secs / 86400 + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year, March-based
    let mp = (5 * doy + 2) / 153; // March-based month, [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = yoe + era * 400 + u64::from(month <= 2); // January/February roll over

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn temp_gguf(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        f.sync_all().unwrap();
        (dir, path)
    }

    #[test]
    fn digest_is_well_formed_and_stable() {
        let (_dir, path) = temp_gguf("a.gguf", b"hello");
        let meta = std::fs::metadata(&path).unwrap();

        let d = metadata_digest(&path, &meta);
        assert!(d.starts_with("sha256:"));
        assert_eq!(d.len(), "sha256:".len() + 64);
        // Same file, same answer — repeated calls must not drift.
        assert_eq!(d, metadata_digest(&path, &meta));
    }

    #[test]
    fn digest_does_not_depend_on_contents() {
        // Identity, not contents: two same-sized files differ only by name here,
        // which is what keeps the endpoint from reading gigabytes.
        let (_dir, a) = temp_gguf("a.gguf", b"aaaaa");
        let meta_a = std::fs::metadata(&a).unwrap();
        let (_dir_b, b) = temp_gguf("b.gguf", b"aaaaa");
        let meta_b = std::fs::metadata(&b).unwrap();

        assert_ne!(metadata_digest(&a, &meta_a), metadata_digest(&b, &meta_b));
    }

    #[test]
    fn rfc3339_matches_the_real_calendar() {
        // Leap days, the 2100 non-leap century, and year/month boundaries — every
        // place the old day/365 + day/30 approximation went wrong.
        let cases = [
            (0, "1970-01-01T00:00:00Z"),
            (31_535_999, "1970-12-31T23:59:59Z"),
            (68_212_800, "1972-02-29T12:00:00Z"),
            (951_782_400, "2000-02-29T00:00:00Z"),
            (951_868_800, "2000-03-01T00:00:00Z"),
            (1_735_689_599, "2024-12-31T23:59:59Z"),
            (1_754_300_000, "2025-08-04T09:33:20Z"),
            (1_786_147_200, "2026-08-08T00:00:00Z"),
            (4_107_542_399, "2100-02-28T23:59:59Z"),
            (4_107_542_400, "2100-03-01T00:00:00Z"),
        ];
        for (secs, expected) in cases {
            assert_eq!(rfc3339_utc(secs), expected, "at {secs}");
        }
    }

    #[test]
    fn rfc3339_advances_one_day_at_a_time() {
        // Independent oracle: walk 200 years a day at a time and check every date
        // against a plain month-length table. Catches drift a spot check would miss.
        fn leap(y: u64) -> bool {
            (y.is_multiple_of(4) && !y.is_multiple_of(100)) || y.is_multiple_of(400)
        }
        fn days_in(y: u64, m: u64) -> u64 {
            match m {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                _ if leap(y) => 29,
                _ => 28,
            }
        }

        let (mut y, mut m, mut d) = (1970u64, 1u64, 1u64);
        for day in 0..(200 * 366) {
            assert_eq!(
                rfc3339_utc(day * 86400),
                format!("{y:04}-{m:02}-{d:02}T00:00:00Z"),
                "day {day} after the epoch"
            );
            d += 1;
            if d > days_in(y, m) {
                d = 1;
                m += 1;
                if m > 12 {
                    m = 1;
                    y += 1;
                }
            }
        }
    }

    #[test]
    fn modified_at_reports_the_files_real_mtime() {
        let (_dir, path) = temp_gguf("a.gguf", b"hello");
        let meta = std::fs::metadata(&path).unwrap();
        let secs = meta
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        assert_eq!(modified_at_rfc3339(&meta), rfc3339_utc(secs));
        // A file written now is in this century, not drifting off into the next.
        assert!(modified_at_rfc3339(&meta).starts_with("20"));
    }

    #[test]
    fn digest_changes_when_the_file_is_replaced() {
        let (_dir, path) = temp_gguf("a.gguf", b"hello");
        let before = metadata_digest(&path, &std::fs::metadata(&path).unwrap());

        // A different size is enough to invalidate, without waiting on clock
        // granularity for mtime to tick.
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"hello world").unwrap();
        f.sync_all().unwrap();

        let after = metadata_digest(&path, &std::fs::metadata(&path).unwrap());
        assert_ne!(before, after);
    }
}
