// SHA256 digest helpers and file metadata utilities.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;

/// Compute SHA256 of a file, returning `"sha256:<hex>"`.
pub async fn file_digest(path: PathBuf, cache: Arc<Mutex<HashMap<PathBuf, String>>>) -> String {
    if let Some(cached) = cache.lock().unwrap().get(&path).cloned() {
        return cached;
    }
    tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok::<String, std::io::Error>(format!("sha256:{}", hex::encode(hasher.finalize())))
    })
    .await
    .ok()
    .and_then(|r| r.ok())
    .unwrap_or_else(|| "sha256:unknown".to_string())
}

/// Return the digest for `path`, populating `cache` on first call.
pub async fn get_digest(path: &PathBuf, cache: &Arc<Mutex<HashMap<PathBuf, String>>>) -> String {
    if let Some(cached) = cache.lock().unwrap().get(path).cloned() {
        return cached;
    }
    let digest = file_digest(path.clone(), cache.clone()).await;
    cache.lock().unwrap().insert(path.clone(), digest.clone());
    digest
}

/// Format a file's `modified_at` timestamp as a minimal RFC 3339 UTC string.
pub fn modified_at_rfc3339(meta: &std::fs::Metadata) -> String {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| {
            let s = d.as_secs();
            let sec = s % 60;
            let min = (s / 60) % 60;
            let hour = (s / 3600) % 24;
            let days_since_epoch = s / 86400;
            let year = 1970u64 + days_since_epoch / 365;
            let day_of_year = days_since_epoch % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
        })
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}
