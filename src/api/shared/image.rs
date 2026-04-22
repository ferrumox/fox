use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024; // 20 MB
const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Resolve an image reference to raw bytes.
///
/// Supports:
///   - `data:image/...;base64,<data>` (OpenAI data URI format)
///   - `http://` or `https://` URLs (fetched via reqwest)
///   - Raw base64 string (Ollama format)
pub async fn resolve_image_bytes(image_ref: &str) -> Result<Vec<u8>> {
    if let Some(rest) = image_ref.strip_prefix("data:") {
        let (_, b64) = rest
            .split_once(";base64,")
            .ok_or_else(|| anyhow!("invalid data URI: missing ;base64, separator"))?;
        return STANDARD
            .decode(b64)
            .map_err(|e| anyhow!("base64 decode error: {e}"));
    }

    if image_ref.starts_with("http://") || image_ref.starts_with("https://") {
        let client = reqwest::Client::builder().timeout(FETCH_TIMEOUT).build()?;
        let resp = client.get(image_ref).send().await?.error_for_status()?;
        let content_length = resp.content_length().unwrap_or(0) as usize;
        if content_length > MAX_IMAGE_BYTES {
            return Err(anyhow!(
                "image too large: {} bytes (max {})",
                content_length,
                MAX_IMAGE_BYTES
            ));
        }
        let bytes = resp.bytes().await?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(anyhow!(
                "image too large: {} bytes (max {})",
                bytes.len(),
                MAX_IMAGE_BYTES
            ));
        }
        return Ok(bytes.to_vec());
    }

    // Assume raw base64 (Ollama format).
    STANDARD
        .decode(image_ref)
        .map_err(|e| anyhow!("base64 decode error: {e}"))
}
