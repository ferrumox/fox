use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize the OpenAI `stop` field which can be either a string or an array of strings.
pub(super) fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StopField {
        Single(String),
        Multiple(Vec<String>),
    }

    let opt = Option::<StopField>::deserialize(deserializer)?;
    Ok(opt.map(|v| match v {
        StopField::Single(s) => vec![s],
        StopField::Multiple(v) => v,
    }))
}

pub const DEFAULT_MAX_TOKENS: u32 = 256;

pub(super) fn default_max_tokens() -> Option<u32> {
    Some(DEFAULT_MAX_TOKENS)
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub kv_cache_usage: f32,
    pub queue_depth: usize,
    pub active_requests: usize,
    pub model_name: String,
    pub started_at: u64,
}

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
}
