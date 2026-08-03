use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize the OpenAI `stop` field which can be either a string or an array of strings.
pub(crate) fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
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
    /// OpenAI's cached-prompt breakdown. Omitted entirely when nothing was cached,
    /// so a response looks exactly as it did before when there is nothing to report.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// `usage.prompt_tokens_details` — how much of the prompt fox did not have to
/// re-prefill because it was still resident in the KV cache. This is the only way a
/// client can observe the KV-reuse work from §1 of
/// `docs/design/llama-server-gap-analysis.md`.
#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

impl Usage {
    /// Build a usage block, attaching `prompt_tokens_details` only when some of the
    /// prompt was actually served from cache.
    pub fn new(prompt_tokens: u32, completion_tokens: u32, cached_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: (cached_tokens > 0)
                .then_some(PromptTokensDetails { cached_tokens }),
        }
    }
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
