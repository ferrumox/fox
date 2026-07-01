use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct PullRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct PullStatus {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}
