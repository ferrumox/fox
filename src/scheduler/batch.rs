// Batch types and InferenceRequest for the scheduler.

use tokio::sync::mpsc;

/// Streaming token payload.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u64,
    pub token_id: i32,
    pub text: String,
    pub is_eos: bool,
    /// Set on the final token of a request; indicates why generation stopped.
    pub stop_reason: Option<StopReason>,
}

/// Stop reason when generation ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    Length,
    Preempt,
}

/// Request state machine: Waiting -> Prefilling -> Decoding -> Finished
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// A single inference request in the scheduler.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: u64,
    pub prompt_tokens: Vec<i32>,
    pub last_token: Option<i32>,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub state: RequestState,
    pub kv_block_ids: Vec<usize>,
    pub response_tx: mpsc::UnboundedSender<Token>,
    pub stop_reason: Option<StopReason>,
    /// Sampling temperature (0 = greedy, >0 = stochastic).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
}

impl InferenceRequest {
    pub fn new(
        id: u64,
        prompt_tokens: Vec<i32>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        response_tx: mpsc::UnboundedSender<Token>,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            last_token: None,
            generated_tokens: 0,
            max_new_tokens,
            state: RequestState::Waiting,
            kv_block_ids: Vec::new(),
            response_tx,
            stop_reason: None,
            temperature,
            top_p,
        }
    }

    /// Total tokens so far (prompt + generated).
    pub fn context_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens
    }

    /// Whether this request has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.state == RequestState::Finished
    }
}

/// Output of schedule_step: which requests to prefill vs decode.
#[derive(Debug, Default)]
pub struct ScheduledBatch {
    pub prefill: Vec<u64>,
    pub decode: Vec<u64>,
}

impl ScheduledBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}
