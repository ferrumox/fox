// Batch types and InferenceRequest for the scheduler.

use tokio::sync::mpsc;

/// All sampling hyper-parameters for a single inference request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature (0 = greedy, 1 = unscaled).
    pub temperature: f32,
    /// Top-p nucleus threshold (1.0 = disabled).
    pub top_p: f32,
    /// Top-K filter (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty applied to already-generated tokens (1.0 = disabled).
    pub repetition_penalty: f32,
    /// Optional RNG seed for reproducible sampling.
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
        }
    }
}

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
    /// Token IDs produced so far (used for repetition penalty).
    pub generated_token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub state: RequestState,
    pub kv_block_ids: Vec<usize>,
    pub response_tx: mpsc::UnboundedSender<Token>,
    pub stop_reason: Option<StopReason>,
    pub sampling: SamplingParams,
    /// Stable sequence ID assigned from the Scheduler's pool when admitted.
    /// Used as the llama.cpp seq_id so KV cache slots are never confused across requests.
    pub kv_seq_id: i32,
}

impl InferenceRequest {
    pub fn new(
        id: u64,
        prompt_tokens: Vec<i32>,
        max_new_tokens: usize,
        sampling: SamplingParams,
        response_tx: mpsc::UnboundedSender<Token>,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            last_token: None,
            generated_tokens: 0,
            generated_token_ids: Vec::new(),
            max_new_tokens,
            state: RequestState::Waiting,
            kv_block_ids: Vec::new(),
            response_tx,
            stop_reason: None,
            sampling,
            kv_seq_id: -1,
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

/// Output of schedule_step: which requests to prefill vs decode,
/// plus the sequence IDs that were just preempted (so the engine can wipe their KV state
/// before those IDs are reassigned to new requests).
#[derive(Debug, Default)]
pub struct ScheduledBatch {
    pub prefill: Vec<u64>,
    pub decode: Vec<u64>,
    /// Sequence IDs whose KV cache must be cleared (request was preempted this step).
    pub preempted_seq_ids: Vec<i32>,
}

impl ScheduledBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}
