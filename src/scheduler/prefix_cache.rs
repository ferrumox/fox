use crate::kv_cache::BlockId;

/// One entry in the block-level prefix cache.
///
/// The entry covers the first `block_ids.len() × block_size` tokens of a
/// prompt whose KV state is preserved in llama.cpp under `seq_id`.  Only
/// *complete* blocks are stored — partial trailing blocks are freed when the
/// entry is created.
///
/// `block_ids.len() × block_size` gives the number of tokens covered.
///
/// Keyed by `prompt_block_hashes(prompt_tokens, block_size).last()` — the
/// chain hash of the last complete block, which transitively encodes the full
/// prefix.  A new request with the same first N complete blocks will produce
/// the same hash and will be able to skip re-prefilling those N×block_size
/// tokens.
pub(crate) struct PrefixCacheEntry {
    pub(crate) seq_id: i32,
    pub(crate) block_ids: Vec<BlockId>,
}

/// Process-stable random state for token hashing.
/// Initialized once so `hash_tokens` is deterministic within a single run.
// Used only in tests; #[allow] prevents a spurious dead_code warning.
#[allow(dead_code)]
static HASH_STATE: std::sync::OnceLock<ahash::RandomState> = std::sync::OnceLock::new();

/// Hash a slice of token IDs using ahash (faster and more collision-resistant
/// than DefaultHasher, stable within a single process run).
#[allow(dead_code)]
pub(crate) fn hash_tokens(tokens: &[i32]) -> u64 {
    let state = HASH_STATE.get_or_init(ahash::RandomState::new);
    state.hash_one(tokens)
}
