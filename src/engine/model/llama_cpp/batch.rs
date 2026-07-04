use std::ffi::CString;

use anyhow::{anyhow, Result};

use crate::engine::ffi;
use crate::engine::model::sampling::{sample_greedy, sample_token, SamplerParams};
use crate::engine::model::{InferenceRequestForModel, Logits, PrefillStep};

use super::LlamaCppModel;

impl LlamaCppModel {
    pub(super) fn do_prefill(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
        max_prefill_chunk: usize,
    ) -> Result<Vec<PrefillStep>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Effective start of a request's FIRST prefill chunk: one token before
        // skip_prefix_tokens so the prefix-cache boundary position is always freshly
        // computed (see seq_cp comment below). For a non-hit request this is 0.
        let effective_skip = |r: &InferenceRequestForModel| {
            r.skip_prefix_tokens
                .saturating_sub(1)
                .min(r.prompt_tokens.len())
        };

        // End (exclusive) of this chunk: advance up to max_prefill_chunk tokens from
        // prefill_pos (0 = unbounded / single-shot), clamped to the prompt length.
        let chunk_end = |r: &InferenceRequestForModel| {
            let start = r.prefill_pos.min(r.prompt_tokens.len());
            if max_prefill_chunk == 0 {
                r.prompt_tokens.len()
            } else {
                (start + max_prefill_chunk).min(r.prompt_tokens.len())
            }
        };

        // Copy cached prefix KV data into each request's sequence BEFORE building the
        // batch — but only on the request's FIRST chunk (prefill_pos == effective_skip),
        // so a chunked prefill never re-copies. We copy positions 0..skip-1 (exclusive
        // of the last "cached" position) so the last prefix token is re-submitted below.
        {
            let ctx_guard = self
                ._ctx
                .lock()
                .map_err(|e| anyhow!("lock poisoned: {}", e))?;
            let ctx = ctx_guard.as_ptr();
            unsafe {
                let mem = ffi::llama_get_memory(ctx as *const _);
                if !mem.is_null() {
                    let can_copy = ffi::llama_memory_can_shift(mem);
                    for req in requests.iter() {
                        if let Some(src) = req.prefix_seq_id {
                            if !can_copy {
                                // Recurrent/hybrid model — seq_cp not supported; skip.
                                continue;
                            }
                            if req.prefill_pos != effective_skip(req) {
                                // Not the first chunk — the copy already happened.
                                continue;
                            }
                            let copy_end = req.skip_prefix_tokens.saturating_sub(1) as i32;
                            if copy_end > 0 {
                                ffi::llama_memory_seq_cp(mem, src, req.kv_seq_id, 0, copy_end);
                            }
                        }
                    }
                }
            }
        }

        // This chunk submits prompt_tokens[prefill_pos .. chunk_end] for each request.
        let total_tokens: usize = requests
            .iter()
            .map(|r| chunk_end(r).saturating_sub(r.prefill_pos.min(r.prompt_tokens.len())))
            .sum();

        // Nothing left to submit (all requests already fully prefilled) — report
        // completion without decoding. Defensive; the scheduler shouldn't emit these.
        if total_tokens == 0 {
            return Ok(req_ids
                .iter()
                .enumerate()
                .map(|(i, &req_id)| PrefillStep {
                    req_id,
                    prefill_pos: requests.get(i).map(|r| r.prompt_tokens.len()).unwrap_or(0),
                    logits: None,
                    tokens_in_kv: 0,
                })
                .collect());
        }

        let n_seq_max = requests.len().max(1) as i32;
        let mut batch = unsafe { ffi::llama_batch_init(total_tokens as i32, 0, n_seq_max) };

        // Exactly one entry per request: the batch index whose logits we sample (the
        // final prompt token), or -1 when this chunk doesn't reach the prompt's end.
        let mut batch_logits_indices: Vec<i32> = Vec::with_capacity(requests.len());

        for req in requests.iter() {
            let seq_id = req.kv_seq_id;
            let start = req.prefill_pos.min(req.prompt_tokens.len());
            let end = chunk_end(req);
            let tokens_to_submit = &req.prompt_tokens[start..end];
            if tokens_to_submit.is_empty() {
                batch_logits_indices.push(-1);
                continue;
            }
            let mut req_logits_idx = -1i32;
            for (local_pos, &token) in tokens_to_submit.iter().enumerate() {
                let abs_pos = start + local_pos;
                let idx = batch.n_tokens as usize;
                // Logits only for the final prompt token — i.e. only on the chunk that
                // reaches the end of the prompt.
                let has_logits = abs_pos == req.prompt_tokens.len() - 1;
                unsafe {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = abs_pos as i32;
                    *batch.n_seq_id.add(idx) = 1;
                    let arr = *batch.seq_id.add(idx);
                    *arr.add(0) = seq_id;
                    *batch.logits.add(idx) = if has_logits { 1i8 } else { 0i8 };
                }
                batch.n_tokens += 1;
                if has_logits {
                    req_logits_idx = idx as i32;
                }
            }
            batch_logits_indices.push(req_logits_idx);
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (i, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(i);
            let new_pos = req.map(chunk_end).unwrap_or(0);
            let complete = req
                .map(|r| new_pos >= r.prompt_tokens.len())
                .unwrap_or(false);
            let batch_idx = batch_logits_indices.get(i).copied().unwrap_or(-1);

            if !complete || batch_idx < 0 {
                // Intermediate chunk — advance the cursor, don't sample yet.
                results.push(PrefillStep {
                    req_id,
                    prefill_pos: new_pos,
                    logits: None,
                    tokens_in_kv: 0,
                });
                continue;
            }

            // Final chunk: total tokens submitted across all chunks equals
            // prompt_len - effective_skip (identical to the single-shot path).
            let tokens_in_kv = req
                .map(|r| r.prompt_tokens.len() - effective_skip(r))
                .unwrap_or(0);

            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, batch_idx) };
            if logits_ptr.is_null() {
                results.push(PrefillStep {
                    req_id,
                    prefill_pos: new_pos,
                    logits: Some(Logits::new(vec![], self.eos_token)),
                    tokens_in_kv,
                });
                continue;
            }
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = if let Some(r) = req {
                self.sample_constrained(r, logits_slice)
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push(PrefillStep {
                req_id,
                prefill_pos: new_pos,
                logits: Some(Logits::new(values, sampled)),
                tokens_in_kv,
            });
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    pub(super) fn do_decode(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let n_tokens = requests.len() as i32;
        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, n_tokens) };

        for (batch_slot, req) in requests.iter().enumerate() {
            let input_token = req
                .last_token
                .or_else(|| req.prompt_tokens.last().copied())
                .unwrap_or(self.eos_token);
            // context_len = prompt_len + generated_tokens (already incremented after prefill),
            // so the correct KV position for this token is context_len - 1.
            let pos = req.context_len as i32 - 1;
            let seq_id = req.kv_seq_id; // stable ID — never the batch slot index

            unsafe {
                *batch.token.add(batch_slot) = input_token;
                *batch.pos.add(batch_slot) = pos;
                *batch.n_seq_id.add(batch_slot) = 1;
                let arr = *batch.seq_id.add(batch_slot);
                *arr.add(0) = seq_id;
                *batch.logits.add(batch_slot) = 1i8;
            }
            batch.n_tokens += 1;
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (out_idx, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(out_idx);
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, out_idx as i32) };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token)));
                continue;
            }
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = if let Some(r) = req {
                self.sample_constrained(r, logits_slice)
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled)));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    /// Sample one token from `logits`, honoring the request's GBNF grammar if it set
    /// one. Without a grammar this is exactly `sample_token`. With a grammar, tokens the
    /// grammar forbids at the current position are masked to `-inf` first (so fox's Rust
    /// sampler can never pick them), then the chosen token is fed back to advance the
    /// grammar state.
    pub(super) fn sample_constrained(&self, req: &InferenceRequestForModel, logits: &[f32]) -> i32 {
        let params = SamplerParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            repetition_penalty: req.repetition_penalty,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            generated_ids: &req.generated_token_ids,
            seed: req.seed,
            token_count: req.generated_tokens,
        };
        let Some(grammar) = req.grammar.as_deref() else {
            return sample_token(logits, params);
        };
        match self.grammar_mask(req.id, grammar, logits) {
            Some(masked) => {
                let tok = sample_token(&masked, params);
                self.grammar_accept(req.id, tok);
                tok
            }
            // Grammar failed to parse — generate unconstrained rather than crash (the
            // API layer validates and rejects bad grammars before they reach here).
            None => sample_token(logits, params),
        }
    }

    /// Mask every token the request's grammar forbids to `-inf`, creating the grammar
    /// sampler lazily on first use. Returns `None` if the grammar string fails to parse.
    fn grammar_mask(&self, req_id: u64, grammar: &str, logits: &[f32]) -> Option<Vec<f32>> {
        if !self.grammars.contains_key(&req_id) {
            let root = CString::new("root").ok()?;
            let gstr = CString::new(grammar).ok()?;
            let smpl = unsafe {
                ffi::llama_sampler_init_grammar(self.vocab, gstr.as_ptr(), root.as_ptr())
            };
            if smpl.is_null() {
                tracing::warn!(
                    request_id = req_id,
                    "GBNF grammar failed to parse — generating unconstrained"
                );
                return None;
            }
            self.grammars
                .insert(req_id, super::GrammarSampler { ptr: smpl });
        }
        let ptr = self.grammars.get(&req_id)?.ptr;
        if ptr.is_null() {
            return None;
        }

        let n = logits.len();
        let mut data: Vec<ffi::llama_token_data> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| ffi::llama_token_data {
                id: i as i32,
                logit: l,
                p: 0.0,
            })
            .collect();
        let mut arr = ffi::llama_token_data_array {
            data: data.as_mut_ptr(),
            size: n,
            selected: -1,
            sorted: false,
        };
        // The grammar sampler sets forbidden tokens' logit to -inf in place.
        unsafe { ffi::llama_sampler_apply(ptr, &mut arr) };

        // Scatter the survivors back by token id; anything the grammar dropped stays
        // -inf and gets probability 0 through fox's softmax, so it can never be sampled.
        let mut masked = vec![f32::NEG_INFINITY; n];
        for k in 0..arr.size {
            let td = unsafe { *arr.data.add(k) };
            let id = td.id as usize;
            if id < n {
                masked[id] = td.logit;
            }
        }
        Some(masked)
    }

    /// Advance the request's grammar state with the token that was actually chosen.
    fn grammar_accept(&self, req_id: u64, token: i32) {
        if let Some(entry) = self.grammars.get(&req_id) {
            if !entry.ptr.is_null() {
                unsafe { ffi::llama_sampler_accept(entry.ptr, token) };
            }
        }
    }

    pub(super) fn do_get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let n_embd = self.config.n_embd;
        let n_tokens = tokens.len() as i32;

        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, 1) };
        for (i, &token) in tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                *arr.add(0) = 0; // dedicated seq slot for embeddings
                *batch.logits.add(i) = 1i8; // mark every token for embedding output (mean-pooled below)
            }
            batch.n_tokens += 1;
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        unsafe { ffi::llama_set_embeddings(ctx, true) };

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe {
                ffi::llama_set_embeddings(ctx, false);
                ffi::llama_batch_free(batch);
            }
            return Err(anyhow!("llama_decode (embeddings) failed: {}", ret));
        }

        // The shared generation context is created with pooling_type = NONE, so
        // seq-pooled embeddings (llama_get_embeddings_seq) are unavailable — it
        // returns NULL. Read the per-token embeddings (each token was marked for
        // output above) via llama_get_embeddings_ith and mean-pool them into one
        // vector, then L2-normalize (the standard sentence-embedding convention).
        let mut pooled = vec![0.0f32; n_embd];
        let mut counted = 0usize;
        for i in 0..n_tokens {
            let ptr = unsafe { ffi::llama_get_embeddings_ith(ctx, i) };
            if ptr.is_null() {
                continue;
            }
            let row = unsafe { std::slice::from_raw_parts(ptr, n_embd) };
            for (acc, &v) in pooled.iter_mut().zip(row) {
                *acc += v;
            }
            counted += 1;
        }
        if counted > 0 {
            let inv = 1.0 / counted as f32;
            for v in pooled.iter_mut() {
                *v *= inv;
            }
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in pooled.iter_mut() {
                    *v /= norm;
                }
            }
        }
        let embeddings = pooled;

        unsafe {
            let mem = ffi::llama_get_memory(ctx as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ffi::llama_set_embeddings(ctx, false);
            ffi::llama_batch_free(batch);
        }

        Ok(embeddings)
    }
}
