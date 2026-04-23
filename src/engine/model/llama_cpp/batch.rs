use anyhow::{anyhow, Result};

use crate::engine::ffi;
use crate::engine::model::sampling::{sample_greedy, sample_token, SamplerParams};
use crate::engine::model::{InferenceRequestForModel, Logits};
use crate::engine::mtmd_ffi;

use super::LlamaCppModel;

impl LlamaCppModel {
    pub(super) fn do_prefill(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Copy cached prefix KV data into each request's sequence BEFORE building the batch.
        //
        // We copy positions 0..skip_prefix_tokens-1 (exclusive of the last "cached" position) so
        // that the last prefix token is always re-submitted in the batch below.  This guarantees:
        //   (a) the logits for the boundary position are freshly computed (not stale from seq_cp),
        //   (b) total_tokens is always ≥ 1, avoiding an invalid pos=context_len decode call.
        {
            let ctx_guard = self
                ._ctx
                .lock()
                .map_err(|e| anyhow!("lock poisoned: {}", e))?;
            let ctx = ctx_guard.as_ptr();
            unsafe {
                let mem = ffi::llama_get_memory(ctx as *const _);
                if !mem.is_null() {
                    // Only attempt seq_cp if the memory backend supports it.
                    let can_copy = ffi::llama_memory_can_shift(mem);
                    for req in requests.iter() {
                        if let Some(src) = req.prefix_seq_id {
                            if !can_copy {
                                // Recurrent/hybrid model — seq_cp not supported; skip.
                                continue;
                            }
                            // copy 0..skip-1; skip-1 position will be re-submitted in the batch
                            let copy_end = req.skip_prefix_tokens.saturating_sub(1) as i32;
                            if copy_end > 0 {
                                ffi::llama_memory_seq_cp(mem, src, req.kv_seq_id, 0, copy_end);
                            }
                        }
                    }
                }
            }
        }

        // Effective start of submission: one token before skip_prefix_tokens so the boundary
        // position is always freshly computed (see seq_cp comment above).
        let effective_skip = |r: &InferenceRequestForModel| {
            r.skip_prefix_tokens
                .saturating_sub(1)
                .min(r.prompt_tokens.len())
        };

        let total_tokens: usize = requests
            .iter()
            .map(|r| r.prompt_tokens.len() - effective_skip(r))
            .sum();

        // total_tokens ≥ 1 because effective_skip < prompt_tokens.len() for any non-empty prompt.
        let n_seq_max = requests.len().max(1) as i32;

        let mut batch = unsafe { ffi::llama_batch_init(total_tokens as i32, 0, n_seq_max) };

        let mut batch_logits_indices: Vec<i32> = Vec::with_capacity(requests.len());

        for req in requests.iter() {
            let seq_id = req.kv_seq_id;
            let start = effective_skip(req);
            let tokens_to_submit = &req.prompt_tokens[start..];
            if tokens_to_submit.is_empty() {
                // Should be handled by the total_tokens == 0 branch above, but be defensive.
                batch_logits_indices.push(-1);
                continue;
            }
            for (local_pos, &token) in tokens_to_submit.iter().enumerate() {
                let abs_pos = start + local_pos;
                let idx = batch.n_tokens as usize;
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
                    batch_logits_indices.push(idx as i32);
                }
            }
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
            // tokens_in_kv = tokens actually placed in the KV during this prefill call.
            let tokens_in_kv = req
                .map(|r| r.prompt_tokens.len() - effective_skip(r))
                .unwrap_or(0);
            let batch_idx = batch_logits_indices.get(i).copied().unwrap_or(-1);
            let logits_ptr = if batch_idx >= 0 {
                unsafe { ffi::llama_get_logits_ith(ctx, batch_idx) }
            } else {
                std::ptr::null_mut()
            };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token), tokens_in_kv));
                continue;
            }
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = if let Some(r) = req {
                sample_token(
                    logits_slice,
                    SamplerParams {
                        temperature: r.temperature,
                        top_p: r.top_p,
                        top_k: r.top_k,
                        min_p: r.min_p,
                        repetition_penalty: r.repetition_penalty,
                        frequency_penalty: r.frequency_penalty,
                        presence_penalty: r.presence_penalty,
                        generated_ids: &r.generated_token_ids,
                        seed: r.seed,
                        token_count: r.generated_tokens,
                    },
                )
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled), tokens_in_kv));
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
                sample_token(
                    logits_slice,
                    SamplerParams {
                        temperature: r.temperature,
                        top_p: r.top_p,
                        top_k: r.top_k,
                        min_p: r.min_p,
                        repetition_penalty: r.repetition_penalty,
                        frequency_penalty: r.frequency_penalty,
                        presence_penalty: r.presence_penalty,
                        generated_ids: &r.generated_token_ids,
                        seed: r.seed,
                        token_count: r.generated_tokens,
                    },
                )
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled)));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    pub(super) fn do_get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let n_embd = self.config.num_heads * self.config.head_dim;
        let n_tokens = tokens.len() as i32;

        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, 1) };
        for (i, &token) in tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                *arr.add(0) = 0; // dedicated seq slot for embeddings
                *batch.logits.add(i) = 0i8; // no logits needed for embeddings
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

        let emb_ptr = unsafe { ffi::llama_get_embeddings_seq(ctx, 0) };
        let embeddings = if emb_ptr.is_null() {
            vec![0.0f32; n_embd]
        } else {
            unsafe { std::slice::from_raw_parts(emb_ptr, n_embd) }.to_vec()
        };

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

    pub(super) fn do_vision_prefill(
        &self,
        seq_id: i32,
        text_prompt: &str,
        image_bytes: &[u8],
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> Result<(usize, Logits)> {
        let mtmd_guard = self
            .mtmd_ctx
            .as_ref()
            .ok_or_else(|| anyhow!("no multimodal projector loaded"))?
            .lock()
            .map_err(|e| anyhow!("mtmd lock poisoned: {}", e))?;
        let mtmd_ptr = mtmd_guard.as_ptr();

        let bitmap = unsafe {
            mtmd_ffi::mtmd_helper_bitmap_init_from_buf(
                mtmd_ptr,
                image_bytes.as_ptr(),
                image_bytes.len(),
            )
        };
        if bitmap.is_null() {
            return Err(anyhow!(
                "failed to decode image (invalid or unsupported format)"
            ));
        }

        let prompt_cstr = std::ffi::CString::new(text_prompt)
            .map_err(|_| anyhow!("prompt contains null byte"))?;
        let input_text = mtmd_ffi::mtmd_input_text {
            text: prompt_cstr.as_ptr(),
            add_special: true,
            parse_special: true,
        };

        let chunks = unsafe { mtmd_ffi::mtmd_input_chunks_init() };
        if chunks.is_null() {
            unsafe { mtmd_ffi::mtmd_bitmap_free(bitmap) };
            return Err(anyhow!("failed to allocate mtmd_input_chunks"));
        }

        let bitmaps: [*const mtmd_ffi::mtmd_bitmap; 1] = [bitmap as *const _];
        let tok_ret = unsafe {
            mtmd_ffi::mtmd_tokenize(
                mtmd_ptr,
                chunks,
                &input_text,
                bitmaps.as_ptr() as *mut *const _,
                1,
            )
        };
        if tok_ret != 0 {
            unsafe {
                mtmd_ffi::mtmd_input_chunks_free(chunks);
                mtmd_ffi::mtmd_bitmap_free(bitmap);
            }
            return Err(anyhow!("mtmd_tokenize failed (code {})", tok_ret));
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let lctx = ctx_guard.as_ptr();
        let n_batch = self.effective_ctx as i32;

        let mut new_n_past: i32 = 0;
        // mtmd_ffi and ffi have separate bindgen-generated llama_context types;
        // they're the same C struct so a pointer cast is safe.
        let lctx_mtmd = lctx as *mut mtmd_ffi::llama_context;
        let eval_ret = unsafe {
            mtmd_ffi::mtmd_helper_eval_chunks(
                mtmd_ptr,
                lctx_mtmd,
                chunks,
                0, // n_past = 0 (fresh sequence)
                seq_id,
                n_batch,
                true, // logits_last
                &mut new_n_past,
            )
        };

        unsafe {
            mtmd_ffi::mtmd_input_chunks_free(chunks);
            mtmd_ffi::mtmd_bitmap_free(bitmap);
        }

        if eval_ret != 0 {
            return Err(anyhow!(
                "mtmd_helper_eval_chunks failed (code {})",
                eval_ret
            ));
        }

        let n_vocab = self.config.vocab_size as i32;
        let logits_ptr = unsafe { ffi::llama_get_logits_ith(lctx, -1) };
        if logits_ptr.is_null() {
            return Err(anyhow!("no logits after vision prefill"));
        }
        let logits_slice: &[f32] =
            unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };

        let sampled = sample_token(
            logits_slice,
            SamplerParams {
                temperature,
                top_p,
                top_k,
                min_p: 0.0,
                repetition_penalty,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                generated_ids: &[],
                seed,
                token_count: 0,
            },
        );
        let values = logits_slice.to_vec();

        Ok((new_n_past as usize, Logits::new(values, sampled)))
    }
}
