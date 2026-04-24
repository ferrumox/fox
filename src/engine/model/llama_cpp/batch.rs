use anyhow::{anyhow, Result};

use crate::engine::ffi;
use crate::engine::model::sampling::{sample_greedy, sample_token, SamplerParams};
use crate::engine::model::{
    InferenceRequestForModel, Logits, PreprocessedVision, VisionDecodeParams,
};
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

    /// Phase 1: Preprocess a vision request off the inference thread.
    /// Decodes image, tokenizes text+image, and runs CLIP encoding.
    /// Only needs an mtmd context (acquired from pool), not llama_context.
    pub(super) fn do_vision_preprocess(
        &self,
        text_prompt: &str,
        image_bytes: &[u8],
    ) -> Result<PreprocessedVision> {
        let pool = self
            .mtmd_pool
            .as_ref()
            .ok_or_else(|| anyhow!("no multimodal projector loaded"))?;

        let mtmd_ptr_nn = pool.acquire();
        let mtmd_ptr = mtmd_ptr_nn.as_ptr();

        let result = (|| -> Result<PreprocessedVision> {
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

            unsafe { mtmd_ffi::mtmd_bitmap_free(bitmap) };

            if tok_ret != 0 {
                unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                return Err(anyhow!("mtmd_tokenize failed (code {})", tok_ret));
            }

            // CLIP-encode each image/audio chunk and copy embeddings out.
            let n_chunks = unsafe { mtmd_ffi::mtmd_input_chunks_size(chunks) };
            let n_embd_inp = pool.n_embd_inp;
            let mut image_embeddings: Vec<(usize, std::sync::Arc<Vec<f32>>)> = Vec::new();

            for i in 0..n_chunks {
                let chunk = unsafe { mtmd_ffi::mtmd_input_chunks_get(chunks, i) };
                let chunk_type = unsafe { mtmd_ffi::mtmd_input_chunk_get_type(chunk) };
                // MTMD_INPUT_CHUNK_TYPE_TEXT = 0, IMAGE = 1, AUDIO = 2
                if chunk_type == 0 {
                    continue;
                }

                let encode_ret = unsafe { mtmd_ffi::mtmd_encode_chunk(mtmd_ptr, chunk) };
                if encode_ret != 0 {
                    unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                    return Err(anyhow!("mtmd_encode_chunk failed (code {})", encode_ret));
                }

                let n_tokens = unsafe { mtmd_ffi::mtmd_input_chunk_get_n_tokens(chunk) } as usize;
                let embd_ptr = unsafe { mtmd_ffi::mtmd_get_output_embd(mtmd_ptr) };
                if embd_ptr.is_null() {
                    unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                    return Err(anyhow!("mtmd_get_output_embd returned null"));
                }

                let embd_len = n_embd_inp * n_tokens;
                let embd_slice = unsafe { std::slice::from_raw_parts(embd_ptr, embd_len) };
                image_embeddings.push((i, std::sync::Arc::new(embd_slice.to_vec())));
            }

            Ok(PreprocessedVision {
                chunks: chunks as *mut std::ffi::c_void,
                image_embeddings,
            })
        })();

        pool.release(mtmd_ptr_nn);
        result
    }

    /// Phase 2: Decode preprocessed vision chunks into the LLM's KV cache.
    /// Uses pre-encoded CLIP embeddings — runs on the inference thread with llama_context.
    /// Single-request vision decode: inlines image chunk decode without mtmd pool.
    pub(super) fn do_vision_decode_prefill(
        &self,
        seq_id: i32,
        preprocessed: &PreprocessedVision,
        temperature: f32,
        top_p: f32,
        top_k: u32,
        repetition_penalty: f32,
        seed: Option<u64>,
    ) -> Result<(usize, Logits)> {
        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let lctx = ctx_guard.as_ptr();
        let n_batch = self.effective_ctx as i32;
        let n_mmproj_embd = self.n_embd_inp;

        let chunks = preprocessed.chunks as *mut mtmd_ffi::mtmd_input_chunks;
        let n_chunks = unsafe { mtmd_ffi::mtmd_input_chunks_size(chunks) };
        let mut n_past: i32 = 0;

        let mut embd_map: std::collections::HashMap<usize, &Vec<f32>> =
            std::collections::HashMap::new();
        for (idx, embd) in &preprocessed.image_embeddings {
            embd_map.insert(*idx, embd.as_ref());
        }

        for i in 0..n_chunks {
            let chunk = unsafe { mtmd_ffi::mtmd_input_chunks_get(chunks, i) };
            let chunk_type = unsafe { mtmd_ffi::mtmd_input_chunk_get_type(chunk) };
            let is_last = i == n_chunks - 1;

            if chunk_type == 0 {
                let mut n_tokens_out: usize = 0;
                let tokens_ptr = unsafe {
                    mtmd_ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut n_tokens_out)
                };
                if tokens_ptr.is_null() || n_tokens_out == 0 {
                    continue;
                }

                let text_batch = unsafe { ffi::llama_batch_init(n_batch, 0, 1) };
                let tokens = unsafe { std::slice::from_raw_parts(tokens_ptr, n_tokens_out) };

                let mut ti = 0usize;
                while ti < n_tokens_out {
                    let batch_ptr = &text_batch as *const _ as *mut ffi::llama_batch;
                    unsafe { (*batch_ptr).n_tokens = 0 };
                    while ti < n_tokens_out
                        && (unsafe { (*batch_ptr).n_tokens } as i32) < n_batch
                    {
                        let j = unsafe { (*batch_ptr).n_tokens } as usize;
                        unsafe {
                            *(*batch_ptr).token.add(j) = tokens[ti];
                            *(*batch_ptr).pos.add(j) = n_past;
                            *(*batch_ptr).n_seq_id.add(j) = 1;
                            *(*(*batch_ptr).seq_id.add(j)) = seq_id;
                            *(*batch_ptr).logits.add(j) = 0;
                            (*batch_ptr).n_tokens += 1;
                        }
                        n_past += 1;
                        ti += 1;
                    }
                    let at_end = ti == n_tokens_out;
                    if is_last && at_end {
                        let n = unsafe { (*batch_ptr).n_tokens } as usize;
                        if n > 0 {
                            unsafe { *(*batch_ptr).logits.add(n - 1) = 1 };
                        }
                    }
                    let ret = unsafe { ffi::llama_decode(lctx, text_batch) };
                    if ret != 0 {
                        unsafe { ffi::llama_batch_free(text_batch) };
                        return Err(anyhow!("llama_decode failed on text chunk (code {})", ret));
                    }
                }
                unsafe { ffi::llama_batch_free(text_batch) };
            } else {
                let embd = embd_map
                    .get(&i)
                    .ok_or_else(|| anyhow!("missing pre-encoded embeddings for chunk {}", i))?;
                let n_tokens =
                    unsafe { mtmd_ffi::mtmd_input_chunk_get_n_tokens(chunk) } as i32;

                if self.vision_use_non_causal {
                    unsafe { ffi::llama_set_causal_attn(lctx, false) };
                }
                let mut embd_offset = 0i32;
                while embd_offset < n_tokens {
                    let batch_len = (n_tokens - embd_offset).min(n_batch);
                    let mut pos_arr: Vec<i32> =
                        (0..batch_len).map(|j| n_past + embd_offset + j).collect();
                    let mut n_seq_arr: Vec<i32> = vec![1; batch_len as usize];
                    let mut seq_id_val: i32 = seq_id;
                    let mut seq_ptrs: Vec<*mut i32> =
                        vec![&mut seq_id_val as *mut i32; batch_len as usize];
                    let mut logits_arr: Vec<i8> = vec![0; batch_len as usize];
                    let batch = ffi::llama_batch {
                        n_tokens: batch_len,
                        token: std::ptr::null_mut(),
                        embd: embd.as_ptr().wrapping_add(
                            (embd_offset as usize) * n_mmproj_embd,
                        ) as *mut f32,
                        pos: pos_arr.as_mut_ptr(),
                        n_seq_id: n_seq_arr.as_mut_ptr(),
                        seq_id: seq_ptrs.as_mut_ptr(),
                        logits: logits_arr.as_mut_ptr(),
                    };
                    let ret = unsafe { ffi::llama_decode(lctx, batch) };
                    if ret != 0 {
                        if self.vision_use_non_causal {
                            unsafe { ffi::llama_set_causal_attn(lctx, true) };
                        }
                        return Err(anyhow!("llama_decode failed on image chunk (code {})", ret));
                    }
                    embd_offset += batch_len;
                }
                if self.vision_use_non_causal {
                    unsafe { ffi::llama_set_causal_attn(lctx, true) };
                }
                n_past += unsafe { mtmd_ffi::mtmd_input_chunk_get_n_pos(chunk) };
            }
        }

        let logits_ptr = unsafe { ffi::llama_get_logits_ith(lctx, -1) };
        if logits_ptr.is_null() {
            return Err(anyhow!("no logits after vision decode prefill"));
        }
        let n_vocab = self.config.vocab_size as i32;
        let logits_slice = unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
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
        Ok((n_past as usize, Logits::new(values, sampled)))
    }

    /// Preprocess with optional cached CLIP embeddings.
    /// If cached embeddings are provided, skips the expensive mtmd_encode_chunk call.
    pub(super) fn do_vision_preprocess_with_cache(
        &self,
        text_prompt: &str,
        image_bytes: &[u8],
        cached_embeddings: Option<std::sync::Arc<Vec<f32>>>,
    ) -> Result<PreprocessedVision> {
        let pool = self
            .mtmd_pool
            .as_ref()
            .ok_or_else(|| anyhow!("no multimodal projector loaded"))?;

        let mtmd_ptr_nn = pool.acquire();
        let mtmd_ptr = mtmd_ptr_nn.as_ptr();

        let result = (|| -> Result<PreprocessedVision> {
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

            unsafe { mtmd_ffi::mtmd_bitmap_free(bitmap) };

            if tok_ret != 0 {
                unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                return Err(anyhow!("mtmd_tokenize failed (code {})", tok_ret));
            }

            let n_chunks = unsafe { mtmd_ffi::mtmd_input_chunks_size(chunks) };
            let n_embd_inp = pool.n_embd_inp;
            let mut image_embeddings: Vec<(usize, std::sync::Arc<Vec<f32>>)> = Vec::new();

            for i in 0..n_chunks {
                let chunk = unsafe { mtmd_ffi::mtmd_input_chunks_get(chunks, i) };
                let chunk_type = unsafe { mtmd_ffi::mtmd_input_chunk_get_type(chunk) };
                if chunk_type == 0 {
                    continue;
                }

                if let Some(ref cached) = cached_embeddings {
                    image_embeddings.push((i, std::sync::Arc::clone(cached)));
                    tracing::debug!(chunk = i, "CLIP cache hit — reusing embeddings");
                } else {
                    let encode_ret = unsafe { mtmd_ffi::mtmd_encode_chunk(mtmd_ptr, chunk) };
                    if encode_ret != 0 {
                        unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                        return Err(anyhow!("mtmd_encode_chunk failed (code {})", encode_ret));
                    }

                    let n_tokens =
                        unsafe { mtmd_ffi::mtmd_input_chunk_get_n_tokens(chunk) } as usize;
                    let embd_ptr = unsafe { mtmd_ffi::mtmd_get_output_embd(mtmd_ptr) };
                    if embd_ptr.is_null() {
                        unsafe { mtmd_ffi::mtmd_input_chunks_free(chunks) };
                        return Err(anyhow!("mtmd_get_output_embd returned null"));
                    }

                    let embd_len = n_embd_inp * n_tokens;
                    let embd_slice = unsafe { std::slice::from_raw_parts(embd_ptr, embd_len) };
                    image_embeddings.push((i, std::sync::Arc::new(embd_slice.to_vec())));
                }
            }

            Ok(PreprocessedVision {
                chunks: chunks as *mut std::ffi::c_void,
                image_embeddings,
            })
        })();

        pool.release(mtmd_ptr_nn);
        result
    }

    /// Batch vision decode: acquires llama_context lock once for all requests.
    /// No mtmd pool context needed — image chunk decode is inlined.
    pub(super) fn do_vision_decode_prefill_batch(
        &self,
        params: Vec<VisionDecodeParams>,
    ) -> Result<Vec<(usize, Logits)>> {
        if params.is_empty() {
            return Ok(vec![]);
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let lctx = ctx_guard.as_ptr();
        let n_batch = self.effective_ctx as i32;
        let n_vocab = self.config.vocab_size as i32;
        let n_mmproj_embd = self.n_embd_inp;

        let mut results = Vec::with_capacity(params.len());

        for p in &params {
            let chunks = p.preprocessed.chunks as *mut mtmd_ffi::mtmd_input_chunks;
            let n_chunks = unsafe { mtmd_ffi::mtmd_input_chunks_size(chunks) };
            let mut n_past: i32 = 0;

            let mut embd_map: std::collections::HashMap<usize, &Vec<f32>> =
                std::collections::HashMap::new();
            for (idx, embd) in &p.preprocessed.image_embeddings {
                embd_map.insert(*idx, embd.as_ref());
            }

            for i in 0..n_chunks {
                let chunk = unsafe { mtmd_ffi::mtmd_input_chunks_get(chunks, i) };
                let chunk_type = unsafe { mtmd_ffi::mtmd_input_chunk_get_type(chunk) };
                let is_last = i == n_chunks - 1;

                if chunk_type == 0 {
                    let mut n_tokens_out: usize = 0;
                    let tokens_ptr = unsafe {
                        mtmd_ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut n_tokens_out)
                    };
                    if tokens_ptr.is_null() || n_tokens_out == 0 {
                        continue;
                    }

                    let text_batch = unsafe { ffi::llama_batch_init(n_batch, 0, 1) };
                    let tokens =
                        unsafe { std::slice::from_raw_parts(tokens_ptr, n_tokens_out) };

                    let mut ti = 0usize;
                    while ti < n_tokens_out {
                        let batch_ptr = &text_batch as *const _ as *mut ffi::llama_batch;
                        unsafe { (*batch_ptr).n_tokens = 0 };

                        while ti < n_tokens_out
                            && (unsafe { (*batch_ptr).n_tokens } as i32) < n_batch
                        {
                            let j = unsafe { (*batch_ptr).n_tokens } as usize;
                            unsafe {
                                *(*batch_ptr).token.add(j) = tokens[ti];
                                *(*batch_ptr).pos.add(j) = n_past;
                                *(*batch_ptr).n_seq_id.add(j) = 1;
                                *(*(*batch_ptr).seq_id.add(j)) = p.seq_id;
                                *(*batch_ptr).logits.add(j) = 0;
                                (*batch_ptr).n_tokens += 1;
                            }
                            n_past += 1;
                            ti += 1;
                        }

                        let at_end = ti == n_tokens_out;
                        if is_last && at_end {
                            let n = unsafe { (*batch_ptr).n_tokens } as usize;
                            if n > 0 {
                                unsafe {
                                    *(*batch_ptr).logits.add(n - 1) = 1;
                                }
                            }
                        }

                        let ret = unsafe { ffi::llama_decode(lctx, text_batch) };
                        if ret != 0 {
                            unsafe { ffi::llama_batch_free(text_batch) };
                            return Err(anyhow!(
                                "llama_decode failed on text chunk (code {})",
                                ret
                            ));
                        }
                    }
                    unsafe { ffi::llama_batch_free(text_batch) };
                } else {
                    let embd = embd_map.get(&i).ok_or_else(|| {
                        anyhow!("missing pre-encoded embeddings for chunk {}", i)
                    })?;

                    let n_tokens =
                        unsafe { mtmd_ffi::mtmd_input_chunk_get_n_tokens(chunk) } as i32;

                    if self.vision_use_non_causal {
                        unsafe { ffi::llama_set_causal_attn(lctx, false) };
                    }

                    // Inline image embedding decode: create embedding batch and call llama_decode.
                    let mut embd_offset = 0i32;
                    while embd_offset < n_tokens {
                        let batch_len = (n_tokens - embd_offset).min(n_batch);
                        let embd_batch = ffi::llama_batch {
                            n_tokens: batch_len,
                            token: std::ptr::null_mut(),
                            embd: embd.as_ptr().wrapping_add(
                                (embd_offset as usize) * n_mmproj_embd,
                            ) as *mut f32,
                            pos: std::ptr::null_mut(),
                            n_seq_id: std::ptr::null_mut(),
                            seq_id: std::ptr::null_mut(),
                            logits: std::ptr::null_mut(),
                        };

                        // Build position/seq arrays on the stack for this sub-batch.
                        let mut pos_arr: Vec<i32> =
                            (0..batch_len).map(|j| n_past + embd_offset + j).collect();
                        let mut n_seq_arr: Vec<i32> = vec![1; batch_len as usize];
                        let mut seq_id_val: i32 = p.seq_id;
                        let mut seq_ptrs: Vec<*mut i32> =
                            vec![&mut seq_id_val as *mut i32; batch_len as usize];
                        let mut logits_arr: Vec<i8> = vec![0; batch_len as usize];

                        let batch_with_meta = ffi::llama_batch {
                            n_tokens: batch_len,
                            token: std::ptr::null_mut(),
                            embd: embd.as_ptr().wrapping_add(
                                (embd_offset as usize) * n_mmproj_embd,
                            ) as *mut f32,
                            pos: pos_arr.as_mut_ptr(),
                            n_seq_id: n_seq_arr.as_mut_ptr(),
                            seq_id: seq_ptrs.as_mut_ptr(),
                            logits: logits_arr.as_mut_ptr(),
                        };
                        let _ = embd_batch; // suppress unused

                        let ret = unsafe { ffi::llama_decode(lctx, batch_with_meta) };
                        if ret != 0 {
                            if self.vision_use_non_causal {
                                unsafe { ffi::llama_set_causal_attn(lctx, true) };
                            }
                            return Err(anyhow!(
                                "llama_decode failed on image chunk (code {})",
                                ret
                            ));
                        }
                        embd_offset += batch_len;
                    }

                    if self.vision_use_non_causal {
                        unsafe { ffi::llama_set_causal_attn(lctx, true) };
                    }
                    n_past += unsafe { mtmd_ffi::mtmd_input_chunk_get_n_pos(chunk) };
                }
            }

            let logits_ptr = unsafe { ffi::llama_get_logits_ith(lctx, -1) };
            if logits_ptr.is_null() {
                return Err(anyhow!("no logits after vision decode prefill"));
            }
            let logits_slice =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };

            let sampled = sample_token(
                logits_slice,
                SamplerParams {
                    temperature: p.temperature,
                    top_p: p.top_p,
                    top_k: p.top_k,
                    min_p: 0.0,
                    repetition_penalty: p.repetition_penalty,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    generated_ids: &[],
                    seed: p.seed,
                    token_count: 0,
                },
            );
            let values = logits_slice.to_vec();
            results.push((n_past as usize, super::Logits::new(values, sampled)));
        }

        Ok(results)
    }

    /// Combined vision prefill (backward compat): preprocess + decode in one call.
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
        let preprocessed = self.do_vision_preprocess(text_prompt, image_bytes)?;
        self.do_vision_decode_prefill(
            seq_id,
            &preprocessed,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            seed,
        )
    }
}
