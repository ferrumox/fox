use std::ffi::CString;

use anyhow::{anyhow, Result};

use crate::engine::ffi;
use crate::engine::model::sampling::{sample_greedy, sample_token, SamplerParams};
use crate::engine::model::{
    InferenceRequestForModel, KvCacheFullAtMinimum, Logits, Model, PrefillStep,
};

use super::LlamaCppModel;

/// Whether a `llama_decode` return code of `ret` on a batch of `len` requests should
/// be retried by bisecting the batch, and if so, the split point (left-half length).
///
/// Per `llama.h`: 0 = success, 1 = "could not find a KV slot for the batch (try
/// reducing the size of the batch or increase the context)", 2 = aborted, -1 =
/// invalid input, < -1 = fatal. Only `1` is retryable this way — 2/-1/<-1 stay
/// immediately fatal. A batch already at `len <= 1` that still returns `1` is a
/// genuine failure, not a "reduce batch size" situation, so it also returns `None`
/// and the caller falls through to its normal error path.
fn bisection_split(ret: i32, len: usize) -> Option<usize> {
    if ret == 1 && len > 1 {
        Some(len / 2)
    } else {
        None
    }
}

/// Allocate an `n_batch` token budget across `desired_lens` (one per request,
/// in submission order), returning each request's actually-allowed length.
/// Once the budget is exhausted, later requests get `0` for this call — never
/// partially over budget, and never negative. This exists because
/// `llama_decode` aborts via `GGML_ASSERT(n_tokens_all <= cparams.n_batch)`
/// with no graceful return code, unlike `ret==1` ("no KV slot"), so the
/// aggregate submitted-token count must never be allowed to exceed `n_batch`
/// in the first place — see the call site in `do_prefill_batch` for why this
/// can happen even with `max_prefill_chunk` already capping each request's
/// own desired length.
fn allocate_batch_budget(desired_lens: &[usize], n_batch: usize) -> Vec<usize> {
    let mut budget = n_batch;
    desired_lens
        .iter()
        .map(|&desired| {
            let allowed = desired.min(budget);
            budget -= allowed;
            allowed
        })
        .collect()
}

/// One group of requests sharing the same LoRA selection (by adapter name — see
/// `LoraSelection`'s doc comment for why equality is name-based). `None` is its
/// own group ("no adapter"). `do_prefill`/`do_decode` decode each group as an
/// independent sub-batch, switching the context's active adapter set between
/// groups — see `docs/design/lora-support.md`.
struct LoraGroup {
    selection: Option<crate::scheduler::LoraSelection>,
    req_ids: Vec<u64>,
    requests: Vec<InferenceRequestForModel>,
}

/// Partition requests into `LoraGroup`s, preserving each request's relative
/// order within its group. Linear (not hash-based) since batch sizes are small
/// (bounded by `--max-batch-size`) and this keeps output order deterministic.
fn group_by_lora(req_ids: &[u64], requests: &[InferenceRequestForModel]) -> Vec<LoraGroup> {
    let mut groups: Vec<LoraGroup> = Vec::new();
    for (&id, req) in req_ids.iter().zip(requests.iter()) {
        let key = req.lora.as_ref().map(|l| l.name.as_str());
        match groups
            .iter_mut()
            .find(|g| g.selection.as_ref().map(|l| l.name.as_str()) == key)
        {
            Some(g) => {
                g.req_ids.push(id);
                g.requests.push(req.clone());
            }
            None => groups.push(LoraGroup {
                selection: req.lora.clone(),
                req_ids: vec![id],
                requests: vec![req.clone()],
            }),
        }
    }
    groups
}

impl LlamaCppModel {
    /// Set the context's active LoRA adapter set to match `selection` (`None`
    /// clears it) before decoding a group's batch. `llama_set_adapters_lora`
    /// only rebuilds the compute graph when the requested set actually differs
    /// from what's already active, so calling this unconditionally per group —
    /// even every step, even for the common all-`None` case — is cheap; fox
    /// doesn't need to track "did it change" itself.
    fn apply_lora_group(&self, selection: Option<&crate::scheduler::LoraSelection>) -> Result<()> {
        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();
        let ret = match selection {
            Some(sel) => {
                let (adapter, _) = self.lora_adapters.get(&sel.name).ok_or_else(|| {
                    anyhow!("LoRA adapter '{}' is not loaded on this model", sel.name)
                })?;
                let mut adapters = [adapter.as_ptr()];
                let mut scales = [sel.scale];
                unsafe {
                    ffi::llama_set_adapters_lora(ctx, adapters.as_mut_ptr(), 1, scales.as_mut_ptr())
                }
            }
            None => unsafe {
                ffi::llama_set_adapters_lora(ctx, std::ptr::null_mut(), 0, std::ptr::null_mut())
            },
        };
        if ret != 0 {
            anyhow::bail!("llama_set_adapters_lora failed (code {ret})");
        }
        Ok(())
    }

    pub(super) fn do_prefill(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
        max_prefill_chunk: usize,
    ) -> Result<Vec<PrefillStep>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Multimodal requests are prefilled atomically via mtmd_helper_eval_chunks
        // (its own internal llama_decode calls) — they never join the shared token
        // batch below, so split them out first. No fox-level chunking or bisection
        // retry for this path in v1 (see docs/design/vision-support.md).
        let mut mm_results = Vec::new();
        for (&req_id, req) in req_ids.iter().zip(requests.iter()) {
            if req.multimodal.is_some() {
                mm_results.push(self.do_prefill_multimodal(req_id, req)?);
            }
        }
        if mm_results.len() == requests.len() {
            return Ok(mm_results);
        }
        let (req_ids, requests): (Vec<u64>, Vec<InferenceRequestForModel>) = req_ids
            .iter()
            .zip(requests.iter())
            .filter(|(_, r)| r.multimodal.is_none())
            .map(|(&id, r)| (id, r.clone()))
            .unzip();
        let req_ids = &req_ids[..];
        let requests = &requests[..];

        // Text requests: group by LoRA selection (switching the context's active
        // adapter set is a whole-context operation — see docs/design/lora-support.md
        // — so requests using different adapters can never share one llama_decode
        // call) and prefill each group independently.
        let mut results = mm_results;
        for group in group_by_lora(req_ids, requests) {
            self.apply_lora_group(group.selection.as_ref())?;
            results.extend(self.do_prefill_batch(
                &group.req_ids,
                &group.requests,
                max_prefill_chunk,
            )?);
        }
        Ok(results)
    }

    /// The actual token-batch prefill for one group of requests that all share the
    /// same LoRA selection (already applied to the context by the caller). Never
    /// called directly outside `do_prefill` — see that function's LoRA-group split.
    fn do_prefill_batch(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
        max_prefill_chunk: usize,
    ) -> Result<Vec<PrefillStep>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Start of a request's FIRST prefill chunk. `skip_prefix_tokens` is already the
        // exact count of resident tokens the scheduler guaranteed: it applies
        // llama-server's [TAG_PROMPT_LOGITS] guard at admission (stepping back one
        // token when the prompt is *entirely* resident, so at least one token still
        // decodes and produces logits) and trims the sequence to exactly that point
        // via `ScheduledBatch::kv_trims`. No extra `-1` here — that adjustment used to
        // live in three separate places and each had to agree with the others.
        let effective_skip =
            |r: &InferenceRequestForModel| r.skip_prefix_tokens.min(r.prompt_tokens.len());

        // Desired end (exclusive) of this chunk before the n_batch cap below: advance
        // up to max_prefill_chunk tokens from prefill_pos (0 = unbounded / single-shot),
        // clamped to the prompt length.
        let desired_chunk_end = |r: &InferenceRequestForModel| {
            let start = r.prefill_pos.min(r.prompt_tokens.len());
            if max_prefill_chunk == 0 {
                r.prompt_tokens.len()
            } else {
                (start + max_prefill_chunk).min(r.prompt_tokens.len())
            }
        };

        // llama.cpp aborts via GGML_ASSERT(n_tokens_all <= cparams.n_batch) — unlike
        // ret==1 ("no KV slot"), there is no graceful return code for this, so it must
        // never be reached rather than retried. `max_prefill_chunk` alone doesn't
        // guarantee this: it caps one request's own chunk, but several requests
        // admitted into the same step each contribute their own chunk to the SAME
        // batch, and their sum is what n_batch actually bounds (also possible with a
        // single request if max_prefill_chunk itself exceeds n_batch — a small
        // --max-context-len shrinks n_batch below the 512-token default). Allocate the
        // n_batch budget across requests in submission order; once exhausted, later
        // requests in this call submit zero tokens this step — their `prefill_pos`
        // doesn't advance, so the scheduler re-offers them next step, exactly the
        // existing mechanism a single request's own multi-step chunking already relies
        // on (see the empty-`tokens_to_submit` handling below).
        let n_batch = unsafe {
            let ctx_guard = self
                ._ctx
                .lock()
                .map_err(|e| anyhow!("lock poisoned: {}", e))?;
            ffi::llama_n_batch(ctx_guard.as_ptr() as *const _) as usize
        };
        let desired_lens: Vec<usize> = requests
            .iter()
            .map(|r| {
                let start = r.prefill_pos.min(r.prompt_tokens.len());
                desired_chunk_end(r).saturating_sub(start)
            })
            .collect();
        let allowed_lens = allocate_batch_budget(&desired_lens, n_batch);
        let chunk_ends: Vec<usize> = requests
            .iter()
            .zip(allowed_lens.iter())
            .map(|(r, &allowed_len)| r.prefill_pos.min(r.prompt_tokens.len()) + allowed_len)
            .collect();
        let chunk_end = |i: usize| chunk_ends[i];

        // Copy a sibling sequence's KV into this request's sequence BEFORE building the
        // batch — but only on the request's FIRST chunk (prefill_pos == effective_skip),
        // so a chunked prefill never re-copies. Copies exactly positions
        // `[0, skip_prefix_tokens)`, matching what `effective_skip` then declines to
        // resubmit; the scheduler already guaranteed `skip_prefix_tokens` leaves at
        // least one token to decode.
        //
        // Currently unreachable (`prefix_seq_id` is always `None`) — a request inherits
        // its predecessor's seq_id outright rather than copying out of it. Kept for the
        // shared-prefill fork of `n>1`/`best_of`, which needs a copy because the source
        // is a *live* sibling. See docs/design/llama-server-gap-analysis.md §0.3.
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
                            let copy_end = req.skip_prefix_tokens as i32;
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
            .enumerate()
            .map(|(i, r)| chunk_end(i).saturating_sub(r.prefill_pos.min(r.prompt_tokens.len())))
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

        for (i, req) in requests.iter().enumerate() {
            let seq_id = req.kv_seq_id;
            let start = req.prefill_pos.min(req.prompt_tokens.len());
            let end = chunk_end(i);
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
        // Keep the MTP driver in step with the target: it drafts from the hidden rows
        // of this decode. No-op when no MTP head is attached.
        #[cfg(fox_mtp)]
        if ret == 0 {
            unsafe { self.mtp_process(&batch) };
        }
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            if let Some(split) = bisection_split(ret, requests.len()) {
                // Release the mutex before recursing — it isn't reentrant, and each
                // recursive call re-acquires its own guard. Each half re-runs the
                // prefix-donation seq_cp above; safe/idempotent (same source range,
                // prefill_pos hasn't advanced since this failed call never returned),
                // just a small redundant cost on the rare retry path.
                drop(ctx_guard);
                self.decode_bisection_retries
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    batch_len = requests.len(),
                    split,
                    "llama_decode (prefill): no KV slot for batch (ret=1) — retrying as two smaller batches"
                );
                let (ids_a, ids_b) = req_ids.split_at(split);
                let (req_a, req_b) = requests.split_at(split);
                let mut results = self.do_prefill_batch(ids_a, req_a, max_prefill_chunk)?;
                results.extend(self.do_prefill_batch(ids_b, req_b, max_prefill_chunk)?);
                return Ok(results);
            }
            return Err(anyhow!(
                "llama_decode failed: {} (batch size {})",
                ret,
                requests.len()
            ));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (i, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(i);
            let new_pos = req.map(|_| chunk_end(i)).unwrap_or(0);
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

            // Final chunk: the KV now holds the WHOLE prompt (donated prefix +
            // submitted chunks). The scheduler derives the next decode position from
            // this, so it must be the total — never the submitted count.
            let tokens_in_kv = req.map(|r| r.prompt_tokens.len()).unwrap_or(0);

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
            let needs_logits = req.map(|r| r.needs_logits).unwrap_or(false);
            let values: Vec<f32> = if needs_logits {
                logits_slice.to_vec()
            } else {
                Vec::new()
            };
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

    /// Prefill a single multimodal request atomically via `mtmd_helper_eval_chunks`,
    /// which runs mtmd's own internal `llama_decode` calls for whatever mix of
    /// text/image chunks the prompt contains. Never joins the shared token batch in
    /// `do_prefill` above — mtmd owns `_ctx` for the duration of the call, so no
    /// other request's decode/prefill step interleaves with it (a documented v1
    /// tradeoff; see `docs/design/vision-support.md`). Also unlike `do_prefill`,
    /// there is no fox-level chunking (the whole prompt is one atomic call) and no
    /// bisection retry on OOM — a failure here surfaces as a normal `EngineError`.
    fn do_prefill_multimodal(
        &self,
        req_id: u64,
        req: &InferenceRequestForModel,
    ) -> Result<PrefillStep> {
        let mtmd_ctx = self
            .mtmd_ctx
            .ok_or_else(|| anyhow!("multimodal request but model has no mmproj loaded"))?;
        let chunks = req
            .multimodal
            .as_ref()
            .ok_or_else(|| anyhow!("do_prefill_multimodal called without multimodal chunks"))?;

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        // n_past is always 0: multimodal requests skip fox's chunked-prefill and
        // prefix-cache machinery entirely (see prompt_tokens being left empty for
        // them), so this is always a fresh sequence's first and only prefill call.
        let mut new_n_past: i32 = 0;
        let ret = unsafe {
            ffi::mtmd_helper_eval_chunks(
                mtmd_ctx.as_ptr(),
                ctx,
                chunks.as_raw() as *const ffi::mtmd_input_chunks,
                0,
                req.kv_seq_id,
                self.effective_ctx as i32,
                true, // logits_last
                &mut new_n_past,
            )
        };
        if ret != 0 {
            return Err(anyhow!("mtmd_helper_eval_chunks failed (code {ret})"));
        }

        let n_vocab = self.config.vocab_size as i32;
        let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, -1) };
        let logits = if logits_ptr.is_null() {
            Logits::new(vec![], self.eos_token)
        } else {
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = self.sample_constrained(req, logits_slice);
            Logits::new(logits_slice.to_vec(), sampled)
        };

        Ok(PrefillStep {
            req_id,
            prefill_pos: new_n_past as usize,
            logits: Some(logits),
            tokens_in_kv: new_n_past as usize,
        })
    }

    pub(super) fn do_decode(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Group by LoRA selection — same reasoning as do_prefill: the adapter set
        // is a whole-context property, so requests using different adapters can
        // never share one llama_decode call. Unlike prefill, decode has no
        // multimodal split (decode is always plain token-based, regardless of
        // whether the request's *first* turn was multimodal).
        let mut results = Vec::with_capacity(requests.len());
        for group in group_by_lora(req_ids, requests) {
            self.apply_lora_group(group.selection.as_ref())?;
            results.extend(self.do_decode_batch(&group.req_ids, &group.requests)?);
        }
        Ok(results)
    }

    /// The actual token-batch decode for one group of requests that all share the
    /// same LoRA selection (already applied to the context by the caller). Never
    /// called directly outside `do_decode` — see that function's LoRA-group split.
    fn do_decode_batch(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let n_tokens = requests.len() as i32;
        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, n_tokens) };

        // Emit the batch in ascending kv_seq_id order. llama.cpp's `split_equal`
        // (used for every decode whenever the KV cache is non-unified, i.e.
        // `n_stream > 1` — see `llama-kv-cache.cpp`'s
        // `n_stream == 1 ? split_simple : split_equal`) walks the batch in order and
        // only keeps growing the current ubatch while
        // `batch.seq_id[i][0] == last_seq_id + 1` (`llama-batch.cpp`). Emitting in
        // scheduler-admission order instead means a 4-sequence decode batch is split
        // into four 1-token ubatches, and the GPU matmul degrades to `ncols_dst=1` —
        // real batching on fox's side, none on llama.cpp's. Combined with the
        // scheduler's min-first seq_id pool (which keeps the live IDs dense), this
        // hands llama.cpp one full-width ubatch, matching what `llama-server` gets
        // from iterating its slots in `slot.id` order.
        //
        // `order` maps batch position -> index into `requests`/`req_ids`; the logits
        // read below inverts it via `slot_of` so the returned order is unchanged.
        let mut order: Vec<usize> = (0..requests.len()).collect();
        order.sort_by_key(|&i| requests[i].kv_seq_id);
        let mut slot_of = vec![0usize; requests.len()];
        for (batch_slot, &orig_idx) in order.iter().enumerate() {
            slot_of[orig_idx] = batch_slot;
        }

        for (batch_slot, &orig_idx) in order.iter().enumerate() {
            let req = &requests[orig_idx];
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
        // Keep the MTP driver in step with the target: it drafts from the hidden rows
        // of this decode. No-op when no MTP head is attached.
        #[cfg(fox_mtp)]
        if ret == 0 {
            unsafe { self.mtp_process(&batch) };
        }
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            if let Some(split) = bisection_split(ret, requests.len()) {
                // Release the mutex before recursing — it isn't reentrant, and each
                // recursive call re-acquires its own guard.
                drop(ctx_guard);
                self.decode_bisection_retries
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                tracing::warn!(
                    batch_len = requests.len(),
                    split,
                    "llama_decode: no KV slot for batch (ret=1) — retrying as two smaller batches"
                );
                let (ids_a, ids_b) = req_ids.split_at(split);
                let (req_a, req_b) = requests.split_at(split);
                let mut results = self.do_decode_batch(ids_a, req_a)?;
                results.extend(self.do_decode_batch(ids_b, req_b)?);
                return Ok(results);
            }
            // Bisection is exhausted (batch already at size 1) and it's
            // specifically "no KV slot" (ret==1), not some other fatal code —
            // signal which request this is so the engine layer (which owns
            // the scheduler + context-shift config this layer doesn't have)
            // can try one reactive context roll before giving up. See
            // docs/design/reactive-context-rolling.md.
            if ret == 1 && requests.len() == 1 {
                return Err(anyhow::Error::new(KvCacheFullAtMinimum {
                    req_id: req_ids[0],
                }));
            }
            return Err(anyhow!(
                "llama_decode failed: {} (batch size {})",
                ret,
                requests.len()
            ));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (out_idx, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(out_idx);
            // Undo the ascending-seq_id emission order: this request's logits live at
            // the batch position it was written to, not at its index in `req_ids`.
            let batch_slot = slot_of.get(out_idx).copied().unwrap_or(out_idx);
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, batch_slot as i32) };
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
            // Copying the full vocab-sized logits vector is only useful when the
            // caller actually reads it (`logprobs`) — skip it otherwise, since
            // it's a real per-token cost on real vocabs (128K+ entries).
            let needs_logits = req.map(|r| r.needs_logits).unwrap_or(false);
            let values: Vec<f32> = if needs_logits {
                logits_slice.to_vec()
            } else {
                Vec::new()
            };
            results.push((req_id, Logits::new(values, sampled)));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    /// Sample one token from `logits`, honoring the request's GBNF grammar (mask
    /// forbidden tokens, then advance the grammar with the pick), `min_tokens` (suppress
    /// end-of-generation until the floor is reached), and the sampler knobs (`min_p`,
    /// `logit_bias`, temperature, top-k/p, penalties). Without a grammar or an active
    /// EOG suppression it is exactly `sample_token`.
    pub(super) fn sample_constrained(&self, req: &InferenceRequestForModel, logits: &[f32]) -> i32 {
        let params = Self::sampler_params(req, &req.generated_token_ids, req.generated_tokens);
        let grammar = req.grammar.as_deref();
        let suppress_eog = req.min_tokens > req.generated_tokens && !self.eog_tokens.is_empty();

        // Fast path: no grammar and nothing to mask → sample the raw logits directly.
        if grammar.is_none() && !suppress_eog {
            return sample_token(logits, params);
        }

        // Working buffer: grammar-masked logits when a grammar is set (plain copy if it
        // failed to parse), else a plain copy.
        let mut buf = match grammar {
            Some(g) => self
                .grammar_mask(req.id, g, logits)
                .unwrap_or_else(|| logits.to_vec()),
            None => logits.to_vec(),
        };
        if suppress_eog {
            self.mask_eog(&mut buf);
        }

        let tok = sample_token(&buf, params);
        if grammar.is_some() {
            self.grammar_accept(req.id, tok);
        }
        tok
    }

    /// Build `SamplerParams` from a request, with an explicit penalty context and
    /// token count (they advance per position during speculative verification).
    fn sampler_params<'a>(
        req: &'a InferenceRequestForModel,
        generated: &'a [i32],
        token_count: usize,
    ) -> SamplerParams<'a> {
        SamplerParams {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repetition_penalty: req.repetition_penalty,
            frequency_penalty: req.frequency_penalty,
            presence_penalty: req.presence_penalty,
            repeat_last_n: req.repeat_last_n,
            top_n_sigma: req.top_n_sigma,
            min_keep: req.min_keep,
            logit_bias: req.logit_bias.as_deref(),
            generated_ids: generated,
            seed: req.seed,
            token_count,
        }
    }

    /// Mask every end-of-generation token to `-inf` (min_tokens suppression).
    fn mask_eog(&self, buf: &mut [f32]) {
        for &id in &self.eog_tokens {
            if (id as usize) < buf.len() {
                buf[id as usize] = f32::NEG_INFINITY;
            }
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

    /// One speculative-decoding step: verify the given `drafts` (already proposed by
    /// an `engine::speculative::Proposer` — n-gram lookup or a draft model) all in a
    /// single `llama_decode`, and commit the longest prefix the model would itself
    /// have produced. Returns the committed tokens — always ≥ 1 (the ordinary next
    /// token), plus any accepted drafts and one bonus token.
    ///
    /// Exact regardless of proposer: each committed token is a genuine target sample
    /// conditioned on the accepted prefix, so with a fixed seed the output is
    /// byte-identical to non-speculative decoding — a wrong draft is simply rejected.
    /// Grammar is not applied here — the engine only speculates when no grammar is
    /// active.
    pub(super) fn do_speculative_decode(
        &self,
        req: &InferenceRequestForModel,
        drafts: Vec<i32>,
    ) -> Result<Vec<Logits>> {
        // Full logical sequence so far (prompt + generated) — used below only to
        // find the true last token; the drafts themselves are supplied by the caller.
        let mut seq: Vec<i32> =
            Vec::with_capacity(req.prompt_tokens.len() + req.generated_token_ids.len());
        seq.extend_from_slice(&req.prompt_tokens);
        seq.extend_from_slice(&req.generated_token_ids);

        let last_token = req
            .last_token
            .or_else(|| seq.last().copied())
            .unwrap_or(self.eos_token);
        let base_pos = req.context_len as i32 - 1;
        let seq_id = req.kv_seq_id;

        let n = 1 + drafts.len(); // last_token + drafts

        // Verify batch: last_token then each draft at consecutive positions, all with logits.
        let mut batch = unsafe { ffi::llama_batch_init(n as i32, 0, 1) };
        let tokens_in: Vec<i32> = std::iter::once(last_token)
            .chain(drafts.iter().copied())
            .collect();
        for (i, &tok) in tokens_in.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = base_pos + i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                *arr.add(0) = seq_id;
                *batch.logits.add(i) = 1i8;
            }
        }
        batch.n_tokens = n as i32;

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();
        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        // Keep the MTP driver in step with the target: it drafts from the hidden rows
        // of this decode. No-op when no MTP head is attached.
        #[cfg(fox_mtp)]
        if ret == 0 {
            unsafe { self.mtp_process(&batch) };
        }
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode (speculative) failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size;
        let mut committed: Vec<Logits> = Vec::with_capacity(n);
        // Penalty context grows with each committed token so per-position sampling matches
        // the non-speculative path exactly (repetition/frequency/presence penalties).
        let mut running_generated = req.generated_token_ids.clone();
        let mut accepted = 0usize;
        for i in 0..n {
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, i as i32) };
            if logits_ptr.is_null() {
                break;
            }
            let logits = unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab) };
            let gen_count = req.generated_tokens + i;
            let tok = self.sample_verify_position(req, logits, &running_generated, gen_count);
            committed.push(Logits::new(logits.to_vec(), tok));
            running_generated.push(tok);
            // Accept only if this matches the drafted token; otherwise this is the bonus /
            // first-mismatch token and we stop.
            if i < drafts.len() && tok == drafts[i] {
                accepted += 1;
            } else {
                break;
            }
        }

        // Keep last_token + accepted drafts (positions base_pos ..= base_pos+accepted);
        // drop the rejected draft tail from the KV cache.
        let keep_end = base_pos + accepted as i32 + 1;
        unsafe {
            let mem = ffi::llama_get_memory(ctx as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_rm(mem, seq_id, keep_end, -1);
            }
        }
        unsafe { ffi::llama_batch_free(batch) };

        if committed.is_empty() {
            return Err(anyhow!("speculative decode produced no token"));
        }
        Ok(committed)
    }

    /// Draft-model proposal step (0.16): feed `new_tokens` into this model's own KV
    /// at `seq_id` (starting at `base_pos`), then greedily decode up to `draft_len`
    /// further tokens, extending the same KV sequence. No penalty context and no
    /// grammar — a draft only needs to be a plausible proposer, the target's verify
    /// step is what determines correctness. Stops early on an end-of-generation token.
    /// Caller (`engine::speculative::DraftModelProposer`) owns `seq_id` exclusively
    /// and is responsible for trimming/clearing it between requests.
    pub(super) fn do_draft_propose(
        &self,
        seq_id: i32,
        new_tokens: &[i32],
        base_pos: i32,
        draft_len: usize,
    ) -> Vec<i32> {
        // Nothing new to feed means nothing to base a proposal on — this call doesn't
        // persist logits across calls. The caller always has ≥ 1 new token in
        // practice (each decode step commits at least one real token).
        if draft_len == 0 || new_tokens.is_empty() {
            return Vec::new();
        }
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let ctx = ctx_guard.as_ptr();
        let n_vocab = self.config.vocab_size;

        // Feed `new_tokens`, requesting logits only at the last position — that's
        // all that's needed to pick the first draft token.
        let n = new_tokens.len();
        let mut batch = unsafe { ffi::llama_batch_init(n as i32, 0, 1) };
        for (i, &tok) in new_tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = base_pos + i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                *arr.add(0) = seq_id;
                *batch.logits.add(i) = i8::from(i == n - 1);
            }
        }
        batch.n_tokens = n as i32;
        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Vec::new();
        }
        let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, (n - 1) as i32) };
        if logits_ptr.is_null() {
            unsafe { ffi::llama_batch_free(batch) };
            return Vec::new();
        }
        let mut logits = unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab) }.to_vec();
        unsafe { ffi::llama_batch_free(batch) };
        let mut next_pos = base_pos + n as i32;

        let mut drafts = Vec::with_capacity(draft_len);
        loop {
            let tok = sample_greedy(&logits);
            if self.is_eog_token(tok) {
                break;
            }
            drafts.push(tok);
            if drafts.len() >= draft_len {
                // Reached the cap — skip decoding one more token purely to prep
                // logits for a position we'll never use.
                break;
            }

            // Decode this one new draft token to get logits for the next position.
            let mut batch = unsafe { ffi::llama_batch_init(1, 0, 1) };
            unsafe {
                *batch.token.add(0) = tok;
                *batch.pos.add(0) = next_pos;
                *batch.n_seq_id.add(0) = 1;
                let arr = *batch.seq_id.add(0);
                *arr.add(0) = seq_id;
                *batch.logits.add(0) = 1;
            }
            batch.n_tokens = 1;
            let ret = unsafe { ffi::llama_decode(ctx, batch) };
            next_pos += 1;
            if ret != 0 {
                unsafe { ffi::llama_batch_free(batch) };
                break;
            }
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, 0) };
            if logits_ptr.is_null() {
                unsafe { ffi::llama_batch_free(batch) };
                break;
            }
            logits = unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab) }.to_vec();
            unsafe { ffi::llama_batch_free(batch) };
        }

        drafts
    }

    /// Sample the target at one verify position, mirroring `sample_constrained` minus the
    /// grammar (speculation runs only when no grammar is active): `min_p` / `logit_bias`
    /// via `SamplerParams`, and EOG suppression below `min_tokens`. `generated` is the
    /// penalty context up to (not including) this position.
    fn sample_verify_position(
        &self,
        req: &InferenceRequestForModel,
        logits: &[f32],
        generated: &[i32],
        gen_count: usize,
    ) -> i32 {
        let params = Self::sampler_params(req, generated, gen_count);
        if req.min_tokens > gen_count && !self.eog_tokens.is_empty() {
            let mut buf = logits.to_vec();
            self.mask_eog(&mut buf);
            sample_token(&buf, params)
        } else {
            sample_token(logits, params)
        }
    }

    /// Score one (query, document) pair via the model's classification head.
    ///
    /// Deliberately does NOT reuse `do_get_embeddings`: that one marks every token
    /// for output and mean-pools by hand precisely because the generation context
    /// resolves to `pooling_type = NONE`. A reranker resolves to `RANK` instead, where
    /// llama.cpp writes one score per *sequence* and only `llama_get_embeddings_seq`
    /// can read it. A NULL return there is the model telling us it is not a reranker.
    pub(super) fn do_rerank_score(&self, tokens: &[i32]) -> Result<f32> {
        if tokens.is_empty() {
            return Err(anyhow!("cannot score an empty sequence"));
        }
        let n_tokens = tokens.len() as i32;

        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, 1) };
        for (i, &token) in tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                // Same dedicated slot as embeddings: outside the scheduler's pool, so
                // scoring can never clobber a live generation's KV.
                *arr.add(0) = self.embed_seq_id;
                // RANK pooling reduces over the sequence; only the last token needs
                // its output flagged.
                *batch.logits.add(i) = i8::from(i + 1 == tokens.len());
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

        let cleanup = |ctx: *mut ffi::llama_context| unsafe {
            let mem = ffi::llama_get_memory(ctx as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_rm(mem, self.embed_seq_id, 0, -1);
            }
            ffi::llama_set_embeddings(ctx, false);
            ffi::llama_batch_free(batch);
        };

        if ret != 0 {
            cleanup(ctx);
            return Err(anyhow!("llama_decode (rerank) failed: {}", ret));
        }

        let ptr = unsafe { ffi::llama_get_embeddings_seq(ctx, self.embed_seq_id) };
        if ptr.is_null() {
            cleanup(ctx);
            return Err(anyhow!(
                "model has no reranking head — llama.cpp resolved its pooling type to \
                 something other than RANK, so there is no sequence score to read"
            ));
        }
        let score = unsafe { *ptr };
        cleanup(ctx);
        Ok(score)
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
                // Dedicated embeddings slot, OUTSIDE the scheduler's seq pool: writing
                // (and wiping, below) a pool id would clobber a live generation's KV.
                *arr.add(0) = self.embed_seq_id;
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
                ffi::llama_memory_seq_rm(mem, self.embed_seq_id, 0, -1);
            }
            ffi::llama_set_embeddings(ctx, false);
            ffi::llama_batch_free(batch);
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use super::{allocate_batch_budget, bisection_split};

    #[test]
    fn allocate_batch_budget_grants_everyone_when_budget_is_ample() {
        assert_eq!(allocate_batch_budget(&[10, 20, 5], 100), vec![10, 20, 5]);
    }

    #[test]
    fn allocate_batch_budget_starves_later_requests_when_exhausted() {
        // 100 budget: req 0 gets all 60 it wants, req 1 gets only 40 of its 80,
        // req 2 (already out of budget) gets 0 and must wait for the next step.
        assert_eq!(allocate_batch_budget(&[60, 80, 30], 100), vec![60, 40, 0]);
    }

    #[test]
    fn allocate_batch_budget_caps_a_single_request_alone() {
        // Confirms the fix also covers max_prefill_chunk itself exceeding
        // n_batch with only one request in the call — no concurrency needed.
        assert_eq!(allocate_batch_budget(&[512], 64), vec![64]);
    }

    #[test]
    fn allocate_batch_budget_zero_n_batch_starves_everyone() {
        assert_eq!(allocate_batch_budget(&[10, 20], 0), vec![0, 0]);
    }

    #[test]
    fn allocate_batch_budget_empty_input() {
        assert_eq!(allocate_batch_budget(&[], 100), Vec::<usize>::new());
    }

    #[test]
    fn allocate_batch_budget_never_exceeds_n_batch_in_aggregate() {
        let allowed = allocate_batch_budget(&[36; 8], 64);
        assert_eq!(allowed.iter().sum::<usize>(), 64);
        // Exactly matches the crash this fix closes: 8 requests x 36 tokens
        // (288) vs n_batch=64 — every allowed length together must total
        // exactly the budget (fully used, never exceeded).
    }

    #[test]
    fn ret_one_splits_a_batch_in_half() {
        assert_eq!(bisection_split(1, 4), Some(2));
    }

    #[test]
    fn ret_one_splits_an_odd_batch_favoring_the_left_half() {
        assert_eq!(bisection_split(1, 3), Some(1));
    }

    #[test]
    fn ret_one_on_a_single_request_is_not_retryable() {
        // Already the smallest possible batch — "reduce batch size" doesn't apply.
        assert_eq!(bisection_split(1, 1), None);
    }

    #[test]
    fn ret_one_on_an_empty_batch_is_not_retryable() {
        // Defensive: both do_prefill/do_decode already early-return before decoding
        // an empty request list, so this should never occur in practice.
        assert_eq!(bisection_split(1, 0), None);
    }

    #[test]
    fn success_is_never_retried() {
        assert_eq!(bisection_split(0, 4), None);
    }

    #[test]
    fn other_return_codes_are_never_retried() {
        // 2 = aborted, -1 = invalid input, < -1 = fatal — none are "reduce batch size".
        for ret in [2, -1, -5] {
            assert_eq!(
                bisection_split(ret, 4),
                None,
                "ret={ret} must not be retried"
            );
        }
    }
}
