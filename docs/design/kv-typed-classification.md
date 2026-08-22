# Typed KV lifetimes, phase 1: does the discipline pay for itself?

**Verdict: no, not as scoped.** Of twelve verified KV lifecycle bugs in fox's
history, **two** would have failed to compile under a typed API. Six more are
catchable only as a *class*, and only after the invariant had already been
discovered by a failing production server. Four are not type-shaped at all.

Against the decision rule agreed before the classification started — 4 or 5 of
~6–8 rejected justifies the refactor, 1 kills it — this is **in between and near
the bottom**: 2 of 12 hard, ~5 of 12 if every "class" answer is generously scored
at half. The rule's threshold is ~60–70% of the set; this is ~42% at its most
flattering and 17% at its strictest.

There is a sharper finding underneath the count, and it is the reason the answer is
"no" rather than "maybe":

> **The design proposed for phase 2 catches none of them.** `Owned`/`Shared` block
> handles with copy-on-write protect *writes through block handles*. In fox,
> nothing writes through a block handle. Both bugs that a type system would have
> stopped are in the **sequence** lifecycle, not the block pool.

That redirection is cheap enough to have been done rather than proposed: §8 records
the `SeqId` newtype that shipped with this document, the `-1` sentinel it replaced,
and the limit that only appeared by trying it — a newtype does not reach through a
bindgen pointer, so it took a helper function, not a type, to make `a4171eb` stop
compiling.

Details, evidence and the two directions of the argument follow.

---

## 0. Why the block half is vacuous in fox

fox's `scheduler/slots.rs` says it in its own words:

> fox's `BlockId`s never reach llama.cpp […] llama.cpp allocates its own cells by
> `(seq_id, pos)`. So a slot's `blocks` is only *how much pool budget this sequence
> is charged*, and block identity is irrelevant.

Three consequences, all checkable in the tree today:

1. **No block is ever written to.** Writes go through `llama_decode` with
   `(seq_id, pos)` pairs. A `Shared` type whose selling point is the absence of a
   `write` method removes a method that has no callers.
2. `KVCacheManager::copy_on_write` and `is_shared` have **zero callers** outside
   their own module (`grep` confirms). `f41b223` deleted the last one on purpose,
   and its commit message explains why: privatising a block would "allocate budget
   for memory that nobody occupies — while copying no KV, because at this layer
   there is no KV to copy."
3. The invariant that licenses that deletion — *only whole blocks below the
   divergence point are shared, so a shared block never receives a write* — is
   therefore not enforceable by the type. There is no write to reject.

So the phase-2 design as written types the budget ledger, and the bugs live in the
resource the ledger is counting. The typed block pool is real and it works (see
§4), but in fox it is a solution looking for its problem.

---

## 1. The set

Twelve bugs, after verifying the candidate list. Every one read from the diff, not
the subject line. Paired commits are split where they fixed two independent bugs,
because scoring them as one would let a hit carry a miss.

| # | commit | what actually went wrong | verdict |
|---|---|---|---|
| 1 | `3f9b6e8` (b) | error paths returned a `seq_id` to the pool without `clear_sequence` — every later occupant of that sequence failed too, forever | **B — impossible by construction** |
| 2 | `a4171eb` (b) | `do_get_embeddings` hardcoded `seq_id = 0`, then `seq_rm`'d it; the pool hands out `0..max_batch` | **A — would not compile** |
| 3 | `3f9b6e8` (a) | a donated sequence kept its whole tail (prompt + generation) while the cache entry promised only the prefix | C — class |
| 4 | `076d16a` | three reuse paths, one guard: the scheduler skipped prefill and marked tokens resident that nobody had copied; llama.cpp aborted | C — class (strongest) |
| 5 | `f9d051a` | the checkpoint was taken on eviction, so the blob held prompt **plus** response instead of the prompt boundary | C — class |
| 6 | `a4171eb` (a) | a request that had rolled its context donated anyway; cells at `[0, cached)` were mid-generation tokens, not the prefix the key promised | C — class |
| 7 | `5141b1c` (a) | a preempted request lost its KV but kept its generated-token counter; re-prefill of the bare prompt left a positional gap | C — class |
| 8 | `5141b1c` (b) | admission livelock: newcomer and runner evicted each other every step | **D — no** |
| 9 | `7f735e0` | `trim_sequence` returns `true` without rewinding once the tail cell is invalidated; reuse was accepted over that lie | **D — no** |
| 10 | `d5140b0` | `prefilled_tokens` recorded the *submitted* count, not the KV's true length; first decode wrote into occupied cells | **D — no** |
| 11 | `173c4d7` | `llama_memory_seq_cp` on a recurrent backend → SIGABRT | C — class (expensive) |
| 12 | `f41b223` + `aa54463` | each sharer of a prefix still reserved its own blocks for it: 282 blocks held where 72 were in use | **D — no** |

**A** = the buggy line is a type error, with no prior knowledge of the invariant.
**B** = the omission becomes unwriteable (linear handle + `Drop`), also with no prior
knowledge. **C** = the family is caught once someone encodes the rule, which means
the rule had to be found first. **D** = not caught.

**Tally: A 1, B 1, C 6, D 4.**

### Excluded, and why

- `42e4416` — the prompt cache picked a checkpoint by insertion order. This is a
  **selection-policy and determinism** bug: every candidate was a valid, live,
  correctly-accounted checkpoint, and the fix is a tie-break comparator. Nothing
  about ownership, sharing or ordering. It was on the candidate list; it does not
  belong there.
- `06aaf6d` — a false positive, per `STATUS.md:146`. Not a bug.
- `9c8d7d6`, `901b079`, `f7ad9c9` — memory **sizing**, excluded by the brief.
- `e672837` — rolling fired exactly *at* `n_ctx` while a speculative verify batch
  needs `draft_len + 1` cells. Capacity arithmetic; same category as sizing.
- `7f241f8`, `1c36faf` — seq_id *ordering* for ubatch merging. Throughput, not
  lifetime.
- `5b882e2`, `91a7641` — terminal-token and error signalling, not KV.

Three bugs the candidate list did not have (`076d16a`, `f9d051a`, and both halves
of `3f9b6e8`/`a4171eb` scored separately) came out of the history sweep. Two of
those additions are the only hard hits in the whole set — see §5.

---

## 2. The two that would not have compiled

### #2 — `a4171eb`, embeddings on sequence 0

```c
*arr.add(0) = 0; // dedicated seq slot for embeddings
```

A comment asserting a fact that was false. `seq_id` is an `i32` everywhere it
travels, so the literal `0` is as valid a sequence as any the pool issued. The
scheduler hands out `0..max_batch` and issues `0` **last**, which is why nothing
below full concurrency ever hit it. Under load, an embedding request wrote a live
generation's KV and then wiped it.

If a sequence id can only come from the pool, that line is a type error:

```rust
place_token(&mut batch, 0, 0);
//                      ^ expected `&SeqLease<'_>`, found integer
```

Verified as a `compile_fail` doctest on `resident::place_token`. Nothing had to be
known in advance about embeddings, about concurrency, or about which id the pool
issues last. The bug is *unwriteable*, which is the strongest form of the claim
this exercise was testing.

### #1 — `3f9b6e8`, the poisoned sequence

The prefill and decode error paths returned the `seq_id` to the pool without
clearing its llama.cpp KV. The next request assigned that sequence collided with
the leftover cells and failed too — and so did the next, permanently.

This is the textbook linear-resource bug: an early return that skips a cleanup
call. A `SeqLease` that clears in `Drop` makes the skip impossible, because there is
no `return` that bypasses a destructor:

```rust
fn failing_decode(pool: &SeqPool) -> Result<(), ()> {
    let _lease = pool.claim().ok_or(())?;
    Err(())            // and the sequence is cleared anyway
}
```

Two honest caveats. `Drop` does not *reject* anything — it makes the correct thing
automatic, which is arguably stronger but is a different mechanism, and `mem::forget`
and reference cycles are safe Rust, so the guarantee is "no accidental omission",
not "no omission". Scored as a hit regardless: the bug was an accidental omission.

---

## 3. The six that need the invariant found first

This is the category that decides the exercise, so it needs stating precisely.

A typestate encodes a rule. It does not discover one. `Track<Rolled>` has no
`donate` method **because someone already knew that rolling invalidates donation**
— and knowing that is exactly what it took to write `if req.rolled_tokens > 0 {
return None; }`. For a bug's first occurrence, the type system is not in the room.

What it does buy is real but narrower: the rule is then enforced at every call site
including ones written later, by people who never read the commit that found it.

- **#6 `a4171eb`(a)** and **#5 `f9d051a`** are the cleanest typestate shapes:
  `checkpoint` and `donate` exist on `Track<Prefilled>` and on nothing else.
  Both `compile_fail` doctests are in the crate. Both invariants were found by
  running a real server.
- **#3 `3f9b6e8`(a)** is the same shape at one remove: the cache entry's promise
  ("this holds exactly the prefix") and the sequence's actual extent were two
  claims that nothing tied together. A constructor `fn donate(seq: Trimmed) ->
  Entry` ties them. Again: someone had to notice the tail was there.
- **#7 `5141b1c`(a)** is caught only if the position counters live *inside* the
  KV-owning handle, so that losing the KV loses them. That is a defensible design
  and the types make it natural — but nothing in the type system forces the
  counter to move there, and in fox it is also what the client sees, so it has an
  independent reason to live on the request.
- **#11 `173c4d7`** needs a *capability parameter*: `share_prefix` exists only for
  a memory backend that supports `seq_cp`. Expressible — but the model is chosen
  at runtime from a GGUF file, so it means splitting the engine into two
  monomorphised worlds at load time. Large cost, one bug.
- **#4 `076d16a`** is the best argument in the whole set, and it is worth separating
  from the rest. The capability check already existed and was already known to be
  necessary — it had been added in `173c4d7`. The bug was that it was enforced in
  **one of three** reuse paths, so the other two skipped prefill and marked tokens
  resident that nobody had copied. Here the type system's actual strength applies
  with no discovery caveat: if the only way to obtain a shared prefix is a
  constructor that demands the capability, "resident but never copied" is
  unrepresentable and all three paths are forced through the same door. This is
  enforcement propagation, which is what types are genuinely good at.

If one bug in twelve justified the refactor, it would be this one. One does not.

---

## 4. The four that are not type-shaped

- **#9 `7f735e0`** — the fix is an inequality: `resident_at_trim - n_past > budget`.
  Rust has no dependent types, which the brief accepts up front, so a rollback
  *distance* cannot be a type. Worth dwelling on because it directly contradicts
  the prior estimate: a `Rewindable` marker is the wrong shape here. Rewindability
  on a recurrent cache is a **quantity** (`n_rs_seq` snapshots), not a capability.
  A type that removed `trim` from recurrent sequences would forbid exactly the
  in-budget reuse the fix deliberately preserves — 151 of a 173-token prompt still
  cached on turn 3. The one sub-part that *is* typestate-shaped (a restored blob
  has no snapshot history, so its budget is zero) is not method-absence either:
  on an attention cache a restored blob may still be trimmed, so it is a
  conditional, not a missing method.
- **#10 `d5140b0`** — `prefilled_tokens` held the submitted count where the KV's
  length was meant. Pure position arithmetic. The only structural mitigation is
  to derive the extent from the table rather than store it alongside — a
  data-modelling choice available today, with or without new types.
- **#12 `f41b223`/`aa54463`** — nothing in a type system forbids allocating blocks
  you did not need. `Owned`/`Shared` makes correct sharing *expressible*; it cannot
  make over-reservation *inexpressible*. This scores below the "half" the prior
  estimate gave it. (It is also a budget over-count, not corruption — arguably not
  a lifecycle bug at all.)
- **#8 `5141b1c`(b)** — a livelock. No type system in any language rejects one.

---

## 5. Against the prior estimate

The estimate written before the classification, and how it held:

| estimate | outcome |
|---|---|
| "el orden de los recortes se atrapa entero" | **Partly.** The typestate holds (§3), but the trim-order bugs are all first-occurrence discoveries, so "entero" overstates it. Class, not capture. |
| "el retroceso de recurrentes … sí como clase (`trim` sólo en secuencias `Rewindable`)" | **No, not even as a class.** Rewindability is a quantity here; a `Rewindable` marker either forbids the reuse the fix preserves or is vacuous. §4. |
| "la contabilidad duplicada sólo a medias" | **Less than half.** The types cannot reject over-allocation at all. §4. |

All three came out weaker than estimated, and the two hard hits — hardcoded
sequence 0, and the unclear-on-error sequence — appear nowhere in it. So the
classification is not a restatement of the prior; where it agrees it agrees more
weakly, and where it disagrees it disagrees in the direction that hurts the thesis.

---

## 6. Costs the design does not mention

Three, all found by writing the types rather than by reasoning about them.

1. **Linear handles need interior mutability on the pool.** A handle that frees
   itself does so in `Drop`, `Drop` gets `&self`, so the pool must be reachable
   through a shared reference and therefore carry a `RefCell` — a `Mutex` in fox,
   where the scheduler is concurrent and this is the hot path. The runtime failure
   mode removed from the call sites reappears, smaller, at the pool. `Pool` in
   `resident.rs` is written this way and the doc comment says so.
2. **`'p` is viral.** Every struct that stores a handle acquires a lifetime
   parameter. In fox that reaches `InferenceRequest`, the running batch, and the
   slot table — structures that live in a `Mutex` inside an `Arc<Scheduler>` and
   are handed to an async engine loop. Self-referential-struct territory, i.e.
   arenas or indices, i.e. back to ids.
3. **`compile_fail` is coarse evidence.** A doctest passes if the snippet fails to
   compile for *any* reason. Each of the seven in `resident.rs` was checked against
   real `rustc` output for the intended error code (E0599 twice, E0308, E0382 twice,
   E0597, and the missing-`write` E0599). Anyone extending them should do the same.

---

## 7. The alternative that already exists

`make e2e` — a real server, a real model, driven over HTTP — found bugs #1, #3, #6,
#7 and #10, and is the reason #12's metric lie was noticed. It was built *because*
of this bug family and it now gates every release.

That matters for the comparison. The question is not "types versus nothing", it is
"types versus the gate that already catches most of this". Types would have caught
two of twelve; the e2e gate caught five and continues to. A month spent widening
`make e2e` (concurrent embeddings alongside generation would have caught #2 the day
it landed) buys more per unit of effort than a month spent on `'p` parameters.

---

## 8. What was done instead

Not the phase-2 refactor. The two hard hits both live in the sequence lifetime, so
that is what got typed — hours, not weeks, and independent of any thesis. This
landed alongside the classification:

**`src/seq.rs` — a `SeqId` newtype with no public constructor.** Minting happens in
exactly two named, `pub(crate)` places: `SeqId::slot(index)`, which only the slot
table calls, and `SeqId::dedicated(raw)` for the two sequences that legitimately sit
outside the slot range (the embeddings slot, and the draft model's own sequence in
its own llama.cpp context). Every `Model` trait method that addresses a sequence —
`clear_sequence`, `trim_sequence`, `copy_sequence_range`, `roll_context`,
`state_seq_save`/`load`, `mtp_*`, `draft_propose` — now takes `SeqId`, and `.raw()`
is called at the FFI boundary and nowhere else.

**The `-1` sentinel is gone.** `InferenceRequest::kv_seq_id` is `Option<SeqId>`, so
"parked, the slot owns the KV now" is a variant rather than a magic number. Every
`if req.kv_seq_id >= 0` became a `let Some(seq_id) = …`, which is the same check the
compiler now insists on. `ScheduledBatch`'s `kv_trims` / `kv_clears` / `kv_saves` /
`kv_restores` / `preempted_seq_ids`, `prefix_seq_id` and `fork_source` are typed too.

### The limit that only showed up by trying it

The newtype **on its own did not stop `a4171eb`.** Reintroducing the bug verbatim
still compiled:

```rust
*arr.add(0) = 0; // dedicated seq slot for embeddings
```

`llama_batch` is a bag of bindgen pointers. A newtype guards Rust function
signatures; it does not reach through a `*mut *mut i32`. Assuming otherwise would
have been exactly the kind of unearned claim this exercise exists to catch.

What closes it is a helper — `set_batch_row(&batch, idx, token, pos, seq: SeqId,
wants_logits)` — now the single sanctioned way to fill a batch row, used at all seven
sites that used to hand-roll the pointer arithmetic. Through it, the bug is a type
error:

```
error[E0308]: mismatched types: expected `SeqId`, found integer
```

Stated honestly: a raw store into `batch.seq_id` remains *expressible*. What changed
is that writing it means hand-rolling pointer arithmetic next to a helper that
already exists — visible in a diff, rather than invisible inside a comment asserting
the slot is dedicated. That is a weaker guarantee than "unwriteable", and it is the
guarantee that is actually available at an FFI boundary.

### `#[must_use]`, and what it found

`Model::trim_sequence` now carries `#[must_use]` with the reason from `f5214df` item
3 (`true` is not proof the rollback happened; `false` means re-prefill). It
immediately flagged a second, live instance of the same bug: `DraftModelProposer`
discarded the result when trimming last round's unconfirmed speculative tail. A
refused trim there leaves the tail resident, so the next `draft_propose` feeds tokens
at positions that already hold cells. Now it wipes and re-syncs — one lost round of
drafting instead of drafts built on a state the model never reached.

One attribute, one real find. This is the cheapest thing in the whole document.

### What did NOT change

**`KVCacheManager` is untouched.** It is not where these bugs are. Its two dead
methods (`copy_on_write`, `is_shared` — zero callers since `f41b223`) are left alone
rather than built upon; deleting them would be honest housekeeping for another day.

### Verification

`make ci` green (367 lib + 43 integration + 32 `fox-offload` + 11 doctests, clippy
`-D warnings`, both the stub and the real llama.cpp build). And the e2e gate, which
is what actually covers this family — `make e2e` on Qwen2.5-7B-Instruct-Q4_K_M,
**23/23**, including check 13 (embeddings then chat, the `a4171eb` path), check 9
(four concurrent clients), check 12 (mid-stream disconnect), check 17 (identical
prompt ×6 with EOS armed) and speculative decoding, which exercises the trim fix.

## 9. If the paper still wants this section

Write it with this result, not around it. "Two of twelve production bugs in a real
inference server would have been compile errors; six more require encoding an
invariant that a failing server had to discover first; the mechanism the design
centres on turned out to guard an operation the system does not perform" is a more
interesting and more credible finding than a favourable count would have been.

Add the FFI limit, because it generalises past fox: at a C boundary a newtype buys
*narrowed provenance*, not impossibility, and the difference is measurable — it took
a helper function, not a type, to make the bug stop compiling. The `compile_fail`
evidence in `crates/fox-offload/src/resident.rs` supports the rest — the types work;
they are pointed at the wrong resource.
