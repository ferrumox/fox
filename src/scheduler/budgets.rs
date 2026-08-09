//! Deterministic performance budgets for the scheduler.
//!
//! # Why this exists
//!
//! Everything fox claims over `llama-server` is a claim about *work avoided*: a
//! burst behind a shared system prompt prefills that prompt once, a second chat
//! turn re-reads nothing, and sequences sharing a prefix are charged for it once.
//! Every one of those is a refactor away from silently disappearing, and until
//! this file existed nothing in CI would have noticed — `cargo test` asserts the
//! reuse path is *correct*, never that it is *taken*.
//!
//! # Why counters and not milliseconds
//!
//! A shared GitHub runner cannot measure a second reliably, and a timing gate
//! that flakes teaches everyone to re-run it until green, which is worse than no
//! gate. These numbers are counts — tokens submitted to the model, KV blocks
//! reserved, prefix hits — produced by a pure scheduler with no model behind it.
//! The same input yields the same count on any machine, so the check is exact
//! equality rather than a ceiling, and an *improvement* fails just as loudly as a
//! regression: it should land in `perf-budgets.json` as part of the change that
//! earned it, not be absorbed silently.
//!
//! # The baseline arm
//!
//! Each scenario also runs with `--kv-reuse` off — the pre-0.19 behaviour, which
//! is what `llama-server` does on the same workload. Recording both arms means the
//! file states the *ratio*, which is the actual README claim; an absolute number
//! alone cannot distinguish "reuse got worse" from "the scenario got smaller".
//!
//! # Updating
//!
//! ```bash
//! FOX_UPDATE_BUDGETS=1 cargo test --lib scheduler::budgets
//! git diff perf-budgets.json    # the diff IS the performance review
//! ```

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::{InferenceRequest, SamplingParams, Scheduler, StopReason};
use crate::kv_cache::KVCacheManager;

/// KV block size used by every scenario. Fixed here rather than taken from the
/// default so a change to the default shows up as a deliberate budget edit.
const BLOCK_SIZE: usize = 16;

/// Pool size for the burst scenarios: large enough that *both* arms admit every
/// request, so `prefill_tokens` compares prompt work and nothing else.
///
/// This bound is load-bearing, and the first draft got it wrong. At 512 blocks the
/// 16-client baseline arm admitted only 14: each request reserves 35 blocks on its
/// own and 16 of those do not fit, while the reuse arm shares the prefix and fits
/// comfortably. That difference is real — it is the "sharing widens concurrency"
/// claim, and `shared_prefix_admission_pressure` below measures it deliberately —
/// but letting it leak into the prefill scenarios would confound the two effects.
const POOL_BLOCKS: usize = 1024;

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Measured {
    /// Prompt positions actually handed to the model across every request, i.e.
    /// `sum(n_positions - skip_prefix_tokens)` at admission. This is the number
    /// the shared-prefix work is about.
    prefill_tokens: usize,
    /// Peak KV blocks allocated at any point in the run. Sharing a prefix shows up
    /// here: siblings that share blocks never reserve a second copy.
    peak_blocks: usize,
    /// Requests admitted with a non-empty resident prefix.
    prefix_hits: u64,
    /// Requests that made it into the running batch. Equal to the request count in
    /// every scenario except `shared_prefix_admission_pressure`, where it is the
    /// measurement.
    admitted: usize,
}

/// Both arms of one scenario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Arms {
    /// Fox as shipped.
    reuse: Measured,
    /// `--kv-reuse` off: every prompt prefilled from token 0.
    baseline_no_reuse: Measured,
}

#[derive(Debug, Serialize, Deserialize)]
struct BudgetFile {
    schema_version: u32,
    methodology: String,
    scenarios: std::collections::BTreeMap<String, Arms>,
}

fn budget_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("perf-budgets.json")
}

const METHODOLOGY: &str = "\
Counts, not times. Each scenario drives `Scheduler::schedule_step` directly with no \
model behind it, so every figure is deterministic on any machine and the check is \
exact equality. `prefill_tokens` is the prompt work handed to the model, `peak_blocks` \
the high-water KV reservation, `prefix_hits` the admissions that found resident KV. \
The `baseline_no_reuse` arm is the same scenario with `--kv-reuse` off, which is what a \
server without fox's prefix work does; the ratio between the arms is the claim, the \
absolute numbers alone are not. These are regression tripwires for scheduler decisions, \
not latency claims about a running server — for those see scripts/ab_bench.sh and \
scripts/ab_shared_prefix.sh.";

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// A model-free stand-in for `Engine::run_loop`.
///
/// Each turn schedules, records the prompt work every newly admitted request was
/// charged, then completes the prefill of everything in the prefill list — one
/// token sampled off the final chunk, which is what moves a request to `Decoding`
/// and therefore what makes it a legal copy donor for the next turn. Requests that
/// were deferred (their donor was still prefilling) come back on the next turn,
/// exactly as in the real loop.
struct Harness {
    sched: Scheduler,
    kv: Arc<KVCacheManager>,
    /// Which arm this is. `park_finished` is a no-op with reuse off, so the
    /// assertion about parking has to know which answer is the correct one.
    kv_reuse: bool,
    seen: HashSet<u64>,
    prefill_tokens: usize,
    peak_blocks: usize,
    // Receivers are parked here: dropping one closes its channel, and a closed
    // channel is a different code path than the one under measurement.
    _rx: Vec<tokio::sync::mpsc::UnboundedReceiver<super::Token>>,
}

impl Harness {
    fn new(slots: usize, kv_reuse: bool) -> Self {
        Self::with_pool(slots, kv_reuse, POOL_BLOCKS)
    }

    fn with_pool(slots: usize, kv_reuse: bool, pool_blocks: usize) -> Self {
        let kv = Arc::new(KVCacheManager::from_kv_tokens(
            BLOCK_SIZE * pool_blocks,
            BLOCK_SIZE,
        ));
        let sched = Scheduler::new(Arc::clone(&kv), slots)
            .with_kv_reuse(kv_reuse, super::DEFAULT_SLOT_PROMPT_SIMILARITY);
        Self {
            sched,
            kv,
            kv_reuse,
            seen: HashSet::new(),
            prefill_tokens: 0,
            peak_blocks: 0,
            _rx: Vec::new(),
        }
    }

    fn submit(&mut self, id: u64, prompt: Vec<i32>, max_new_tokens: usize) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self._rx.push(rx);
        self.sched
            .submit(InferenceRequest::new(
                id,
                prompt,
                max_new_tokens,
                SamplingParams::default(),
                tx,
            ))
            .expect("budget scenarios are sized so nothing is ever refused");
    }

    /// Charge every request admitted since the last call, and update the block
    /// high-water mark.
    fn record_admissions(&mut self) {
        let running = self.sched.running_batch.lock().unwrap();
        for req in running.iter() {
            if self.seen.insert(req.id) {
                // `skip_prefix_tokens` is what the request did NOT have to submit,
                // whether it came from an idle slot, a live sibling, or the RAM cache.
                self.prefill_tokens += req.n_positions() - req.skip_prefix_tokens;
            }
        }
        self.peak_blocks = self.peak_blocks.max(self.kv.allocated_blocks());
    }

    /// Run until `expect_admitted` requests have been admitted. Bounded so a
    /// scheduler bug that stops admitting fails the test instead of hanging CI.
    fn run_until_admitted(&mut self, expect_admitted: usize) {
        let max_turns = expect_admitted * 4 + 8;
        for _ in 0..max_turns {
            let batch = self.sched.schedule_step();
            self.record_admissions();
            if self.seen.len() >= expect_admitted {
                return;
            }
            // Completing prefill is what turns a request into a copy donor.
            for id in batch.prefill {
                self.sched.update_after_token(id, 7_777, true);
            }
        }
        panic!(
            "only {} of {expect_admitted} requests were admitted within {max_turns} turns",
            self.seen.len()
        );
    }

    /// Run until admissions stop, rather than until a known count is reached. For
    /// the scenario where "how many fit" is the measurement, not a precondition.
    fn run_until_quiescent(&mut self, turns: usize) {
        let mut stalled = 0;
        for _ in 0..turns {
            let before = self.seen.len();
            let batch = self.sched.schedule_step();
            self.record_admissions();
            // Two consecutive turns with no admission and nothing left to prefill
            // means the pool is full and the rest are waiting on capacity that only
            // a completing request could release.
            stalled = if self.seen.len() == before {
                stalled + 1
            } else {
                0
            };
            if stalled >= 2 && batch.prefill.is_empty() {
                return;
            }
            for id in batch.prefill {
                self.sched.update_after_token(id, 7_777, true);
            }
        }
    }

    /// Finish a request and park its sequence, the way the engine does on the
    /// completion path — this is what leaves reusable KV behind for a later turn.
    ///
    /// With `--kv-reuse` off, `park_finished` declines by design and the sequence is
    /// cleared instead; asserting that outcome too keeps the baseline arm honest,
    /// since a baseline that quietly started parking would flatter the reuse arm's
    /// ratio without anyone noticing.
    fn finish_and_park(&mut self, id: u64, reply: &[i32]) {
        for (i, &tok) in reply.iter().enumerate() {
            self.sched.update_after_token(id, tok, i == 0);
        }
        self.sched.mark_finished(id, StopReason::Eos);
        assert_eq!(
            self.sched.park_finished(id),
            self.kv_reuse,
            "request {id}: parking must happen with kv_reuse on and never with it off"
        );
    }

    fn measured(&self) -> Measured {
        Measured {
            prefill_tokens: self.prefill_tokens,
            peak_blocks: self.peak_blocks,
            prefix_hits: self
                .sched
                .prefix_hits
                .load(std::sync::atomic::Ordering::Relaxed),
            admitted: self.seen.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// `n` clients carrying the same system prompt, all arriving before anything is
/// scheduled, each with a distinct user tail. The README's headline workload.
///
/// With reuse, the first client prefills the shared span and the other `n-1` copy
/// it out of that still-decoding sibling. Without it, all `n` prefill the same
/// tokens — which is what `llama-server` does, because its slot selection skips
/// busy slots in both the similarity pass and the LRU fallback.
fn shared_prefix_burst(n: usize, kv_reuse: bool) -> Measured {
    const SHARED: usize = 512;
    const TAIL: usize = 16;

    let mut h = Harness::new(n, kv_reuse);
    let system: Vec<i32> = (0..SHARED as i32).collect();
    for i in 0..n {
        let mut prompt = system.clone();
        // Distinct tail per client, disjoint from the shared span's token ids.
        prompt.extend((0..TAIL as i32).map(|t| 100_000 + (i as i32) * 1_000 + t));
        h.submit(i as u64 + 1, prompt, 32);
    }
    h.run_until_admitted(n);
    h.measured()
}

/// One conversation, three turns. Each turn's prompt is everything before it plus
/// the assistant's reply plus the new user message; the request finishes and parks
/// between turns, exactly as a real chat does.
///
/// Turn 2 and 3 should prefill only their new tokens. The old block-hash cache
/// could never serve this: it discarded generated tokens, so every turn re-read
/// the assistant's own reply.
fn multiturn_chat(turns: usize, kv_reuse: bool) -> Measured {
    const FIRST_TURN: usize = 256;
    const USER_MSG: usize = 24;
    const REPLY: usize = 8;

    let mut h = Harness::new(4, kv_reuse);
    let mut conversation: Vec<i32> = (0..FIRST_TURN as i32).collect();

    for turn in 0..turns {
        let id = turn as u64 + 1;
        h.submit(id, conversation.clone(), 32);
        h.run_until_admitted(turn + 1);

        let reply: Vec<i32> = (0..REPLY as i32)
            .map(|r| 900_000 + (turn as i32) * 100 + r)
            .collect();
        h.finish_and_park(id, &reply);
        // Retire the finished request so its slot is parked and matchable.
        h.sched.schedule_step();

        conversation.extend_from_slice(&reply);
        conversation.extend((0..USER_MSG as i32).map(|u| 500_000 + (turn as i32) * 100 + u));
    }
    h.measured()
}

/// The same burst against a pool that cannot hold every request at full price.
///
/// "A shared prefix is paid for once" is not only a prefill claim — sequences
/// sharing a prefix share the block budget for it instead of each reserving a
/// copy, so more of them fit at the same time. This scenario is the one that
/// measures that: `admitted` is the number, and the two arms should differ.
///
/// The pool is sized to hold every request in the reuse arm and to fall short in
/// the baseline, which is exactly the regime a small GPU is in.
fn shared_prefix_admission_pressure(n: usize, kv_reuse: bool) -> Measured {
    const SHARED: usize = 512;
    const TAIL: usize = 16;
    // 528 prompt + 32 new = 560 tokens = 35 blocks per request at full price.
    const POOL: usize = 512;

    let mut h = Harness::with_pool(n, kv_reuse, POOL);
    let system: Vec<i32> = (0..SHARED as i32).collect();
    for i in 0..n {
        let mut prompt = system.clone();
        prompt.extend((0..TAIL as i32).map(|t| 100_000 + (i as i32) * 1_000 + t));
        h.submit(i as u64 + 1, prompt, 32);
    }
    h.run_until_quiescent(n * 4 + 8);
    h.measured()
}

/// Control arm: `n` clients with nothing in common. There is no prefix worth
/// reusing, so both arms must submit every prompt token — this is the scenario
/// that fails if the harness ever stops counting real work, which would otherwise
/// let every other budget pass while measuring nothing.
fn cold_unrelated(n: usize, kv_reuse: bool) -> Measured {
    const PROMPT: usize = 128;

    let mut h = Harness::new(n, kv_reuse);
    for i in 0..n {
        // Disjoint token id ranges: no two prompts share even a first token.
        let base = (i as i32 + 1) * 10_000;
        h.submit(i as u64 + 1, (base..base + PROMPT as i32).collect(), 32);
    }
    h.run_until_admitted(n);
    h.measured()
}

fn all_scenarios() -> std::collections::BTreeMap<String, Arms> {
    let mut out = std::collections::BTreeMap::new();
    let mut add = |name: &str, f: &dyn Fn(bool) -> Measured| {
        out.insert(
            name.to_string(),
            Arms {
                reuse: f(true),
                baseline_no_reuse: f(false),
            },
        );
    };

    add("shared_prefix_burst_8_clients", &|r| {
        shared_prefix_burst(8, r)
    });
    add("shared_prefix_burst_16_clients", &|r| {
        shared_prefix_burst(16, r)
    });
    add("shared_prefix_admission_pressure_16_clients", &|r| {
        shared_prefix_admission_pressure(16, r)
    });
    add("multiturn_chat_3_turns", &|r| multiturn_chat(3, r));
    add("cold_unrelated_4_clients", &|r| cold_unrelated(4, r));
    out
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn scheduler_stays_within_its_performance_budget() {
    let measured = all_scenarios();
    let path = budget_path();

    if std::env::var_os("FOX_UPDATE_BUDGETS").is_some() {
        let file = BudgetFile {
            schema_version: 1,
            methodology: METHODOLOGY.to_string(),
            scenarios: measured,
        };
        let mut json = serde_json::to_string_pretty(&file).expect("serialise budgets");
        json.push('\n');
        std::fs::write(&path, json).expect("write perf-budgets.json");
        eprintln!("perf-budgets.json rewritten from this run — review the diff before committing");
        return;
    }

    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {}: {e}\nrun `FOX_UPDATE_BUDGETS=1 cargo test --lib scheduler::budgets` to create it",
            path.display()
        )
    });
    let recorded: BudgetFile =
        serde_json::from_str(&raw).expect("perf-budgets.json is not valid for this schema");
    assert_eq!(
        recorded.schema_version, 1,
        "perf-budgets.json schema_version is not the one this test writes"
    );

    let mut problems = Vec::new();

    for (name, got) in &measured {
        let Some(want) = recorded.scenarios.get(name) else {
            problems.push(format!(
                "  {name}: new scenario, not yet in perf-budgets.json"
            ));
            continue;
        };
        for (arm, got, want) in [
            ("reuse", got.reuse, want.reuse),
            (
                "baseline_no_reuse",
                got.baseline_no_reuse,
                want.baseline_no_reuse,
            ),
        ] {
            for (field, got, want) in [
                ("prefill_tokens", got.prefill_tokens, want.prefill_tokens),
                ("peak_blocks", got.peak_blocks, want.peak_blocks),
                (
                    "prefix_hits",
                    got.prefix_hits as usize,
                    want.prefix_hits as usize,
                ),
                ("admitted", got.admitted, want.admitted),
            ] {
                if got == want {
                    continue;
                }
                // Work avoided is the point, so for prefill_tokens and peak_blocks
                // less is the improvement; for prefix_hits and admitted, more is.
                let improved = if field == "prefix_hits" || field == "admitted" {
                    got > want
                } else {
                    got < want
                };
                let verdict = if improved { "IMPROVED" } else { "REGRESSED" };
                problems.push(format!(
                    "  {name}/{arm}/{field}: {verdict}  budget {want} -> measured {got}"
                ));
            }
        }
    }
    for name in recorded.scenarios.keys() {
        if !measured.contains_key(name) {
            problems.push(format!(
                "  {name}: in perf-budgets.json but no longer measured"
            ));
        }
    }

    assert!(
        problems.is_empty(),
        "scheduler performance budget moved:\n{}\n\n\
         A REGRESSED line means the scheduler is doing work it used to avoid — that is a\n\
         bug in the change, not in this file. An IMPROVED line should be committed:\n\
         the number belongs in the same change that earned it.\n\n\
         To record the new numbers:\n\
         \x20 FOX_UPDATE_BUDGETS=1 cargo test --lib scheduler::budgets\n",
        problems.join("\n")
    );
}
