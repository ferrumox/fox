# Compatibility Policy

> Applies from 0.20.5. Last updated 2026-08-06.

Fox is a drop-in replacement for Ollama and an OpenAI-compatible server. That means
fox's interface is not only used by people reading its docs — it is consumed by Open
WebUI, Continue.dev, the `ollama` CLI, the `openai` SDKs and editor plugins, none of
which fox controls and all of which break silently and remotely when a response shape
changes.

This document states which parts of fox are a promise, which are merely observable, what
counts as breaking each of them, and what has to be true before 1.0.

---

## Current status

Fox is **0.x pre-release**. Under [SemVer 2.0.0](https://semver.org/) the public API is
not stable during 0.x, and fox does not claim otherwise. What this document adds is
predictability *within* that: the commitments below are what early adopters can plan
against, and they are enforced by tooling rather than by intention where that is
possible.

There is one binary, `fox`, and one version number. Nothing is published to crates.io —
`Cargo.toml` carries the metadata but the crate is not released, so the Rust API is not a
surface anyone can depend on. Artifacts are the GitHub release tarballs
(`fox-X.Y.Z-x86_64-unknown-linux-gnu[-vulkan].tar.gz`) and the `ferrumox/fox` Docker
image, which always ship together at the same version.

---

## Tiers

The distinction that matters is not "public vs private" but **who breaks when it
changes**. Something a third-party tool parses is a stronger commitment than something a
human reads.

### Tier 1 — Contract

Changing these breaks software that is not in this repository. They move only on a minor
bump during 0.x, with a CHANGELOG entry that says so, and after 1.0 only on a major.

| Surface | Detail |
|---|---|
| OpenAI routes | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/models/:model_id` — request fields accepted and response shape |
| Ollama routes | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/show`, `/api/delete`, `/api/pull`, `/api/copy`, `/api/create`, `/api/version` — including NDJSON vs SSE framing per route |
| Editor routes | `/infill`, `/rerank` and `/v1/rerank`, `/tokenize`, `/detokenize`, `/apply-template` |
| Fox extensions | `/api/models/:name/load` and `/api/models/:name/unload`. Fox's own routes, not part of either upstream API, but Tier 1 all the same: they exist so an operator can script residency, which is only useful if the script keeps working |
| Liveness | `/health` returning 200 when the server is up. The *body* is Tier 2 — orchestrators read the status code, dashboards read the body |
| CLI subcommands | `serve`, `run`, `pull`, `list`, `rm`, `show`, `ps`, `models`, `search`, `alias` — names, positional arguments, and exit status |
| Documented flags | Every long flag in [`docs/cli/`](docs/cli/) and [`docs/configuration.md`](docs/configuration.md), and the `FOX_*` variable bound to it |
| Config files | `~/.config/ferrumox/config.toml` and `~/.config/ferrumox/aliases.toml` — path, format, key names |
| Model store | `~/.cache/ferrumox/models` layout, and the GGUF filenames `fox pull` writes there |
| Release bundle | The tarball contains `fox` plus its ggml backend `.so` files, and `fox` is linked `RPATH=$ORIGIN`. **The `.so` files must stay beside the binary**; a layout change here breaks every existing install script, so it is Tier 1 even though it is not code |
| Auth | `FOX_API_KEY` / `--api-key` semantics: `Authorization: Bearer <key>` on all routes |

### Tier 2 — Observable, best-effort

Real interfaces that people build on, but where a change costs a dashboard edit rather
than a broken client. Changed on a minor bump with a CHANGELOG entry; no deprecation
window is promised.

- **Prometheus metrics** — the `fox_*` names on `/metrics` and their label sets. Every
  metric carries a `model` label, capped at 32 distinct values per process with the
  remainder collapsed into `model="<other>"`; the cap is a Tier 2 number and may move.
- **Introspection routes** — `/slots`, `/props`, `/lora-adapters`, and the body of
  `/health`. These report internal state by design, so they move when the internals do.
- **Diagnostic subcommands** — `probe`, `bench`, `bench-kv`, `bench-prefill`,
  `bench-spec`. Developer tools whose output is meant to change as what is worth
  measuring changes. `fox bench --output json` is the most stable of them, and still
  Tier 2.
- **Log output** — human format and the `--json-logs` field set.
- **CLI human output** — the tables `fox list`, `fox ps` and `fox show` print. Parse them
  at your own risk; `/api/tags` and `/api/ps` are the Tier 1 answer to the same question.

### Tier 3 — No commitment

Changed at any time, in any release, with no entry required.

- **Undocumented environment variables.** These are escape hatches and internal tuning,
  not configuration: `FOX_CONFIG`, `FOX_KV_UNIFIED`, `FOX_LLAMA_LOG`, `FOX_N_THREADS`.
  Anything else `FOX_*` that does not appear in `docs/configuration.md` is in this tier
  by default, including the build-time (`FOX_SKIP_LLAMA`) and test-only
  (`FOX_GOLDEN_MODEL`, `FOX_UPDATE_BUDGETS`) variables.
- **Everything in `src/`.** The crate is not published; module layout, types and the
  scheduler's behaviour are implementation.
- **`perf-budgets.json`.** A regression tripwire for this repo's CI, not a published
  performance claim. Its numbers are expected to move.
- **The vendored `llama.cpp` commit.** Pinned as a submodule and bumped whenever it is
  useful to bump.

---

## What counts as a breaking change

Per surface, concretely.

**HTTP** — removing a route; removing or renaming a response field; changing a field's
type or its meaning; narrowing what a request field accepts; changing a status code for
an unchanged condition; changing stream framing (SSE ↔ NDJSON, or the `data:` / `[DONE]`
convention); making a previously optional request field required.

**CLI** — removing or renaming a subcommand or a documented flag; changing a flag from
taking a value to not, or the reverse; changing a documented default; changing a
subcommand's exit status for an unchanged outcome.

**Configuration** — removing or renaming a config key or a documented `FOX_*` variable;
changing the precedence order (flag > env > config file > default); moving a config or
cache path.

**Packaging** — changing the tarball layout, the binary names inside it, or the runtime
search path for the backend `.so` files.

### What does not count

Stated explicitly, because treating these as breaking would freeze the project for no
one's benefit:

- **Adding** a route, subcommand, flag, config key, response field, or metric.
- Accepting a request field that was previously rejected.
- Any change in **speed, memory use, batching, or scheduling decisions** — provided
  output is unchanged. Prompt reuse, block sharing and admission order are
  implementation. `perf-budgets.json` tracks them; it does not promise them.
- Any change in **generated text** attributable to a vendored `llama.cpp` bump, a
  different quantization, or non-deterministic sampling. Fox does not promise token-level
  reproducibility across versions.
- Changes to log wording, progress output, or colour.
- Fixing a route whose behaviour **diverged from the API it claims to implement**. If
  fox's `/v1/chat/completions` disagrees with OpenAI's documented behaviour, the
  divergence is the bug and correcting it is a fix, not a break — see below.

---

## Ollama and OpenAI compatibility

The compatibility claim is specifically this: **an unmodified client pointed at fox
works**. It is not a claim that fox implements every field of either API, nor that it
mirrors an upstream version.

- **Coverage is a subset, and it grows.** A field fox does not implement is a gap to
  close, not a promise broken. Gaps live in
  [`docs/design/vllm-gap-analysis.md`](docs/design/vllm-gap-analysis.md) and
  [`docs/design/llama-server-gap-analysis.md`](docs/design/llama-server-gap-analysis.md).
- **Upstream is the specification, fox is the implementation.** When the two disagree,
  fox is wrong. Correcting such a divergence is a patch-level fix even though it changes
  observable behaviour, because clients were already written against upstream. The
  CHANGELOG entry must say what changed and why.
- **Upstream breaking changes are not inherited automatically.** If OpenAI or Ollama
  changes an existing response shape, fox weighs following it against breaking clients
  already working, and records the decision in the CHANGELOG.
- **`/api/version` reports fox's own version**, not an Ollama version — `0.20.5`, not
  whatever Ollama release the surface tracks. Clients that gate features on a version
  comparison are therefore comparing against the wrong number. This is deliberate
  (reporting a fake Ollama version is worse), it has not caused a problem so far because
  both projects are on 0.x, and it is on the 1.0 checklist below.

---

## Versioning during 0.x

| Bump | Means |
|---|---|
| **Patch** (0.20.4 → 0.20.5) | Bug fixes, performance work, documentation, internals. No Tier 1 change. A correction that brings a route back in line with the API it implements is allowed here. |
| **Minor** (0.20.x → 0.21.0) | New features, and any Tier 1 or Tier 2 change. This is where breaking changes live during 0.x. |
| **Major** | Reserved for 1.0. |

Every release has a CHANGELOG entry — enforced, not requested: `scripts/release.sh`
refuses to prepare a release whose version has no entry, and the Release workflow
re-checks it before building.

### Version reporting is itself a guarantee

Versions 0.14 through 0.18 were published with `Cargo.toml` still reading `0.11.0`. The
binaries reported the wrong version for six releases and nobody noticed, which made
`fox --version` — and therefore `/api/version` — untrustworthy for that range.

That class of failure is now closed by tooling: `scripts/release.sh` requires a clean
tree and a matching CHANGELOG entry, `scripts/publish.sh` refuses to tag unless
`Cargo.toml` equals the tag, and the Release workflow re-verifies the match, unpacks the
tarball it is about to publish and runs it to confirm it reports the right version. The
policy in this document is only worth as much as the version number it attaches to.

---

## Deprecation

During 0.x, when a Tier 1 surface has to change and a compatible path exists:

1. The replacement ships first, in the same release or earlier.
2. The old name keeps working for **at least one minor release**, and warns once on
   stderr when used — never per request.
3. The CHANGELOG entry names both the old and the new spelling, in the release that
   introduces the deprecation and again in the release that removes it.
4. Removal happens in a minor bump, not a patch.

Where no compatible path exists — a response shape that was simply wrong, a flag whose
meaning was inverted — the change lands in a minor release with a CHANGELOG entry saying
plainly what breaks and what to do about it.

This has not always been the practice. The 0.6.0 rename from `ferrum-engine` to
`ferrumox` moved every environment variable from `FERRUM_*` to `FOX_*` with no aliases
and no window. That was defensible at 0.6 with no users; it is what this section exists
to prevent repeating.

---

## Before 1.0

1.0 is not a date. These are the conditions:

- [ ] **Prebuilt binaries for macOS and Windows.** Today the release workflow builds
      Linux x86_64 only, and the other platforms are told to build from source or use
      WSL2. A 1.0 that most desktop users cannot install is not a 1.0.
- [x] **Metric names settled.** Done in 0.21: the prefix is `fox_`, matching the binary
      and every user-facing string. It was `ferrumox_` through 0.20, and renaming broke
      existing dashboards — which is precisely why it had to happen before 1.0 rather
      than after, when it could not have happened at all.
- [x] **A per-model dimension on metrics, with a cardinality bound.** Done in 0.21: every
      metric carries `model`, capped at 32 distinct values with the remainder collapsed
      into `model="<other>"`. The cap is not optional — model names come from arbitrary
      HuggingFace repos, so the label set is influenced from outside the server.
- [ ] **A decision on `/api/version`.** Either keep reporting fox's version and document
      it as the answer, or report a compatibility version clients can gate on. Not both,
      and not by accident.
- [ ] **Tier 1 covered by tests that fail on drift.** Route shapes and CLI surface
      asserted, the way `scripts/check_docs_flags.py` already asserts that documented
      flag names exist.
- [ ] **The gap analyses closed or declared.** Each remaining gap either implemented or
      written down as a deliberate non-goal.
- [ ] **One release cycle with no Tier 1 change.** The surface has to demonstrate it has
      stopped moving before it is declared stable.

After 1.0, Tier 1 changes require a major bump, and the deprecation window becomes two
minor releases rather than one.

---

## Reporting a compatibility problem

A client that works against Ollama or the OpenAI API and fails against fox is a bug, even
when fox's behaviour is defensible in isolation. Open an issue with the client name, the
request, and what each server returned. That comparison is the whole report — see
[CONTRIBUTING.md](CONTRIBUTING.md).
