# ZENITH Roadmap: From Credibility Fixes to a Fundable P1

This document turns the strategic assessment (an Opus-model review of commit `cf0e7c6`, grounded in the actual code and in competitor research current as of September 2026) into a concrete, sequenced execution plan. It exists so that "we should fix the docs" and "we should add filtering" turn into tracked work with acceptance criteria, not just intentions.

Read this alongside `DECISIONS.md` (why things are built the way they are) and `bench/BENCHMARK.md` (the numbers). This document is about what happens next, and in what order, and why that order.

---

## Status legend

- **DONE** — implemented and verified in this repo, not just planned.
- **NEXT** — the next item to actually build.
- **PLANNED** — sequenced, not started.
- **[FMT]** — breaks the on-disk index format; must ship as one bundled version bump, not piecemeal.
- **[EMB!]** — risks the embeddable, zero-config promise if implemented carelessly (e.g. by pulling ops complexity into `pkg/zenith`).

---

## P0: credibility fixes (~2 weeks, high leverage, low effort)

These are the things that make a technical reader trust — or distrust — everything else in the pitch. All of P0 should land before any external conversation.

| # | Item | Status |
|---|------|--------|
| P0-1 | Fix the 10-result cap; add limit/offset to gRPC and HTTP; add tests above 10 | **DONE** |
| P0-2 | Make docs match reality: remove Nerve references, fix README's gRPC list, implement `WithMemoryLimit` or delete the claim | **DONE** |
| P0-3 | Add CI: `go test -race`, CGo on/off matrix, Linux/macOS/Windows, badge | **DONE** (this session) |
| P0-4 | Bound crash recovery: checkpoint by default past a WAL size threshold; write the computed vector into the WAL record so replay never re-embeds | **DONE** (`checkpointIfWALTooLarge`, 64MB default; WAL Put records carry text + vector + attrs) |
| P0-5 | Record the embedder's model ID and vector size in the file header; refuse to open on mismatch | **DONE** (snapshot v5 header; `ErrEmbedderMismatch`) |

### P0-1 and P0-2 — what actually changed

- `internal/config.Config.MaxResults` was dead (declared, defaulted to 10, never read). It's now the real internal RRF candidate cap (default 1000), wired into all three ranker-construction sites (`pkg/zenith/zenith.go`, `cmd/server/main.go`, `cmd/zenith/engine.go`). `Limit(n)` in the library, `SearchRequest.limit/offset` in gRPC (new proto fields, `internal/server.paginate`), and the CLI's `--max` now all have real candidates to slice from instead of hitting a hardcoded cap of 10 before any of that logic runs.
- `WithMemoryLimit(limitBytes int64)` is implemented in `pkg/zenith/options.go`, using an ~11 KB/doc estimate derived from the measured heap delta in `bench/BENCHMARK.md` (1,127 MB / 100,000 docs). It's approximate accounting by design — documented as such — and exempts overwrites of existing IDs from counting as growth. `Add`/`AddBatch` return `ErrIndexFull` (already declared, now actually reachable) once the projected total would exceed the limit.
- `CLAUDE.md` and `README.md` now describe `internal/localembedder` (the real in-process ONNX embedder) instead of the deleted `nerve/` Python sidecar, and the README's gRPC method list matches `document.proto` exactly (`IndexDocuments`, `Search`, `IndexPDF`, `GetDocument`, `DeleteDocument`).
- Regression tests added: `TestSearch_LimitAboveTen` (`pkg/zenith/zenith_test.go`), three `WithMemoryLimit` tests (`pkg/zenith/options_test.go`), six `paginate` tests (`internal/server/pagination_test.go`).

### P0-3 — CI

Added `.github/workflows/ci.yml`: a matrix over `{ubuntu-latest, macos-latest, windows-latest} × {CGO_ENABLED=0, CGO_ENABLED=1}`, running `go build ./...`, `go vet ./...`, and `go test -race ./...` (race detector requires CGo, so the CGo-off leg skips `-race` and runs a plain `go test`). README badge added. This is the direct mitigation for risk #9 (bus-factor / single-maintainer discount) and risk #10 (credibility gaps) below — an accelerator, or any serious adopter, checks for this before reading anything else.

### P0-4 and P0-5 — what was built, and a deliberate deviation from this document's earlier plan

An earlier revision of this section said to hold both back and ship them together with the segment migration, to avoid a second format bump (risk #5). They were built now instead, because both fix correctness problems that hurt today (silently mixing vector spaces after an embedder change; crash recovery that re-runs ONNX inference for every WAL record) and the codebase's established precedent for a version bump is a hard, explicit `ErrIncompatibleVersion` rather than silent corruption. **The cost is real:** the snapshot is now v5 (unreleased, so no shipped v4 users are stranded by it), and the segment migration (P1-4) will be a further bump. The mitigation for that is P1-1: `zenith migrate` and a published compatibility policy must land *with* the segment format, not after.

What changed:

- **P0-5** — the snapshot header now stores the embedder name and vector dimension before the gob body; `Load` fails fast with `ErrEmbedderMismatch` (surfaced as `zenith.ErrEmbedderMismatch`) without decoding the body. Embedders identify themselves through an optional `embedding.Named` interface, so custom `WithEmbedder` types are not a breaking change; an embedder that doesn't implement it is recorded as `unknown` and never mismatch-checked.
- **P0-4** — `Add`/`AddBatch` embed once and reuse the vector for both the WAL record and indexing; replay calls `AddWithVectorAttrs` and never touches the embedder for the document vector. A WAL past 64MB triggers a synchronous checkpoint even when `WithCheckpointInterval` was never set. Per-token word vectors are still warmed on replay; they are deduplicated by vocabulary, so that cost is bounded by unique terms, not documents. Old raw-text WAL values still decode (legacy fallback).

---

## P1: table stakes (1–3 months)

Ordered by dependency, not by the report's original numbering — filtering and the format migration gate several other items.

1. **`zenith migrate` tool + published compatibility policy.** *(P0-4/P0-5 format work is done — see above; this item is now what remains: the migration tool and policy, which must ship with the segment format.)* **NEXT-with-P1-4 [FMT]**
2. **Metadata and filtering** — **DONE at the library level.** `AddWithAttrs` / `AddBatchWithAttrs`, `WithFilter(Eq | In | Range | Exists | And | Or | Not)`, applied to the lexical *and* vector candidate sets before rank fusion (so `Limit(1)` returns the best *matching* document, not a filtered view of the global top), persisted in the snapshot and the WAL. **Not yet done:** roaring-bitmap postings for high-selectivity filters (today it is a per-candidate predicate, O(candidates)), gRPC/HTTP/CLI surface for attributes, and `Explain` + `WithFilter` (currently rejected with `ErrInvalidOption`).
3. **Lexical candidate pass rebuild** — **PARTIAL.** Done: Levenshtein automaton over the existing `vellum` FST replaces BK-tree fuzzy search in the query path (identical match set and distances, verified against the BK-tree on random multi-byte vocab; falls back to the BK-tree above edit distance 3), and RRF fusion no longer sorts every candidate through map lookups (materialised sort keys + bounded top-N heap; verified against a full-sort reference). Measured on a synthetic 100k-doc Zipf corpus, lexical-only: **19.8ms → 11.9ms mean per query** (fuzzy step alone ~3x faster at 20k docs). **Not done:** capped prefix-fragment expansion, WAND/MaxScore pruning, dropping unused TF-IDF maintenance (`engine.go`), and — importantly — the numbers above are a *synthetic* corpus, not MS MARCO; the 177ms/346ms baseline in the report is on a different corpus and hardware, so do not compare them directly until P1-7 reruns everything on the same corpus.
4. **Memory-mapped immutable segments + tombstones**, replacing whole-index gob snapshots. This is where the existing SSTable/bloom-filter/compaction code in `internal/storage` finally becomes load-bearing instead of parallel infrastructure. **[FMT]**
5. **ANN index (HNSW, pure Go, filter-aware)** — **DONE, not persisted.** `internal/ann` is a pure-Go HNSW over the existing float16 vectors (graph only; no second copy of the vectors). `vectorPass` uses it above `WithANNThreshold` (default 20,000 docs) and scans exactly below that. Filters run inside graph traversal; below 10% selectivity it falls back to an exact scan over the allowed documents. Measured: recall@10 0.967 vs brute force (6k clustered 64-dim vectors, ef=64); top-10 overlap 1.000 with exact hybrid search in the engine test; vector pass at 30k docs (64-dim) **9.2ms exact → 0.62ms ANN**. **Known costs:** the graph is rebuilt from stored vectors on `Open` (not saved to disk — persisting it belongs with the P1-4 segment format), so very large indexes pay build time at startup; ANN hands `annK=150` candidates to rank fusion rather than every positive-scoring document; recall on real 384-dim embeddings has not been measured (only synthetic vectors) — do that in P1-7 before trusting the default.
6. **A better default embedding model** (bge-small / gte-small for English, multilingual-e5-small for the Hinglish/Devanagari story), declared by ID, made safe by the header work in item 1.
7. **Honest, reproducible benchmarks** — **PARTIAL (harness only; no new numbers published).** Done: the `bench` module was not building at all (missing `golang.org/x/text` in its `go.sum`; fixed), and `go run ./cmd/benchmark --queryset=typo|codemixed --seed=N` now derives seeded, deterministic query sets from the MS MARCO queries (`bench/internal/queryset`, unit-tested), graded against the same qrels. `typo` applies one realistic edit (delete/repeat/nearby-key/transposition, never the first letter) per eligible word. `codemixed` is a **synthetic noise proxy** (romanised-Hindi function words interleaved with English content words) — it cannot test romanised spelling variation or Devanagari, so replace it with human-authored code-mixed data before making any Hinglish quality claim. **Not done, and needs your machine/network:** the MS MARCO download (no `.cache` here), Bleve/SQLite runs of the new sets, and competitors that need extra infrastructure (chromem-go, sqlite-vec, LanceDB, Meilisearch, Qdrant), BEIR subsets, and per-signal ablations.

## P2: server-grade (3–6 months, only once the server has real users)

8. Unify the gRPC server and HTTP sidecar into one server with persistent collections; add an auth interceptor (API keys, optional mTLS), request limits, Prometheus metrics, OpenTelemetry traces, online backup (copy immutable segments + `zenith dump/restore`). **[EMB!] — keep entirely in `cmd/`, never in `pkg/zenith`.**
9. Thin Python/TypeScript SDKs generated from an OpenAPI spec of the HTTP API. Defer native Python bindings via `c-shared` — a Go runtime inside CPython is painful and low-value until there's demand.

## P3: only with demand pull

10. Read replicas by shipping WAL + segments to S3 (Litestream-style). No Raft, no sharding.
11. Hosted serverless namespaces on object storage, same segment format serving both the embedded library and the cloud product (the Turbopuffer/LanceDB pattern).

## Explicitly not doing

- Distributed clustering before there are users.
- Billion-vector ambitions.
- A generic vector-database feature race against Qdrant or Milvus.

---

## Risk-by-risk mitigation plan

Each row of the original risk table, with the concrete roadmap item(s) that address it and what "done" looks like.

| # | Risk | Mitigation plan | Roadmap tie-in | Best-in-class target |
|---|------|------------------|-----------------|----------------------|
| 1 | Crowded, well-funded hybrid-search market — commoditized | Don't compete as "a vector database." Own **embedded, noisy-text, explainable matching** as the category. Publish the typo-perturbed / code-mixed benchmark where BK-tree, phonetic, and Unicode work actually show a gap versus competitors. | P1-7 (benchmarks) | Recognized leader on published noisy-query retrieval benchmarks for embeddable engines, one-line setup. |
| 2 | Go-only market ceiling | Go library stays the core for Go shops. Generated Python/TypeScript clients over the unified HTTP server reach everyone else. Sidecar pattern already has one live user (Kshetra IQ) to point to. Native bindings only with actual demand. | P2-9 | Any language gets the same engine via one container image or import, identical results and explain output. |
| 3 | Latency 40–80× slower than Bleve; fuzzy/phonetic layers are the cost | Levenshtein automaton over FST, capped prefix expansion, WAND pruning, HNSW, drop unused TF-IDF maintenance. Add a latency regression gate to CI once P1-3 lands. | P1-3, P1-5 | Sub-20ms p50 hybrid at 1M docs on a laptop, typo tolerance still on. |
| 4 | Memory/scale ceiling — ~11KB/doc, all in RAM; 1M docs ≈ 11GB | Memory-mapped immutable segments with compressed postings, float16/int8 vectors on disk, a real `WithMemoryLimit` (done — currently RAM-estimate based, becomes exact once segments land), published RAM-per-doc figure. | P1-1, P1-4 | RAM bounded by working set; index is a disk file like SQLite. |
| 5 | On-disk format churn — gob v4, no migration path | Batch every format change (vector-in-WAL, model-ID header, segment layout, ANN) into one v5 release with a `zenith migrate` tool, then publish a written compatibility policy plus golden-file tests of old formats in CI. | P1-1 (blocks P1-4, P1-5) | SQLite-style promise: files written today open in every future 1.x release. |
| 6 | Durability/recovery — checkpointing off by default, replay re-embeds everything, checkpoints lock the whole index | Store vectors in WAL records, checkpoint on WAL size by default, move to incremental segment flushes, add fault-injection (`kill -9`) tests in CI. | P0-4 → P1-1, P1-4 | Crash-safe with bounded recovery time, verified by a public torture-test suite. |
| 7 | Embedding-model dependency — MiniLM is dated, CGo/ONNX adds build friction, model not recorded in the index | Model ID + dimension in the file header (P0-5), curated model registry with a better default (P1-6), prebuilt release binaries per OS, keep the no-CGo lexical-only mode fully first-class. | P0-5, P1-1, P1-6 | Model choice is one option; results are reproducible per model ID; zero-toolchain binaries exist. |
| 8 | Missing basics: filtering, auth, multi-tenancy | Schemaless attribute filters in the library (P1-2). API keys, per-collection tenancy, and metrics stay server-only (P2-8) so the embeddable library stays simple. | P1-2, P2-8 | Filtered hybrid search as easy as `Search(q, Where("tenant","=",x))`. |
| 9 | Single-maintainer bus factor — ~193 commits, one person, no CI | CI is now live (P0-3, done this session). Next: `CONTRIBUTING.md` with "good first issues" pulled from this roadmap, `DECISIONS.md` as onboarding material, technical cofounder/first hire as part of any accelerator ask. | P0-3 (done) | Active contributor base, reproducible builds; project outlives any one person. |
| 10 | Credibility gaps in docs — a documented-but-unimplemented feature, stale architecture description, the 10-result bug, the 32-query headline number | P0-1/P0-2 (done this session) close the concrete gaps. Ongoing: always lead with the 6,980-query MS MARCO numbers (not the 32-query subset), link the benchmark scripts so anyone can re-run them. | P0-1, P0-2 (done) | Known for honest benchmarks; the existing benchmarking rigor becomes a brand asset rather than a liability. |
| 11 | Monetizing a library is hard | Open-core: MIT library + server, paid hosted tier reusing the segment format once it exists (P3-11), paid support for code-mixed/regional-language deployments with Kshetra IQ as a reference case. | P1-1 (segment format is the prerequisite), P3-11 | The DuckDB → MotherDuck pattern: free embedded engine is the funnel top for a hosted product. |

### Monetization guardrails (risk #11) — rules applied while building

The open-core plan only works if the free engine stays trustworthy and the paid layer has somewhere to attach. These are constraints on every change, and where the code already leaves room:

1. **No telemetry, ever, by default.** A local-first search engine that phones home forfeits the funnel top. `zenith doctor --json` exists so a user *chooses* to attach diagnostics to a support request (versions, OS, paths, check outcomes — never index contents or queries); nothing is sent automatically. This is the paid-support on-ramp without a data-collection liability.
2. **No license-gated core features.** Filtering, crash recovery, install tooling, and the embedder-mismatch guard stay in the MIT library. What is left to charge for is operations, not correctness: hosted namespaces, backups/replicas, SLAs, and regional-language deployment support.
3. **Seams a hosted tier needs, already in place:** metadata attributes + pre-fusion filtering are the natural per-tenant isolation key (`Eq("tenant", id)`); the embedder identity header lets a hosted service swap or version models without silently corrupting customer indexes; `WithMemoryLimit` gives a per-tenant resource cap; versioned formats fail loudly instead of corrupting.
4. **Exit stays cheap.** Data lives in one directory; `zenith uninstall` keeps it by default and only deletes on explicit `--purge`. Lock-in through data hostage would poison an open-core funnel.
5. **What is still missing for monetization, deliberately not built yet (P2/P3 are gated on real users):** auth/quotas/metrics on the server, an export/backup command that a hosted tier could import, and tenant-aware limits. **Flag:** the repo had no `LICENSE` file although the README says MIT — without one, the open-core story is legally ambiguous to any adopter. A `LICENSE` file has been added to match the README; the copyright holder line is the maintainer's to confirm.

---

## Sequencing summary

**Done this session (P0-1, P0-2, P0-3):** the result-limit bug across library/gRPC/CLI, `WithMemoryLimit`, doc corrections (`CLAUDE.md`, `README.md`), and CI.

**Done since (P0-4, P0-5, P1-2, P1-3 partial, P1-5, P1-7 harness, plus CLI install/uninstall/doctor):** see the status notes above. **Historical plan for the format work, kept for context:** design the v5 format once — vector-in-WAL, model-ID header, segment layout — rather than bumping the format twice. This single piece of work unblocks P1-2 through P1-5 and directly answers risks #4, #5, #6, and #7.

**Then (P1-2 through P1-7):** filtering first (biggest adoption blocker), then the latency rebuild and segment migration in parallel since they touch different subsystems, then ANN, then the honest published benchmark that the whole pitch leans on.

P2 and P3 are explicitly gated on the server having real users — building auth, multi-tenancy, or hosted infrastructure before that point would be effort spent on a shape of demand that doesn't exist yet.
