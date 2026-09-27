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
| P0-4 | Bound crash recovery: checkpoint by default past a WAL size threshold; write the computed vector into the WAL record so replay never re-embeds | PLANNED — **[FMT: WAL record]** |
| P0-5 | Record the embedder's model ID and vector size in the file header; refuse to open on mismatch | PLANNED — **[FMT]**, bundle with P0-4 |

### P0-1 and P0-2 — what actually changed

- `internal/config.Config.MaxResults` was dead (declared, defaulted to 10, never read). It's now the real internal RRF candidate cap (default 1000), wired into all three ranker-construction sites (`pkg/zenith/zenith.go`, `cmd/server/main.go`, `cmd/zenith/engine.go`). `Limit(n)` in the library, `SearchRequest.limit/offset` in gRPC (new proto fields, `internal/server.paginate`), and the CLI's `--max` now all have real candidates to slice from instead of hitting a hardcoded cap of 10 before any of that logic runs.
- `WithMemoryLimit(limitBytes int64)` is implemented in `pkg/zenith/options.go`, using an ~11 KB/doc estimate derived from the measured heap delta in `bench/BENCHMARK.md` (1,127 MB / 100,000 docs). It's approximate accounting by design — documented as such — and exempts overwrites of existing IDs from counting as growth. `Add`/`AddBatch` return `ErrIndexFull` (already declared, now actually reachable) once the projected total would exceed the limit.
- `CLAUDE.md` and `README.md` now describe `internal/localembedder` (the real in-process ONNX embedder) instead of the deleted `nerve/` Python sidecar, and the README's gRPC method list matches `document.proto` exactly (`IndexDocuments`, `Search`, `IndexPDF`, `GetDocument`, `DeleteDocument`).
- Regression tests added: `TestSearch_LimitAboveTen` (`pkg/zenith/zenith_test.go`), three `WithMemoryLimit` tests (`pkg/zenith/options_test.go`), six `paginate` tests (`internal/server/pagination_test.go`).

### P0-3 — CI

Added `.github/workflows/ci.yml`: a matrix over `{ubuntu-latest, macos-latest, windows-latest} × {CGO_ENABLED=0, CGO_ENABLED=1}`, running `go build ./...`, `go vet ./...`, and `go test -race ./...` (race detector requires CGo, so the CGo-off leg skips `-race` and runs a plain `go test`). README badge added. This is the direct mitigation for risk #9 (bus-factor / single-maintainer discount) and risk #10 (credibility gaps) below — an accelerator, or any serious adopter, checks for this before reading anything else.

### P0-4 and P0-5 — why they're not done yet

Both are format-breaking. The report's own guidance (item 5, roadmap section) is explicit: *"Bundle this with item 8 rather than bumping the version again."* Rushing P0-4/P0-5 out independently would mean a second format bump when P1-8 (memory-mapped segments) lands, which is exactly the "format churn" failure mode that risk #5 warns about. The correct sequencing is:

1. Design the v5 on-disk format once, covering: vector-in-WAL-record (P0-4), model-ID-and-dimension header (P0-5), and the segment layout needed for P1-8.
2. Ship `zenith migrate` alongside it.
3. Publish the compatibility policy immediately after, while it's still true.

This is tracked as the first item under P1 sequencing below, not skipped.

---

## P1: table stakes (1–3 months)

Ordered by dependency, not by the report's original numbering — filtering and the format migration gate several other items.

1. **v5 format design + `zenith migrate` tool** (folds in P0-4, P0-5). Must land before P1-8 and before P1-9, since both want to touch the on-disk layout. **[FMT: the big one]**
2. **Metadata and filtering** (`Add(ctx, id, text, zenith.Attrs{...})`, eq/in/range/tag filters, roaring-bitmap candidate sets applied pre-fusion). Report explicitly calls this the biggest adoption blocker — you cannot build multi-user apps without it. Schemaless, optional. **[FMT] [EMB!]**
3. **Lexical candidate pass rebuild**: Levenshtein automaton over the existing `vellum` FST replacing BK-tree fuzzy search, capped prefix-fragment expansion, WAND/MaxScore pruning, and dropping the TF-IDF maintenance that's persisted but unused in ranking (`engine.go:634-635,1068`). Target: p50 under 20ms at 100k docs (current: 177ms BM25-only, 346ms hybrid).
4. **Memory-mapped immutable segments + tombstones**, replacing whole-index gob snapshots. This is where the existing SSTable/bloom-filter/compaction code in `internal/storage` finally becomes load-bearing instead of parallel infrastructure. **[FMT]**
5. **ANN index (HNSW, pure Go, filter-aware)** in `vectorPass`, brute force below ~20k docs. **[FMT]**
6. **A better default embedding model** (bge-small / gte-small for English, multilingual-e5-small for the Hinglish/Devanagari story), declared by ID, made safe by the header work in item 1.
7. **Honest, reproducible benchmarks** against Bleve, chromem-go, sqlite-vec+FTS5, LanceDB, Meilisearch hybrid, Qdrant hybrid — on BEIR subsets, a typo-perturbed set, and a code-mixed (Hinglish) set. Publish per-signal ablations (fuzzy/phonetic/n-gram/neural); cut any signal that adds nothing.

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

---

## Sequencing summary

**Done this session (P0-1, P0-2, P0-3):** the result-limit bug across library/gRPC/CLI, `WithMemoryLimit`, doc corrections (`CLAUDE.md`, `README.md`), and CI.

**Next (P0-4, P0-5 → P1-1):** design the v5 format once — vector-in-WAL, model-ID header, segment layout — rather than bumping the format twice. This single piece of work unblocks P1-2 through P1-5 and directly answers risks #4, #5, #6, and #7.

**Then (P1-2 through P1-7):** filtering first (biggest adoption blocker), then the latency rebuild and segment migration in parallel since they touch different subsystems, then ANN, then the honest published benchmark that the whole pitch leans on.

P2 and P3 are explicitly gated on the server having real users — building auth, multi-tenancy, or hosted infrastructure before that point would be effort spent on a shape of demand that doesn't exist yet.
