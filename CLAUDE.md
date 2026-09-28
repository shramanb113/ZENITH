# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. You don't have explicit permission to commit you will not commit and add to github under any circumstance.

## Commands

```bash
# Build everything
go build ./...

# Run the gRPC server (port :8080)
go run ./cmd/server

# Run all tests
go test ./...

# Run tests for a single package
go test ./internal/ranking/...
```

Vector search uses an in-process ONNX embedder (`internal/localembedder`) — the model ships inside the Go binary via `go:embed`, so no separate sidecar process is needed. It requires `CGO_ENABLED=1` and a C compiler (MinGW-w64 on Windows, gcc on Linux/macOS); without CGo the engine falls back to a deterministic hash-based embedder (`embedding.NewDeterministicEmbedder`), which degrades semantic search to non-semantic but keeps lexical (BM25/n-gram/fuzzy) search fully working.

The gRPC server loads the index from `zenith.db` on startup and saves it on graceful shutdown (SIGINT/SIGTERM).

## Architecture

ZENITH is split into two independent engine layers that are wired together in `cmd/server/main.go`:

### 1. Storage Engine (`internal/storage/storage_engine.go`)

The LSM-tree pipeline:

- **WAL** (`internal/storage/wal/`) — append-only crash log; replayed into MemTable on Open
- **MemTable** (`internal/storage/memtable/`) — concurrent skip-list; frozen and flushed when it exceeds `MemTableMaxSize` (64MB default)
- **SSTable** (`internal/storage/sstable/`) — immutable sorted files written by the GroupCommitter; Bloom filter + sparse index per file
- **FST** (`internal/analysis/fst.go`) — rebuilt from the global term vocabulary after every SSTable flush; used by the Analyzer for prefix-search query resolution

The storage engine owns the FST and vocabulary; the index engine calls `AddTerms()` after each document.

### 2. Index Engine (`internal/index/engine.go`)

The search orchestrator — owns all sub-indexes and the scoring pipeline:

- **Layers** — the index is a stack of immutable, memory-mapped **segments** (`internal/segment`, wiring in `internal/index/layers.go`) plus a small mutable in-memory **delta** (the maps in `InvertedIndex`, `VectorStore`, `PhoneticIndex` and the BM25 stats). A document is live in exactly one place; replacing or removing a segment document marks its row dead in a per-segment bitset and records the deletion for the next flush. Search reads the delta and every segment through `eachFragDoc` / `eachPhonDoc` / `eachVector`, so results are identical to a single in-memory index (enforced by `layers_test.go`).
- **InvertedIndex** — the delta's postings lists keyed by edge n-gram fragments and Soundex phonetic codes (segments hold the same data, delta-varint compressed)
- **VectorStore** — document and word vectors stored as float16 to halve memory; magnitudes cached separately
- **PhoneticIndex** — Soundex buckets for phonetic matching
- **Fuzzy lookup** — a Levenshtein automaton walking the FST (`internal/analysis/fst.go`, cost ∝ matches). `analysis.BKTree` is only a lazily-built fallback (edit distance above 2, or FST not built yet). Allowed edit distance scales with word length when `Config.FuzzyByLength` is set — see Configuration

**Add pipeline** (per document): `Analyzer.Analyze` → embed (in-process ONNX call via `internal/localembedder`) → write postings to the delta's InvertedIndex + PhoneticIndex + BM25. (The BK-tree is built lazily and TF-IDF is no longer maintained.)

**Search pipeline**: lexical pass (capped n-gram prefixes + phonetic + FST fuzzy) → vector pass (exact dot-product scan below `WithANNThreshold` docs, default 20k; HNSW graph in `internal/ann` above it — not persisted, rebuilt on Load and re-pointed at the new mapping after each flush/compaction) → `rankAndFuse` (RRF + BM25 tiebreak) → neural expansion if results are absent or weak

### 3. Analysis (`internal/analysis/`)

`StandardAnalyzer`: regex tokenise (camelCase-aware) → lowercase → stop-word filter → Snowball Porter2 stem → FST prefix-resolve.

`Analyze()` is for indexing. `AnalyzeQuery()` additionally expands synonyms — never call it during indexing.

### 4. Ranking (`internal/ranking/`)

- `RRFRanker` — Reciprocal Rank Fusion with k=60; input slices are copied before sorting to avoid caller mutation
- `BM25Scorer` — used as tiebreaker when RRF scores are within epsilon (1e-6)
- `TFIDFScorer` — no longer maintained by the engine; kept as a standalone library type

### 5. Embedding (`internal/localembedder/`)

An in-process embedder running a registered ONNX model (`internal/modelspec`: `gte-small` — the bundled default, chosen on measured out-of-domain hybrid quality —, `all-MiniLM-L6-v2`, `bge-small-en-v1.5`; all 384-dim, int8, sharing the BERT-uncased vocab) via ONNX Runtime. The bundled model (named by `assets/model.id`), tokenizer vocab and the platform-specific ONNX Runtime library are compiled into the Go binary with `go:embed` **only when `CGO_ENABLED=1`** (`bundle_cgo.go`; a no-CGo binary carries none of it and stays ~22 MB) and extracted to a temp directory on first use. Other registry models load from `~/.zenith/models/<id>/model.onnx` (`zenith models pull`, `--model`, `zenith.WithModel`). The model files are gitignored: `go run scripts/download_assets.go [-model id]` fetches them (and CI does so for its cgo legs). An index records `onnx:<id>` and refuses a different model; the CLI follows the model recorded in the index. Requires `CGO_ENABLED=1` and a C compiler.

The Go side wraps it with a caching layer (`internal/embedding/cache.go`, LRU of 10,000 entries). Embedding failures — including CGo being unavailable at build time — are non-fatal: `pkg/zenith`'s `buildEmbedder` and `cmd/server/main.go` both fall back to `embedding.NewDeterministicEmbedder(384)` (a hash-based, non-semantic embedder) so lexical search keeps working.

There used to be a separate FastAPI/Python sidecar (`nerve/`) serving the same model over HTTP; it was removed in favor of the in-process embedder above. `internal/sidecar/` is unrelated — it's the HTTP/JSON namespace API (see `internal/sidecar/sidecar.go`), not an embedding service.

## Key Wiring

`cmd/server/main.go` is the assembly point:

```
StandardAnalyzer → localembedder (ONNX) or DeterministicEmbedder → CachingEmbedder
RRFRanker
index.NewEngine(config, embedder, scorer, analyzer) → gRPC server
```

Index persistence is **not** the LSM storage engine (that is wired in `storage_engine.go` for the term store/WAL). `engine.Save(path)` / `engine.Load(path)` use the segment format documented in `FORMAT.md`: `zenith.db` is a small manifest, segments live beside it as `zenith.db.seg-NNNNNN`. `Save` to the bound path is an *incremental flush* (the delta becomes one new segment; cost ∝ the delta, not the index), `Save` to another path exports a merged copy, `Compact()` merges segments without holding the engine lock, and `Load` memory-maps the files. Commit = fsync the segment, then atomically rename the manifest; orphans are deleted on open. Old gob files (v4/v5) are refused with `*LegacyFormatError` (matches `ErrIncompatibleVersion`) until `zenith migrate` converts them; `Save` will not overwrite one.

## Configuration

All tuneable parameters live in `internal/config/config.go` (`DefaultConfig()`). Notable values:

- `FuzzyMaxDist` — BK-tree edit distance threshold (default 2)
- `RRFConstant` — RRF k value (default 20.0; tuned on MS MARCO dev, see the comment in `DefaultConfig()`)
- `MaxResults` — internal RRF candidate cap (default 1000), not a user-facing page size — see "Result limits" below
- `PhoneticWeight`, `VectorWeight`, `NeuralWeight` — scoring blend weights
- `MemTableMaxSize` — SSTable flush threshold (64MB)
- `NerveGRPCAddr` — dead config left over from the removed Nerve sidecar; not read anywhere in the codebase

### Metadata filtering, file format, install

- **Filtering**: `AddWithAttrs` / `AddBatchWithAttrs` attach string/bool/number attributes; `Search(..., WithFilter(Eq/In/Range/Exists/And/Or/Not))` applies them to the lexical and vector candidate sets *before* rank fusion. Attrs are persisted in the snapshot and in WAL records. `WithFilter` cannot be combined with `Explain`.
- **Format v6**: manifest header = magic + version + embedder name + vector dim, then a JSON segment list (CRC-checked). `Load` returns `ErrEmbedderMismatch` on a different embedder identity (custom embedders opt in via an optional `Name() string`; without it they're recorded as "unknown" and never checked). Migration only from the previous format (`zenith migrate` / `zenith.Migrate`, keeps a `.v<N>.bak`); a mismatch is a hard error. `zenith compact` merges segments; `zenith doctor` re-verifies every checksum. Crash safety is exercised by `internal/index/crash_test.go` (named failpoints via `ZENITH_FAILPOINT` plus hard-kill loops).
- **WAL**: Put records carry text + vector + attrs so replay never re-embeds documents; a WAL over 64MB triggers a synchronous checkpoint.
- **CLI install**: `zenith install` (per-user dir + PATH), `zenith doctor [--json]`, `zenith uninstall [--purge] [--yes]`. Uninstall keeps `~/.zenith` (the index) unless `--purge`.

### Result limits

The RRF ranker's internal candidate cap (`Config.MaxResults`) and the user-facing page size are two different things:

- **`pkg/zenith`**: `WithLimit(n)` (default 10) sets the default page size at `Open` time; `Limit(n)` overrides it per `Search` call. Both are applied in `buildResults` after the engine returns candidates — the engine itself now returns up to `MaxResults` (1000) candidates so a `Limit()` above 10 actually has something to truncate from.
- **gRPC**: `SearchRequest.limit`/`.offset` (added to `document.proto`); `internal/server.paginate` applies them, defaulting to 10 when `limit` is unset.
- **CLI** (`zenith search --max N`): slices the same underlying candidate list client-side.

Before this was wired up, the ranker was always constructed with a fixed internal cap of 10 (`topN=0` defaulting via `ranking.defaultTopN`), so no caller-side limit above 10 could ever have an effect — `Limit(30)` silently still returned 10 results on all three surfaces.

Storage engine config (`internal/storage/storage_engine.go`, `DefaultEngineConfig()`):

- `MemTableMaxSize` — freeze threshold (64MB)
- `CommitWindow` — group-committer batch window (4ms)
- `CompactorConfig.L0Threshold` — L0 file count that triggers L0→L1 compaction (default 4)
- `CompactorConfig.LevelSizeBase` — L1 byte budget (10MB); each Ln = L(n-1) × LevelSizeMult (10×)
- `CompactorConfig.CompactionInterval` — background compaction tick (30s)

## Storage implementation status

| Component | Status | Notes |
|-----------|--------|-------|
| WAL | Done | CRC-framed, SyncAlways mode; SyncPeriodic/GroupCommit stub-blocked |
| MemTable | Done | Skip-list backed (`internal/storage/memtable/skiplist.go`); O(log n) ops, pre-sorted iterator |
| SSTable | Done | Block-structured, CRC per block, Bloom filter + sparse index per file |
| Group Committer | Done | Batches concurrent flushes into one fsync |
| Leveled Compaction | Done | Background goroutine; L0 threshold + Ln size triggers; tombstone pruning at last level |
| FST dictionary | Done | Rebuilt after every flush; wired into StandardAnalyzer for prefix-search query resolution |
| WAL benchmarks | Done | `internal/storage/wal/wal_bench_test.go` — append, parallel, mixed, recovery at 1K/10K/100K |
