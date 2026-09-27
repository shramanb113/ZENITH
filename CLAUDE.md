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

- **InvertedIndex** — postings lists keyed by edge n-gram fragments and Soundex phonetic codes
- **VectorStore** — document and word vectors stored as float16 to halve memory; magnitudes cached separately
- **PhoneticIndex** — Soundex buckets for phonetic matching
- **BKTree** (`internal/analysis/bktree.go`) — Levenshtein-based fuzzy term lookup, O(log n) via triangle inequality pruning

**Add pipeline** (per document): `Analyzer.Analyze` → embed (in-process ONNX call via `internal/localembedder`) → write postings to InvertedIndex + PhoneticIndex + BKTree + BM25 + TF-IDF

**Search pipeline**: lexical pass (n-gram + phonetic + BK-tree fuzzy) → vector pass (dot product against all doc vectors) → `rankAndFuse` (RRF + BM25 tiebreak) → neural expansion if results are absent or weak

### 3. Analysis (`internal/analysis/`)

`StandardAnalyzer`: regex tokenise (camelCase-aware) → lowercase → stop-word filter → Snowball Porter2 stem → FST prefix-resolve.

`Analyze()` is for indexing. `AnalyzeQuery()` additionally expands synonyms — never call it during indexing.

### 4. Ranking (`internal/ranking/`)

- `RRFRanker` — Reciprocal Rank Fusion with k=60; input slices are copied before sorting to avoid caller mutation
- `BM25Scorer` — used as tiebreaker when RRF scores are within epsilon (1e-6)
- `TFIDFScorer` — kept in sync on every Add/Remove but not used in the main ranking path

### 5. Embedding (`internal/localembedder/`)

An in-process embedder that runs `all-MiniLM-L6-v2` (384-dimensional, int8-quantized, ~22MB) via ONNX Runtime. The model, tokenizer vocab, and platform-specific ONNX Runtime shared library are all bundled into the Go binary with `go:embed` (`internal/localembedder/assets/`) and extracted to a temp directory on first use — no separate process, no Python, no network call. Requires `CGO_ENABLED=1` and a C compiler.

The Go side wraps it with a caching layer (`internal/embedding/cache.go`, LRU of 10,000 entries). Embedding failures — including CGo being unavailable at build time — are non-fatal: `pkg/zenith`'s `buildEmbedder` and `cmd/server/main.go` both fall back to `embedding.NewDeterministicEmbedder(384)` (a hash-based, non-semantic embedder) so lexical search keeps working.

There used to be a separate FastAPI/Python sidecar (`nerve/`) serving the same model over HTTP; it was removed in favor of the in-process embedder above. `internal/sidecar/` is unrelated — it's the HTTP/JSON namespace API (see `internal/sidecar/sidecar.go`), not an embedding service.

## Key Wiring

`cmd/server/main.go` is the assembly point:

```
StandardAnalyzer → localembedder (ONNX) or DeterministicEmbedder → CachingEmbedder
RRFRanker
index.NewEngine(config, embedder, scorer, analyzer) → gRPC server
```

Index persistence uses `encoding/gob` (not the LSM storage engine) via `engine.Save("zenith.db")` / `engine.Load("zenith.db")`. The LSM engine is wired in `storage_engine.go` but the index-layer persistence is separate.

## Configuration

All tuneable parameters live in `internal/config/config.go` (`DefaultConfig()`). Notable values:

- `FuzzyMaxDist` — BK-tree edit distance threshold (default 2)
- `RRFConstant` — RRF k value (default 20.0; tuned on MS MARCO dev, see the comment in `DefaultConfig()`)
- `MaxResults` — internal RRF candidate cap (default 1000), not a user-facing page size — see "Result limits" below
- `PhoneticWeight`, `VectorWeight`, `NeuralWeight` — scoring blend weights
- `MemTableMaxSize` — SSTable flush threshold (64MB)
- `NerveGRPCAddr` — dead config left over from the removed Nerve sidecar; not read anywhere in the codebase

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
