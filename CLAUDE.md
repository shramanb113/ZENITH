# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

**Search pipeline**: lexical pass (capped n-gram prefixes + phonetic + FST fuzzy) → vector pass (exact dot-product scan below `WithANNThreshold` docs, default 20k; HNSW graph in `internal/ann` above it — persisted as the `<db>.ann` sidecar, see "Frozen layer, checkpoints, ANN sidecar" below) → `rankAndFuse` (RRF + BM25 tiebreak) → neural expansion if results are absent or weak

### 3. Analysis (`internal/analysis/`)

`StandardAnalyzer`: regex tokenise (camelCase-aware) → lowercase → stop-word filter → Snowball Porter2 stem → FST prefix-resolve. `Segment()` (`internal/analysis/segment.go`) splits Han/Hiragana/Katakana/Hangul characters one span per character — those scripts carry no whitespace between words, so a whole sentence would otherwise become one oversized token; no dictionary segmenter is used (none is in go.mod). Other non-ASCII scripts (Devanagari, Arabic, ...) are whitespace-delimited already and stay one span per run, matras/marks attached via `isWordRune`. Thai (or any other script needing real dictionary segmentation) is a known, documented gap, not built.

`Analyze()` is for indexing. `AnalyzeQuery()` additionally expands synonyms — never call it during indexing.

### 4. Ranking (`internal/ranking/`)

- `RRFRanker` — Reciprocal Rank Fusion with k=60; input slices are copied before sorting to avoid caller mutation
- `BM25Scorer` — used as tiebreaker when RRF scores are within epsilon (1e-6)
- `TFIDFScorer` — no longer maintained by the engine; kept as a standalone library type

### 5. Embedding (`internal/localembedder/`)

An in-process embedder running a registered ONNX model (`internal/modelspec`: `gte-small` — the bundled default, chosen on measured out-of-domain hybrid quality —, `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, all 384-dim, int8, sharing the bert-base-uncased WordPiece vocab; plus `labse` and `distiluse-multilingual`, opt-in multilingual — see below) via ONNX Runtime. The bundled model (named by `assets/model.id`), tokenizer vocab and the platform-specific ONNX Runtime library are compiled into the Go binary with `go:embed` **only when `CGO_ENABLED=1`** (`bundle_cgo.go`; a no-CGo binary carries none of it and stays ~22 MB) and extracted to a temp directory on first use. Other registry models load from `~/.zenith/models/<id>/model.onnx` (`zenith models pull`, `--model`, `zenith.WithModel`). The model files are gitignored: `go run scripts/download_assets.go [-model id]` fetches them (and CI does so for its cgo legs). An index records `onnx:<id>` and refuses a different model; the CLI follows the model recorded in the index. Requires `CGO_ENABLED=1` and a C compiler.

`Spec.HiddenDims` lets a model's final output size (`Dims`, what's recorded in the index header) differ from its transformer's native hidden size — needed when `DenseURL` points at a non-square projection (distiluse-multilingual: 768 hidden -> 512 output) rather than a square one (LaBSE: 768 -> 768); zero means "same as Dims," so every model without this need is unaffected. `Spec.NoTokenTypeIDs` is true for DistilBERT-family models, whose ONNX graph has no `token_type_ids` input at all (unlike BERT) — offering one makes onnxruntime refuse the run ("Invalid input name"); `internal/localembedder/model.go`'s session creation and `infer` build that input only when the model actually has it.

The Go side wraps it with a caching layer (`internal/embedding/cache.go`, LRU of 10,000 entries). Embedding failures — including CGo being unavailable at build time — are non-fatal: `pkg/zenith`'s `buildEmbedder` and `cmd/server/main.go` both fall back to `embedding.NewDeterministicEmbedder(384)` (a hash-based, non-semantic embedder) so lexical search keeps working.

**Multilingual (`labse`, `distiluse-multilingual`, both opt-in):** `Spec.VocabURL` lets a registry entry ship its own vocab (fetched by `zenith models pull <id>` alongside the model, not bundled) and `Spec.Cased` tells the WordPiece tokenizer to skip lowercasing — both default to empty/false so every existing model is unaffected. `labse`'s base ONNX export (`Xenova/LaBSE`) exposes `last_hidden_state` only — not the full sentence-transformers LaBSE pipeline. Rather than approximate that gap with a different pooling choice, `Spec.DenseURL` fetches the real missing piece: `sentence-transformers/LaBSE`'s small separate `2_Dense` module (a trained Linear(768,768)+Tanh projection, ~2.4 MB safetensors, saved alongside the much larger base model) is downloaded by `zenith models pull labse` and applied in `internal/localembedder` after CLS pooling (confirmed via that repo's `1_Pooling/config.json` / `2_Dense/config.json`, not guessed), closing the gap instead of working around it. `Spec.DenseURL` is empty for every model that doesn't need it. Picking a multilingual ONNX export matters: the obvious `bert-base-multilingual-cased` export on Xenova's hub is a `BertForMaskedLM` head (output `logits` over the vocab, not hidden states) and cannot produce embeddings at all — verified by inspecting the ONNX graph's output names/shapes before wiring it in, twice (once for the bundled-repo export, once for the legacy `transformers.js` monorepo export — both carry the same head). `distiluse-multilingual` (`Xenova/distiluse-base-multilingual-cased-v2`, confirmed `DistilBertModel` architecture — a real feature-extraction export) is used instead: a genuine sentence-transformers model (mean pooling, `2_Dense` Linear(768,512)+Tanh, knowledge-distilled for cross-lingual similarity across 50+ languages, not naive mean-pooled vanilla BERT), smaller than LaBSE. Both models are evaluated by `internal/index/multilingual_beir_test.go` (`ZENITH_MODEL_EVAL=1`): real nDCG@10/Recall@10 against a small hand-curated fixture set (`testdata/multilingual/<lang>/`, 7 languages) using the same harness as `TestBEIR`, replacing the old dot-product-only similarity check in `multilingual_test.go` — this is smoke-scale (16 passages/6 queries per language, not independently reviewed by native speakers), not BEIR/MIRACL-rigor benchmarking.

There used to be a separate FastAPI/Python sidecar (`nerve/`) serving the same model over HTTP; it was removed in favor of the in-process embedder above. `internal/sidecar/` is unrelated — it's the HTTP/JSON API (see `internal/sidecar/sidecar.go`), not an embedding service, and it never touches `zenith.db`. `/v1/ns/*` namespaces are `:memory:` engines, rebuilt on every `PUT` and gone on restart or LRU eviction. `/v1/collections/*` are persistent: each collection is its own `pkg/zenith` DB under `--collections-dir` (`internal/collections`), lazily opened, idle-closed, never auto-deleted, and durable through the library's WAL. The `--db` gRPC/CLI raw-engine path is unchanged and still has no WAL.

## Key Wiring

`cmd/server/main.go` is the assembly point:

```
StandardAnalyzer → localembedder (ONNX) or DeterministicEmbedder → CachingEmbedder
RRFRanker
index.NewEngine(config, embedder, scorer, analyzer) → gRPC server
```

Index persistence is **not** the LSM storage engine (that is wired in `storage_engine.go` for the term store/WAL). `engine.Save(path)` / `engine.Load(path)` use the segment format documented in `FORMAT.md`: `zenith.db` is a small manifest, segments live beside it as `zenith.db.seg-NNNNNN`. `Save` to the bound path is an _incremental flush_ (the delta becomes one new segment; cost ∝ the delta, not the index), `Save` to another path exports a merged copy, `Compact()` merges segments without holding the engine lock, and `Load` memory-maps the files. Commit = fsync the segment, then atomically rename the manifest; orphans are deleted on open. Old gob files (v4/v5) are refused with `*LegacyFormatError` (matches `ErrIncompatibleVersion`) until `zenith migrate` converts them; `Save` will not overwrite one.

## Configuration

All tuneable parameters live in `internal/config/config.go` (`DefaultConfig()`). Notable values:

- `FuzzyMaxDist` — BK-tree edit distance threshold (default 2)
- `RRFConstant` — RRF k value (default 20.0; tuned on MS MARCO dev, see the comment in `DefaultConfig()`)
- `MaxResults` — internal RRF candidate cap (default 1000), not a user-facing page size — see "Result limits" below
- `PhoneticWeight`, `VectorWeight`, `NeuralWeight` — scoring blend weights
- `MemTableMaxSize` — SSTable flush threshold (64MB)
- `NerveGRPCAddr` — dead config left over from the removed Nerve sidecar; not read anywhere in the codebase

### Metadata filtering, file format, install

- **Filtering**: `AddWithAttrs` / `AddBatchWithAttrs` attach string/bool/number attributes; `Search(..., WithFilter(Eq/In/Range/Exists/And/Or/Not))` applies them to the lexical and vector candidate sets _before_ rank fusion. Attrs are persisted in the snapshot and in WAL records. `WithFilter` combines with `Explain`.
  - A filter is data (`index.FilterSpec`, JSON ops `eq`, `in`, `range`, `exists`, `and`, `or`, `not`, `none`; depth ≤ 16, ≤ 512 nodes), so the same filter works on every surface: `zenith.FilterFromJSON`, gRPC `SearchRequest.filter` (`FilterNode`) and `IndexRequest.attrs`, the sidecar's JSON body, and the CLI (`zenith index --attr k=v`, `zenith search --where k=v` / `--filter '<json>'`).
  - `attrIndex` (`internal/index/attrindex.go`) keeps per-field value → ordinal postings. A selective filter (estimate ≤ `max(2000, docs/50)`) is resolved to a candidate set and the vector pass scores only those; `Not` falls back to a scan. The index-path result equals the scan-path result (property test).
  - Journal value format with attrs: `0xFF 'Z' 'A' '1' | uvarint(len) | attrs JSON | text` (`internal/index/journal.go`).

### Frozen layer, checkpoints, ANN sidecar

- **Non-blocking flush.** A flush freezes the delta in O(1) under `Engine.mu` into `Engine.frozen` (a read-only layer that is still searched, with its own dead set), swaps in fresh sub-indexes, writes the segment with no engine lock, then swaps it in and commits the manifest. A failed write leaves the frozen layer for a retry. Deletions of frozen docs are carried in `pendingDels` to the next flush. See `internal/index/frozen.go`.
- **Non-blocking checkpoint** (`pkg/zenith/checkpoint.go`): under `db.mu` freeze + `WAL.Rotate` to `<db>.wal.old-NNNNNN`; then, unlocked, `FinishCheckpoint` and delete the archives. Recovery replays archives oldest first, then the live WAL (adds are idempotent, later deletes win).
- **Fault-injection filesystem** (`internal/fsx`): a `Recorder` that reconstructs the on-disk state after a power cut (`Nothing` / `Everything` / `Torn` at 4 KiB page granularity, directory-op prefixes, atomic rename) and a `DiskFull` byte budget (ENOSPC). `internal/index/powerloss_test.go` and `pkg/zenith/powerloss_test.go` sweep every cut point. Segment `Finish` syncs the directory before the manifest rename; `OpenWAL` syncs the directory when it creates the file.
- **ANN sidecar** `<db>.ann` (`internal/ann/persist.go`, `internal/index/ann_persist.go`): magic `ZANN`, version, params, embedder identity, CRC32C. It is a _hint_: nodes carry the segment-generation tag and are bound by id + tag at load; docs the graph lacks are inserted, nodes it holds for docs that are gone are tombstoned, and the file is ignored if more than half the docs would need inserting or the identity differs. Written after compaction, on `Close`, and in the background after 50,000 graph changes.
- **Real full-disk test**: `pkg/zenith/realfs_full_test.go` (`TestRealFS_DiskFull`) runs against a real small filesystem when `ZENITH_SMALLFS` points at one (e.g. a 6 MB tmpfs on Linux/WSL; cross-compile with `GOOS=linux CGO_ENABLED=0 go test -c ./pkg/zenith`). It skips otherwise. Real _power loss_ is only modelled (fsx).
- **Hybrid fusion shortcut**: `internal/ranking/rrf_hybrid.go` computes the exact top-N of weighted RRF without sorting every keyword candidate when the vector list is small (≤ 4096); it is tested equal to the full sort. ANN candidate counts are `annK=300` / `annEF=400` in `internal/index/ann_glue.go` (the knee of the measured recall/latency sweep — see ROADMAP P1-5 before changing them).
- **Query embedding overlap**: `SearchFiltered` starts the query embedding in a goroutine and runs the lexical phase meanwhile; if the embedding is not ready within `embedHoldMax` (100 ms) it releases the read lock, waits, and redoes the lexical phase.
- **Format v6**: manifest header = magic + version + embedder name + vector dim, then a JSON segment list (CRC-checked). `Load` returns `ErrEmbedderMismatch` on a different embedder identity (custom embedders opt in via an optional `Name() string`; without it they're recorded as "unknown" and never checked). Migration only from the previous format (`zenith migrate` / `zenith.Migrate`, keeps a `.v<N>.bak`); a mismatch is a hard error. `zenith compact` merges segments; `zenith doctor` re-verifies every checksum. Crash safety is exercised by `internal/index/crash_test.go` (named failpoints via `ZENITH_FAILPOINT` plus hard-kill loops).
- **WAL**: Put records carry text + vector + attrs so replay never re-embeds documents; a WAL over 64MB triggers a checkpoint that does not block searches (freeze + WAL rotation under the lock, the segment write outside it).
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

| Component          | Status | Notes                                                                                         |
| ------------------ | ------ | --------------------------------------------------------------------------------------------- |
| WAL                | Done   | CRC-framed, SyncAlways mode; SyncPeriodic/GroupCommit stub-blocked                            |
| MemTable           | Done   | Skip-list backed (`internal/storage/memtable/skiplist.go`); O(log n) ops, pre-sorted iterator |
| SSTable            | Done   | Block-structured, CRC per block, Bloom filter + sparse index per file                         |
| Group Committer    | Done   | Batches concurrent flushes into one fsync                                                     |
| Leveled Compaction | Done   | Background goroutine; L0 threshold + Ln size triggers; tombstone pruning at last level        |
| FST dictionary     | Done   | Rebuilt after every flush; wired into StandardAnalyzer for prefix-search query resolution     |
| WAL benchmarks     | Done   | `internal/storage/wal/wal_bench_test.go` — append, parallel, mixed, recovery at 1K/10K/100K   |
