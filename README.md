# ZENITH

[![CI](https://github.com/shramanb113/ZENITH/actions/workflows/ci.yml/badge.svg)](https://github.com/shramanb113/ZENITH/actions/workflows/ci.yml)

> Local-first hybrid search — as a CLI tool and as a Go library.

ZENITH is a search engine built from scratch in Go. It understands meaning, not just keywords — combining lexical matching, BK-tree fuzzy search, and neural vector embeddings into a single hybrid pipeline, ranked with Reciprocal Rank Fusion.

You can use it two ways:

- **As a CLI tool** — install once, point at directories, search from your terminal
- **As a Go library** — `go get` it, call three methods, ship search inside your app

The storage layer is a full LSM-tree (WAL → MemTable → SSTable → Bloom filters), built from scratch. The embedding model (`gte-small`, int8 ONNX; swappable, see [Choosing an embedding model](#choosing-an-embedding-model)) is baked into the binary — no Python, no server, no setup step.

---

## Why ZENITH

Most search tools are server processes — you run Elasticsearch or Meilisearch separately and talk to them over HTTP. That is ops overhead, a network hop, and an external dependency your app cannot ship without.

ZENITH is a library. It compiles into your binary and runs in your process. Same relationship to search that SQLite has to databases: import it, call a few methods, done.

> **ZENITH is to search what SQLite is to databases — zero dependencies, embeds in your app, ships in your binary.**

See [DECISIONS.md](./DECISIONS.md) for the full architectural and strategic reasoning.

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Go | 1.24+ | Required |
| C compiler | gcc / MinGW-w64 | Only to build semantic search from source (CGo). Not needed for the prebuilt release binaries, nor for the lexical-only build. See note below. |
| Ollama | any | Optional — alternative embedder |

**C compiler note:** ZENITH uses CGo to run its embedding model (`gte-small` by default) in-process via ONNX Runtime. On Windows, install MinGW-w64:

```powershell
winget install -e --id MSYS2.MSYS2
# In the MSYS2 UCRT64 terminal:
pacman -S --noconfirm mingw-w64-ucrt-x86_64-gcc
# Add C:\msys64\ucrt64\bin to PATH
```

On Linux/macOS: `sudo apt install gcc` / `xcode-select --install`. If no C compiler is available, ZENITH still builds and runs — it falls back to deterministic (hash-based) embeddings automatically. Lexical and fuzzy search work at full quality; only neural semantic search requires CGo.

---

## Use as a CLI tool

### Install

```bash
go install github.com/shramanb113/ZENITH/cmd/zenith@latest
```

One command. No setup step. The embedding model is embedded in the binary — ZENITH is ready immediately.

**Prebuilt binaries.** Each version tag publishes archives with checksums (built by `.github/workflows/release.yml`) so you need no Go toolchain and no C compiler:

| Archive | What it is |
|---|---|
| `zenith_<version>_<os>_<arch>` | Full build: embedded ONNX model, semantic + lexical search (linux/amd64, darwin/arm64, darwin/amd64, windows/amd64) |
| `zenith-lexical_<version>_<os>_<arch>` | Built without CGo, ~20 MB: BM25 / n-gram / phonetic / fuzzy search only. A supported mode, not a fallback — `zenith doctor` reports which one you are running |

`zenith doctor` checks your install, and verifies the index's checksums.

`go install` puts the binary in `$(go env GOPATH)/bin`, which is often *not* on your PATH. If `zenith` is "not found", or you are running an older copy than the one you just installed, let ZENITH fix it itself:

```bash
zenith install        # copy this binary to a per-user dir and put it on PATH (safe to re-run)
zenith doctor         # diagnose PATH / data dir / auto-start / embedder problems, with the fix for each
```

`zenith install` uses `%LOCALAPPDATA%Programszenith` on Windows and `~/.local/bin` on Linux/macOS (a marked, removable block is added to your shell rc). Add `--autostart` to also register the login watcher, `--dir` to pick another location, `--no-path` to leave PATH alone. Open a new terminal afterwards.

### Uninstall

```bash
zenith uninstall            # removes binary, PATH entry, auto-start; KEEPS your index in ~/.zenith
zenith uninstall --purge    # also deletes the index and settings (irreversible)
zenith uninstall --yes      # skip the confirmation prompt (scripts)
```

Your search index is never deleted unless you pass `--purge`. On Windows the running `.exe` is removed automatically a moment after the command exits. `zenith doctor --json` prints a shareable, contents-free report (versions, OS, paths, check results) for support requests; ZENITH sends nothing anywhere on its own.

```bash
# Index a directory
zenith index ~/Documents

# Search — hybrid lexical + fuzzy + semantic
zenith search "kubernetes memory debugging"

# More results
zenith search -n 20 "kubernetes memory debugging"

# Watch a directory and auto-index changes in real time
zenith watch add ~/Documents
zenith watch install      # register OS boot auto-start
zenith watch start        # start watching now

# Start gRPC server for remote access
zenith serve
```

No `zenith setup` required. That command existed in v1 to install the Python embedding packages. It is now a no-op — everything ships in the binary.

### Upgrading an existing index (on-disk format change)

The index is now a small manifest (`zenith.db`) plus memory-mapped, immutable segment files next to it (`zenith.db.seg-000001`, …): opening is fast regardless of size, most of the index lives in the OS page cache rather than the Go heap, and saving writes only what changed. See [FORMAT.md](./FORMAT.md) for the layout and the compatibility policy.

An index written by an older release is **refused with a clear error, never overwritten or misread**, until you convert it:

```bash
zenith migrate            # in place; keeps the original as zenith.db.v5.bak
zenith doctor             # confirms the format and re-verifies every checksum
zenith compact            # optional: merge segments and reclaim space
```

Migration needs no embedding model (vectors are copied, not recomputed), swaps the new index in atomically, and re-opens and checksums it before reporting success. From Go: `zenith.Migrate(path, "")`.

### Upgrading from v1 (Python sidecar)

If you installed ZENITH before this release, you have a `~/.zenith/nerve/` directory containing a Python virtualenv (~600 MB to 2 GB). The new binary handles cleanup automatically — on first run you will see:

```
  ✓  Migrated to native Go (780 MB freed)
```

To clean up manually at any time:

```bash
zenith setup --clean
```

This kills any running nerve process, removes `~/.zenith/nerve/`, and recovers the disk space. Your index data (`zenith.db`) is untouched.

---

## Use as a Go library

### Install

```bash
go get github.com/shramanb113/ZENITH/pkg/zenith
```

### Usage

```go
import "github.com/shramanb113/ZENITH/pkg/zenith"

// Open a persistent index (created if it does not exist)
db, err := zenith.Open("search.db")
if err != nil {
    log.Fatal(err)
}
defer db.Close()

// Index a document
err = db.Add(ctx, "post-1", "How to debug memory leaks in Go applications")

// Search — hybrid lexical + fuzzy + semantic
results, err := db.Search(ctx, "golang memory debugging")
for _, r := range results {
    fmt.Println(r.ID, r.Score)
}

// Remove a document
err = db.Delete(ctx, "post-1")
```

### In-memory mode (for tests)

```go
// No files written, survives only for the process lifetime
db, err := zenith.Open(":memory:")
```

Use `:memory:` in tests for zero-cleanup, parallel-safe isolation. Same API as persistent mode — no code changes needed between test and production.

### Functional options (power users)

```go
// Bring your own embedder
db, err := zenith.Open("search.db",
    zenith.WithEmbedder(myEmbedder),
)

// Lexical + fuzzy only — disable vector search
db, err := zenith.Open("search.db",
    zenith.WithBM25Only(),
)

// Tune result count
results, err := db.Search(ctx, "query", zenith.WithLimit(20))
```

The zero-argument happy path uses the embedded ONNX model automatically. Functional options exist for the cases where you need control — they never change the default behaviour.

---

## How Embedding Works

ZENITH ships the `gte-small` embedding model (int8 ONNX, ~34 MB) **baked into the binary** via `go:embed`. Inference runs in-process via `yalue/onnxruntime_go` — no Python, no network hop, no separate process.

### Choosing an embedding model

```bash
zenith models list                          # what is available, and what is installed
zenith models pull all-MiniLM-L6-v2         # download one into ~/.zenith/models
zenith index ~/notes --model all-MiniLM-L6-v2
```

From Go: `zenith.Open(path, zenith.WithModel("all-MiniLM-L6-v2"))`. A model that cannot be loaded is an error, never a silent fall back to another one.

**An index remembers its model** and refuses to open with a different one (mixing vector spaces gives plausible-looking but wrong results). The CLI follows the model recorded in the index automatically, so upgrading never strands an existing index: one built with `all-MiniLM-L6-v2` (the bundled model in earlier releases) keeps working after a one-time `zenith models pull all-MiniLM-L6-v2`. To switch an index to another model, re-index — your documents are untouched.

**Why `gte-small` is the default** (measured here, `ZENITH_MODEL_EVAL` / `ZENITH_SCIFACT_HYBRID` / `ZENITH_MSMARCO_HYBRID` tests; a 14-thread desktop):

| Model | Size | Embed 20k docs | Dense-only nDCG@10: MS MARCO / SciFact | **Hybrid** nDCG@10 on SciFact |
|---|---|---|---|---|
| lexical only (no model) | — | — | — | 0.689 |
| all-MiniLM-L6-v2 | 23 MB | 2m19s | 0.946 / 0.652 | 0.691 |
| bge-small-en-v1.5 | 34 MB | 4m30s | 0.961 / 0.699 | 0.727 |
| **gte-small** | 34 MB | 3m52s | 0.965 / 0.714 | **0.738** |

On out-of-domain text (SciFact, scientific claims) the previous default added almost nothing to BM25 in the full hybrid pipeline (+0.2 nDCG points), while `gte-small` adds +5.0, and it beats the previous default by +4.7 nDCG points and +4.3 points of Recall@10. On MS MARCO (which these models were trained near) the two are **tied** in hybrid (Recall@10 0.962 = 0.962 over 107,399 passages). The price is about 1.7× the embedding time (107k passages: 21m41s vs 12m45s) and an 11 MB larger binary; if indexing speed matters more than out-of-domain quality, `--model all-MiniLM-L6-v2` is the faster choice. SciFact is one out-of-domain task, not a guarantee for your documents.

On every query and document add, ZENITH runs this cascade automatically:

```
1. Local ONNX model (embedded, always available if built with CGo)
2. Ollama at localhost:11434       (if --embedder ollama)
3. Deterministic hash embeddings   (fallback — zero dependencies)
```

Cold start with the embedded model: **under 1 second.** The ONNX session is initialised once at startup and held for the process lifetime.

### Multilingual search (opt-in, `labse`)

```bash
zenith models pull labse
zenith index ~/multilingual-notes --model labse
zenith search "capital of France" --model labse   # matches a French-language passage
```

`labse` is cross-lingual: an English query can match a passage written in French, German, Spanish, Hindi, or Japanese (smoke-tested across those five; LaBSE itself covers 109 languages). It is CLS-pooled directly from the ONNX export's `last_hidden_state`, not the full sentence-transformers LaBSE pipeline (which adds a final dense+normalize layer this export doesn't expose) — expect useful, not published-LaBSE-benchmark, cross-lingual retrieval. Needs its own index: it is 768-dim and cannot share a database with the 384-dim default models.

### Use Ollama instead

```bash
ollama pull nomic-embed-text
zenith index --embedder ollama ~/Documents
zenith search --embedder ollama "query"
```

### Deterministic only (fully offline, no C compiler)

```bash
zenith index --embedder deterministic ~/Documents
zenith search --embedder deterministic "query"
```

---

## CLI Reference

### `zenith index`

```bash
zenith index ~/Documents
zenith index --embedder ollama ~/Documents
zenith index --db my-index.db ~/Projects
```

**Flags:**
```
--db           string   Index database file             (default: ~/.zenith/zenith.db)
--fst          string   On-disk FST path                (default: ~/.zenith/data/index.fst)
--embedder     string   auto | local | ollama | deterministic  (default: auto)
--ollama-url   string   Ollama server URL               (default: http://localhost:11434)
--ollama-model string   Ollama embedding model          (default: nomic-embed-text)
--attr         key=value  Attach metadata to every indexed document (repeatable), e.g.
                           --attr tenant=acme --attr year=2024 --attr public=true.
                           Search it back with --where / --filter (see "Metadata filtering").
```

---

### `zenith search`

```bash
zenith search "kubernetes memory debugging"
zenith search -n 20 "kubernetes memory debugging"
zenith search --embedder deterministic "offline query"
zenith search --db my-index.db "query"
```

**Flags:**
```
-n, --max int    Maximum results to display             (default: 10)
--db      string Index database file                   (default: ~/.zenith/zenith.db)
--embedder string auto | local | ollama | deterministic (default: auto)
--model   string Embedding model id, from `zenith models list` (default: the model
                  recorded in the index; mismatches with it are an error)
--where   string Attribute condition, repeatable, all must hold: key=value, key!=value,
                  key>=n, key<=n. See "Metadata filtering".
--filter  string Full JSON filter expression (and/or/not of eq/in/range/exists), combined
                  with any --where conditions. See "Metadata filtering".
--rerank         Rerank the hybrid shortlist with a cross-encoder
                  (needs: zenith models pull ms-marco-MiniLM-L-6-v2)
--rerank-model   Reranker id from `zenith models list --rerankers` (default:
                  ms-marco-MiniLM-L-6-v2)
```

---

### `zenith watch`

```bash
zenith watch add ~/Documents       # add to persistent watchlist
zenith watch list                  # view watchlist
zenith watch remove ~/Documents    # remove from watchlist
zenith watch start                 # watch all listed directories (blocks)
zenith watch install               # register OS boot auto-start
zenith watch uninstall             # remove OS boot auto-start
zenith watch run ~/Downloads       # one-shot watch, not saved to list
zenith watch run --index-first ~/Documents
```

**Auto-start locations:**
| Platform | Location |
|---|---|
| Windows | Task Scheduler — `ZenithWatch`, triggers at login |
| Linux | `~/.config/systemd/user/zenith-watch.service` |
| macOS | `~/Library/LaunchAgents/com.zenith.watch.plist` |

---

### `zenith log`

```bash
zenith log              # last 50 events
zenith log -n 100       # last 100 events
zenith log -f           # stream live (like tail -f)
zenith log --type SEARCH
zenith log --type INDEXED -f
```

---

### Other commands

```bash
zenith serve            # start gRPC server (127.0.0.1:8080; --bind 0.0.0.0 + --key to expose)
zenith serve --port 9090
zenith version          # print version
zenith update           # update to latest release
zenith install          # put this binary on your PATH
zenith doctor           # diagnose install problems (--json for a shareable report)
zenith uninstall        # remove binary/PATH/auto-start (index kept; --purge deletes it)
zenith setup --clean    # remove leftover Python files from v1
```

---

## Architecture

```
zenith index <path>
      │
      ▼
┌─────────────┐     ┌──────────────────────┐     ┌──────────────────────────────┐
│   Crawler   │────▶│       Analyzer        │────▶│         Embedder             │
│  (fsnotify) │     │                      │     │                              │
│  recursive  │     │  regex tokenise      │     │  cascade (default: auto):    │
│  dir walk   │     │  camelCase-aware     │     │  1. ONNX  gte-small          │
│  + live     │     │  lowercase           │     │     (embedded, in-process)   │
│  watching   │     │  stop-word filter    │     │  2. Ollama nomic-embed-text  │
│             │     │  Porter2 stem        │     │  3. deterministic (fallback) │
│             │     │  FST prefix-resolve  │     │                              │
│             │     │  synonym expand      │     │  → []float32 (384-dim)       │
└─────────────┘     └──────────────────────┘     └──────────────┬───────────────┘
                                                                │
                               ┌────────────────────────────────┘
                               ▼
                       ┌───────────────┐
                       │  LSM Storage  │
                       │               │
                       │  WAL          │  crash-safe append log
                       │  MemTable     │  skip-list, O(log n)
                       │  SSTable      │  immutable, block-structured
                       │  Bloom filter │  O(1) miss bypass
                       │  Sparse index │  memory-efficient offsets
                       │  Compaction   │  leveled, background worker
                       └───────┬───────┘
                               │
zenith search <query>          ▼
      │                ┌───────────────┐
      ├───────────────▶│  Query Engine │
      │  lexical       │               │
      │  fuzzy         │  BM25 scoring │
      │  semantic      │  TF-IDF       │
      │                │  edge n-grams │
      │                │  phonetic     │
      │                │  BK-tree      │  O(log n) fuzzy
      │                │  vector cosine│  dot product
      │                │  RRF fusion   │  merges all signals
      │                └───────────────┘
      │
      └──▶ zenith serve ──▶ gRPC server (:8080)
```

---

## Feature Status

| Feature | Status | Notes |
|---|---|---|
| CLI (`index`, `search`, `watch`, `serve`, `version`) | Active | |
| Go library (`pkg/zenith`) | Active | `go get` — embed search in your app |
| LSM storage (WAL, MemTable, SSTable, Compaction) | Active | Ground-up implementation |
| Bloom filter + sparse index | Active | Per-SSTable |
| BK-tree fuzzy matching | Active | O(log n) Levenshtein |
| BM25 + TF-IDF scoring | Active | |
| Edge n-gram prefix search | Active | |
| Phonetic matching (Soundex) | Active | |
| Synonym expansion | Active | |
| FST term dictionary | Active | Rebuilt after every flush |
| RRF hybrid ranking | Active | |
| ONNX embedder (embedded, in-process) | Active | gte-small (default) / all-MiniLM-L6-v2 / bge-small-en-v1.5, int8, CGo |
| Deterministic embedder | Active | Fallback — zero dependencies |
| Ollama embedder | Active | Needs Ollama running |
| gRPC server (`zenith serve`) | Active | Port 8080 default |
| fsnotify incremental watching | Active | |
| Persistent watchlist | Active | `~/.zenith/watchlist.json` |
| Boot auto-start (Windows/Linux/macOS) | Active | |
| Text, Markdown, Go, HTML extractors | Active | |
| PDF extraction | Active | Pure Go via `ledongthuc/pdf` |
| Image indexing (filename + OCR) | Active | OCR opt-in: build with `-tags ocr` + `CGO_ENABLED=1` against libtesseract; falls back to filename-only otherwise |
| In-memory mode (`:memory:`) | Active | Library — zero-cleanup testing |
| OpenAI embedder | Planned | |
| Prometheus metrics | Planned | |
| OpenTelemetry traces | Planned | |

---

## Storage Engine

Full LSM-tree — same architecture as RocksDB and LevelDB, built from scratch.

| Component | Implementation | Status |
|---|---|---|
| Write-Ahead Log | Append-only, CRC-framed, crash-safe | Active |
| MemTable | Skip-list, sorted, O(log n) ops | Active |
| SSTable | Immutable block-structured sorted files | Active |
| Group Committer | Batches concurrent flushes into one fsync | Active |
| Leveled Compaction | Background goroutine, tombstone pruning | Active |
| Bloom Filter | Probabilistic O(1) disk-lookup bypass | Active |
| Sparse Index | Memory-efficient offset map per SSTable | Active |
| FST Dictionary | Rebuilt after every flush, prefix-resolve | Active |

---

## Search Pipeline

**Lexical:** inverted index with BM25 and TF-IDF scoring, Porter2 stemming, edge n-grams, phonetic matching (Soundex), synonym expansion.

**Fuzzy:** a Levenshtein automaton walking the FST (cost proportional to the matches, not the vocabulary). The allowed edit distance scales with word length — 2–3 letter words match exactly, 4–5 letters tolerate one edit, longer words two (`FuzzyMaxDist`, `FuzzyByLength` in `internal/config/config.go`). A fixed distance of 2 on every word made short words match hundreds of unrelated terms; on MS MARCO the length rule (with a cap on very common prefixes) cut median lexical latency from 201 ms to 43 ms with Recall@10 unchanged (0.830) and typo-query recall no worse (0.550 → 0.553).

**Semantic:** `gte-small` (384-dim, selectable) via ONNX Runtime, running in-process. Vector cosine similarity stored as float16 to halve memory. Embedding failures are non-fatal — engine degrades to lexical-only.

**Ranking:** Reciprocal Rank Fusion (RRF, k=20, tuned on MS MARCO dev) merges all result lists. BM25 tiebreaker when RRF scores are within epsilon. All weights tunable in `internal/config/config.go`.

---

## File Support

| Format | Extraction |
|---|---|
| `.txt` `.md` `.log` `.csv` `.json` `.yaml` | Raw UTF-8 text |
| `.go` | AST — identifiers, comments, package name |
| `.py` `.ts` `.js` `.jsx` `.tsx` `.rs` `.java` `.c` `.cpp` | Raw source |
| `.html` `.htm` | Tag-stripped visible text |
| `.pdf` | Pure-Go text extraction via `ledongthuc/pdf` |
| `.jpg` `.jpeg` `.png` `.gif` `.bmp` `.webp` `.tiff` | Filename + directory path tokens, plus OCR'd pixel text when built with `-tags ocr` (Tesseract) |

---

## gRPC API

```protobuf
service SearchService {
  rpc IndexDocuments(IndexRequest)         returns (IndexResponse);
  rpc Search(SearchRequest)                returns (SearchResponse);
  rpc IndexPDF(IndexPDFRequest)             returns (IndexPDFResponse);
  rpc GetDocument(GetDocumentRequest)       returns (GetDocumentResponse);
  rpc DeleteDocument(DeleteDocumentRequest) returns (DeleteDocumentResponse);
}
```

`Search` fuses lexical (BM25 + n-gram + phonetic + BK-tree fuzzy) and vector scoring via RRF — there's no separate `FuzzySearch`/`HybridSearch` RPC, it's all one `Search` call. `SearchRequest` takes an optional `limit`/`offset` for pagination (default limit: 10), and an optional `filter` (`FilterNode`) — see "Metadata filtering" below. See `internal/proto/document.proto` for the source of truth.

### Authentication & binding

`zenith serve` binds `127.0.0.1` by default. `--bind 0.0.0.0` (or any other non-loopback
address) refuses to start unless a shared key is set (`--key` / `ZENITH_KEY`) or you pass
`--allow-unauthenticated` — a server is never silently exposed unauthenticated on a network
interface. When a key is set, every RPC must carry it as gRPC metadata under `x-zenith-key`
(the same secret the HTTP sidecar checks via the `X-Zenith-Key` header); `zenith-client` sends
it automatically when `ZENITH_KEY` is set in its own environment. **Breaking change:** `--http
:7700` now also binds `127.0.0.1` by default — pass `--bind 0.0.0.0` to reach it from outside
the host, same as the gRPC port.

---

## Metadata filtering

Every document can carry string/number/bool attributes, attached at index time and matched
at search time — before rank fusion, on both the lexical and vector candidate sets, not as a
post-filter on the final page of results.

**Attach attrs when indexing:**
```bash
zenith index --attr tenant=acme --attr year=2024 --attr public=true ~/Documents
```
A value that reads as `true`/`false` is a bool, one that parses as a number is a number,
anything else is a string; quote a value (`"2024"`) to force a string.

**Filter when searching**, three equivalent surfaces:

1. **CLI `--where`** — quick key/value conditions (repeatable; all must hold):
   ```bash
   zenith search --where tenant=acme --where "year>=2020" --where "lang!=fr" "kubernetes oom"
   ```
   Supported operators: `=`, `!=`, `>=`, `<=`.

2. **CLI `--filter` / gRPC `FilterNode` / HTTP `"filter"`** — the full JSON expression, for
   `and`/`or`/`not` combinations `--where` can't express:
   ```json
   {"op":"or","args":[
     {"op":"eq",     "field":"lang",   "value":"en"},
     {"op":"exists", "field":"pinned"}
   ]}
   ```
   Operators: `eq` (equals), `in` (equals any of `values`), `range` (numeric, `min`/`max`
   either optional), `exists` (field present), `and`/`or`/`not` (nest other conditions).
   A document missing the field never matches a condition on it (`not` is the one operator
   that matches documents *without* the field). Expressions are capped at depth 16 / 512
   nodes so a filter arriving over the network can't be used as a denial-of-service vector.
   The same grammar is accepted by:
   - CLI: `zenith search --filter '{"op":"or","args":[...]}'` (combined with any `--where`
     conditions — everything must hold)
   - gRPC: `SearchRequest.filter` (`FilterNode` in `document.proto` — a typed oneof of the
     same operators, not JSON on the wire)
   - HTTP sidecar: `"filter"` in the `POST /v1/ns/{ns}/search` body (raw JSON, same shape)
   - Go library: `zenith.FilterFromJSON([]byte)`, or build one directly with `zenith.Eq`,
     `zenith.In`, `zenith.Range`, `zenith.Exists`, `zenith.And`, `zenith.Or`, `zenith.Not`,
     passed via `zenith.WithFilter(f)`

A filter that only touches a handful of documents (estimated ≤ `max(2000, docs/50)`) is
resolved through a per-field attribute index instead of scanning every document; `not`
always falls back to a scan. Either path returns identical results — this is a property
test, not just documentation.

**Not in scope (deferred):** the attribute index currently stores sorted ordinal postings per
value, not roaring-bitmap-compressed postings. That's a pure efficiency project with no
correctness or feature gap at today's scale, so it stays on the backlog (see ROADMAP.md).

---

## Project Structure

```
ZENITH/
├── cmd/
│   ├── zenith/               # CLI entrypoint — index, search, watch, serve, log
│   └── server/               # standalone gRPC server
├── internal/
│   ├── localembedder/        # ONNX inference — WordPiece tokenizer, model session, pooling
│   │   └── assets/           # model.onnx + onnxruntime lib (go:embed, downloaded via go generate)
│   ├── analysis/             # tokenizer, stemmer, n-grams, phonetic, BK-tree, FST
│   ├── crawler/              # fsnotify file watcher + text extractors
│   ├── pdf/                  # PDF text extraction (pure Go)
│   ├── image/                # image file indexing (filename-based)
│   ├── watchlist/            # persistent watchlist (~/.zenith/watchlist.json)
│   ├── autostart/            # OS boot auto-start (Windows/Linux/macOS)
│   ├── embedding/            # Ollama, deterministic adapters + LRU cache
│   ├── storage/              # WAL, MemTable, SSTable, Bloom filter, compaction
│   ├── index/                # inverted index + vector store + search orchestrator
│   ├── ranking/              # RRF + BM25 tiebreak + TF-IDF
│   └── config/               # all tunable parameters
├── pkg/
│   └── zenith/               # public Go library API (go get github.com/shramanb113/ZENITH/pkg/zenith)
├── gen/go/zenithproto/       # generated Protobuf
└── scripts/                  # go generate asset downloader
```

---

## Configuration

All tunable parameters live in `internal/config/config.go`.

| Parameter | Default | Description |
|---|---|---|
| `FuzzyMaxDist` | `2` | Max fuzzy edit distance (FST Levenshtein automaton); with `FuzzyByLength` it applies to words of 6+ letters |
| `FuzzyByLength` | `true` | ≤3 letters exact, 4–5 letters one edit, longer words `FuzzyMaxDist` |
| `RRFConstant` | `20.0` | RRF k value (tuned on MS MARCO dev; plateau k 10–30) |
| `VectorWeight` | `2.0` | RRF weight of the semantic list (lexical list weight is 1.0) |
| `PhoneticWeight` | `0.3` | Phonetic signal blend weight |
| `NeuralWeight` | `1.0` | Zero-result neural-expansion weight |
| `MaxResults` | `1000` | Internal RRF candidate cap, not a page size |
| `MemTableMaxSize` | `64 MB` | SSTable flush threshold |
| `CommitWindow` | `4 ms` | Group-committer batch window |

---

## Build Roadmap

### Phase 1 — Core Foundation ✓
Inverted index, tokenisation, gRPC service, thread-safety, concurrent indexing, binary persistence.

### Phase 2 — Neural Intelligence ✓
Vector store, dot product, cosine similarity, deterministic embeddings, hybrid RRF ranking.

### Phase 3 — Linguistic Mastery ✓
Porter2 stemming, edge n-grams, phonetic matching, BK-tree fuzzy, synonym expansion, FSTs.

### Phase 4 — Storage Engine ✓
WAL with CRC framing, MemTable skip-list, SSTables, group committer, leveled compaction, Bloom filters, sparse index.

### Phase 5 — Native Embeddings + Library API (current)
- [x] ONNX in-process embeddings — Python sidecar eliminated
- [x] `go:embed` model distribution — no setup step, no network call
- [x] Pure-Go PDF extraction
- [x] `pkg/zenith` library API — `go get` embeddable search
- [x] In-memory mode (`:memory:`)
- [x] v1 → v2 auto-migration on upgrade
- [ ] Benchmark suite (p50/p95/p99 vs Meilisearch/Typesense)

### Phase 6 — Production Observability (planned)
Prometheus metrics, OpenTelemetry traces, `make dev-stack`.

---

## Contributing

ZENITH is built in public. Issues and PRs are open.

Every component has a clear boundary and a reason to exist. See [DECISIONS.md](./DECISIONS.md) for the full architectural and strategic reasoning behind every major choice.

---

## License

MIT

## HTTP sidecar mode

`zenith serve --http :7700` serves ZENITH over HTTP/JSON, so services that are not written in Go
can use its hybrid matching without gRPC code generation. The `/v1/ns/*` routes are a per-request
namespace mode for classify-and-discard workloads: each `PUT` builds a short-lived, in-memory
index, you run a batch of explained queries against it, and it is dropped on `DELETE`, after 10
idle minutes, or by LRU. Nothing is written to disk.

Analysis is Unicode-aware (Devanagari matras, nukta and ZWJ folding), and `--synonyms` can bridge
romanised and native-script forms (e.g. `paani <-> water`). Retrieval quality on human-written
Hinglish/Devanagari text has not been benchmarked yet (ROADMAP Deferred backlog item J).

| Route | Purpose |
|---|---|
| `GET /healthz` | version, embedding model id (e.g. `gte-small`, `deterministic` or `none`), synonyms hash |
| `PUT /v1/ns/{ns}/docs` | build (or replace) a namespace from `{"docs":[{"id","text"}]}` |
| `POST /v1/ns/{ns}/search` | many queries at once with `explain`: raw BM25, cosine, and per-term exact / synonym / fuzzy hits with character spans; optional `"filter"` restricts to matching attrs — see "Metadata filtering" |
| `DELETE /v1/ns/{ns}` | drop a namespace (idle namespaces also expire after 10 min; max 200, LRU) |

Flags:
- `--key` (or `ZENITH_KEY`) requires `X-Zenith-Key` on `/v1/*`.
- `--synonyms file` adds query-time bridges such as `paani <-> water`.

Request bodies are capped at 1 MB, and the sidecar never logs document or query text. Explain signals are
absolute. The fused `Score` is relative to the result set, so never threshold on it.

```bash
docker build -t zenith-sidecar .
docker run --rm -p 127.0.0.1:7700:7700 -e ZENITH_KEY=change-me zenith-sidecar
```

## Docker demo (CLI + gRPC, persistent)

The single-container snippet above runs only the in-memory HTTP sidecar (no persistence by
design — see `internal/sidecar`). `docker-compose.yml` instead builds an image with OCR support
(`-tags ocr`, Tesseract) and a `zenith-client` test-suite binary, backed by a named volume so the
index survives container restarts:

```bash
./demo/run.sh
```

walks through, against the real persistent engine: mixed-format ingestion (`.txt`/`.md`/`.csv`/
`.json`/`.log`/images) with per-file metadata attrs, typo tolerance, out-of-domain semantic
search, metadata-filtered search, an OCR'd image hit, cross-encoder reranking (before/after),
the `labse` multilingual model against a separate index, and a SIGKILL of the running server
followed by a check that the index files it had open still return byte-identical results and pass
every segment checksum. Demo documents live in `demo/corpus/`.

**What section 8 does and does not show.** It shows that segments already committed to disk
survive a SIGKILL of a process holding them open: segments are immutable and fsynced before the
manifest points at them. It does *not* show that writes sent to `zenith serve` over gRPC survive a
SIGKILL — that server keeps new writes in memory and saves them only on graceful shutdown.
Write-ahead-logged durability, where every acknowledged write survives a kill, is a property of
the Go library (`pkg/zenith`), exercised by hard-kill loops and simulated power-loss sweeps
(ROADMAP risk 6).

**Validated end-to-end, 2026-10-03** (`docker compose build` + all 8 scripted sections, Docker
Desktop on Windows): typo tolerance and the out-of-domain "heart attack symptoms" → cardiac-notes
query both land correctly; the OCR'd PNG surfaces for a query whose text exists only inside the
scanned image; `--rerank` sharply separates the one relevant document (score 0.998) from the rest
(0.000); the `labse` index's "capital of France" query ranks `french.txt` first ahead of the other
four languages; and the original section 8 (the `zenith-client` diagnostic before and after a
SIGKILL) showed identical results — **but that was not evidence of crash recovery**: the
diagnostic re-indexes its own five documents on every run, so the post-kill run rebuilt exactly
what was being compared. Section 8 was rewritten to compare the saved index directly (above).
On Windows, running it from Git Bash/MSYS requires `MSYS_NO_PATHCONV=1` (already set inside
`demo/run.sh`) so paths like `/home/zenith/corpus/...` reach the container unmangled.

**Real-world Hindi accuracy check, 2026-10-03** (not part of `demo/run.sh`, run ad hoc in the
same container): three live Hindi Wikipedia articles (Taj Mahal, Cricket, Himalaya) were fetched
and indexed alongside the existing one-sentence `hindi.txt`, then queried with English prompts
that share no vocabulary with the Devanagari text — so only the `labse` vector pass can match,
not the lexical/BM25 pass that quietly helps the "capital of France" query above (it shares the
literal word "France"). Initial result: **1 of 4 queries ranked the right document first (25%),
3 of 4 landed it in the top 3 (75%)**; the cricket query never surfaced `cricket.txt` in the top
3 at all.

Root-caused and partially fixed the same day: `labse`'s ONNX export only exposes raw
`last_hidden_state` with no pooling head, so this was initially CLS-pooled directly — a raw
`[CLS]` vector without its trained projection is known to be anisotropic (near-identical cosine
similarity regardless of topic, which matched the 0.08–0.14 scores observed here). Rather than
guess at a pooling strategy, `sentence-transformers/LaBSE`'s own module config was checked
directly: `1_Pooling/config.json` confirms CLS pooling is correct, and the actual missing piece
is `2_Dense` — a small (2.4 MB) trained `Linear(768,768)+Tanh` projection saved separately from
the base model. `Spec.DenseURL` (`internal/modelspec`) now fetches it via `zenith models pull
labse` and `internal/localembedder` applies it after CLS pooling, reconstructing the real
sentence-transformers pipeline instead of approximating it. This closed the anisotropy problem
for same-language retrieval: `TestMultilingualSmoke`'s correct-passage similarity scores went
from compressed near-threshold values to a properly spread 0.32–0.72 across all five languages.

It did **not**, however, change the 4-query English→Hindi zero-shared-vocabulary benchmark
above — mean pooling and the correct trained-projection reconstruction both reproduced the exact
same 1/4 top-1, 3/4 top-3 result as the original raw-CLS baseline, with cricket.txt still never
reaching the top 3. Three different, individually well-justified pooling/projection strategies
converging on identical aggregate accuracy is itself informative: it indicates the ceiling here
is the frozen, int8-quantized embedding model's cross-lingual alignment quality on a small
(8-document), deliberately adversarial corpus with several unrelated-language distractors — not
an implementation bug. See [ROADMAP.md](./ROADMAP.md) item 6 for the full breakdown. Honest
takeaway: `labse` is solid for same-language/curated retrieval (now backed by the real trained
projection, not a workaround) and for single-document demo queries, but arbitrary zero-shared-
vocabulary cross-lingual retrieval at this corpus scale should not be oversold.

