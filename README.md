# ZENITH

[![CI](https://github.com/shramanb113/ZENITH/actions/workflows/ci.yml/badge.svg)](https://github.com/shramanb113/ZENITH/actions/workflows/ci.yml)

> Local-first hybrid search — as a CLI tool and as a Go library.

ZENITH is a search engine built from scratch in Go. It understands meaning, not just keywords — combining lexical matching, BK-tree fuzzy search, and neural vector embeddings into a single hybrid pipeline, ranked with Reciprocal Rank Fusion.

You can use it two ways:

- **As a CLI tool** — install once, point at directories, search from your terminal
- **As a Go library** — `go get` it, call three methods, ship search inside your app

The storage layer is a full LSM-tree (WAL → MemTable → SSTable → Bloom filters), built from scratch. The embedding model (`all-MiniLM-L6-v2`, int8 ONNX) is baked into the binary — no Python, no server, no setup step.

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
| C compiler | gcc / MinGW-w64 | Required for ONNX inference (CGo). See note below. |
| Ollama | any | Optional — alternative embedder |

**C compiler note:** ZENITH uses CGo to run `all-MiniLM-L6-v2` in-process via ONNX Runtime. On Windows, install MinGW-w64:

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

ZENITH ships the `all-MiniLM-L6-v2` embedding model (int8 ONNX, ~22 MB) **baked into the binary** via `go:embed`. Inference runs in-process via `yalue/onnxruntime_go` — no Python, no network hop, no separate process.

On every query and document add, ZENITH runs this cascade automatically:

```
1. Local ONNX model (embedded, always available if built with CGo)
2. Ollama at localhost:11434       (if --embedder ollama)
3. Deterministic hash embeddings   (fallback — zero dependencies)
```

Cold start with the embedded model: **under 1 second.** The ONNX session is initialised once at startup and held for the process lifetime.

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
zenith serve            # start gRPC server (port 8080)
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
│  dir walk   │     │  camelCase-aware     │     │  1. ONNX  all-MiniLM-L6-v2  │
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
| ONNX embedder (embedded, in-process) | Active | all-MiniLM-L6-v2, int8, CGo |
| Deterministic embedder | Active | Fallback — zero dependencies |
| Ollama embedder | Active | Needs Ollama running |
| gRPC server (`zenith serve`) | Active | Port 8080 default |
| fsnotify incremental watching | Active | |
| Persistent watchlist | Active | `~/.zenith/watchlist.json` |
| Boot auto-start (Windows/Linux/macOS) | Active | |
| Text, Markdown, Go, HTML extractors | Active | |
| PDF extraction | Active | Pure Go via `ledongthuc/pdf` |
| Image indexing (filename-based) | Active | |
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

**Fuzzy:** BK-tree over Levenshtein distance, O(log n) via triangle inequality pruning. Tolerates up to 2 edits by default (configurable in `internal/config/config.go`).

**Semantic:** `all-MiniLM-L6-v2` (384-dim) via ONNX Runtime, running in-process. Vector cosine similarity stored as float16 to halve memory. Embedding failures are non-fatal — engine degrades to lexical-only.

**Ranking:** Reciprocal Rank Fusion (RRF, k=60) merges all result lists. BM25 tiebreaker when RRF scores are within epsilon. All weights tunable in `internal/config/config.go`.

---

## File Support

| Format | Extraction |
|---|---|
| `.txt` `.md` `.log` `.csv` `.json` `.yaml` | Raw UTF-8 text |
| `.go` | AST — identifiers, comments, package name |
| `.py` `.ts` `.js` `.jsx` `.tsx` `.rs` `.java` `.c` `.cpp` | Raw source |
| `.html` `.htm` | Tag-stripped visible text |
| `.pdf` | Pure-Go text extraction via `ledongthuc/pdf` |
| `.jpg` `.jpeg` `.png` `.gif` `.bmp` `.webp` `.tiff` | Filename + directory path tokens |

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

`Search` fuses lexical (BM25 + n-gram + phonetic + BK-tree fuzzy) and vector scoring via RRF — there's no separate `FuzzySearch`/`HybridSearch` RPC, it's all one `Search` call. `SearchRequest` takes an optional `limit`/`offset` for pagination (default limit: 10). See `internal/proto/document.proto` for the source of truth.

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
| `FuzzyMaxDist` | `2` | BK-tree edit distance threshold |
| `RRFConstant` | `60.0` | RRF k value |
| `PhoneticWeight` | `0.3` | Phonetic signal blend weight |
| `VectorWeight` | `0.7` | Vector signal blend weight |
| `NeuralWeight` | `1.0` | Neural signal blend weight |
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

`zenith serve --http :7700` serves short-lived, in-memory namespaces over HTTP/JSON. Services that
are not written in Go can use ZENITH's hybrid matching without gRPC code generation.

| Route | Purpose |
|---|---|
| `GET /healthz` | version, embedding model id (`all-MiniLM-L6-v2`, `deterministic` or `none`), synonyms hash |
| `PUT /v1/ns/{ns}/docs` | build (or replace) a namespace from `{"docs":[{"id","text"}]}` |
| `POST /v1/ns/{ns}/search` | many queries at once with `explain`: raw BM25, cosine, and per-term exact / synonym / fuzzy hits with character spans |
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

## Used in Kshetra IQ

[Kshetra IQ](https://github.com/shramanb113/Flat-finder) (SerpApi India Hackathon 2026) uses this sidecar to decide which
issue a noisy review or news snippet is about. That includes misspellings, Hinglish ("paani bhar jaata hai") and
Devanagari ("पानी भर जाता है"). SerpApi does all the fetching, ZENITH only matches, and a deterministic rule engine
owns the verdict. These additions were built during the hackathon and are disclosed in the submission:
- Unicode-aware analysis (Devanagari matras, nukta and ZWJ folding)
- Explain mode
- `WithoutWordVectors`
- the HTTP sidecar
- the synonyms file
- this Dockerfile
