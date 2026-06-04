# Why ZENITH is Built the Way It Is

This document records every significant decision made in ZENITH — what I chose, what I rejected, and why. It exists because architectural choices without recorded reasoning are dead weight: six months later you have code you can't change safely because nobody remembers what constraint it was solving.

If you hit a limitation in ZENITH and want to understand why it exists, this is where to look. If you're thinking of changing something fundamental, start here. If you're evaluating ZENITH as a library and want to understand what you're depending on before you add it to `go.mod`, this document has what you need.

---

## Why this project exists

Most Go developers building web services hit the same moment at some point. They need search. They reach for Elasticsearch, look at the ops overhead, flinch, and either skip search entirely or set up a hosted Algolia account and pay per query forever.

That moment is what ZENITH is trying to solve. Not by being another search server, but by making search something you can just import, the same way you import a database driver. Call three methods. Ship it in your binary. No infrastructure to operate.

That sounds simple but getting there required solving a lot of non-obvious problems. The architecture decisions that follow are the result of working through those problems one by one. None of them are obvious in hindsight, which is exactly why they're written down.

---

## What ZENITH actually is

The tempting way to describe ZENITH is "a fast search engine built in Go." That framing is a trap. It puts ZENITH in direct competition with Meilisearch (fast, great developer experience, built by a funded team in Rust), Typesense (fast, cloud-native, C++), Quickwit (log search, Rust), and Elasticsearch (the enterprise incumbent with a decade of production deployments behind it). There is no version of that fight where ZENITH wins. They have teams, they have benchmarks, they have production users. Competing on raw throughput or feature count against organizations with dedicated resources is just a slow way to lose.

But those engines all share a fundamental property: they are server processes. You run them separately from your application. You talk to them over HTTP. You manage their containers, their upgrades, their disk space, their memory. Every query your application makes crosses a network boundary. They are search infrastructure.

ZENITH is something different. It runs inside your Go application, in the same process, with no network boundary. You import it like a library. You call `zenith.Open()`. It works. There is no Docker container to keep alive, no sidecar process to monitor, no ops overhead at all.

The analogy that captures this precisely is SQLite. SQLite didn't beat PostgreSQL or MySQL on throughput. It didn't out-feature them. It became the most widely deployed database in the world — running in every iPhone, every Android device, every browser, every desktop app — by occupying a different category entirely. When you need a database inside your application, not next to it, SQLite is what you reach for. PostgreSQL doesn't compete there. The categories don't overlap.

ZENITH is claiming that same category for search. The one-line version is: ZENITH is to search what SQLite is to databases. Zero dependencies, embeds in your app, ships in your binary.

The target user is a Go backend developer building an HTTP or gRPC service who wants full-text and semantic search over their own data — blog posts, product catalogs, documentation, notes, anything — without standing up a separate search service or paying for a hosted API. They want to write:

```go
db, _ := zenith.Open("search.db")
db.Add(ctx, "doc1", text)
results, _ := db.Search(ctx, "query")
```

And have it just work. No setup ceremony. No infrastructure to manage.

Go web services are the primary use case because that's where the need is most acute and most common. Almost every Go backend developer eventually hits "I need search" and reaches for Elasticsearch, then grimaces at the operational cost. That moment is the gap ZENITH fills. The demo that resonates is: "I added hybrid semantic search to my Go app in five lines, no Docker." That framing is immediately understood by the audience it's intended for.

The secondary position is that nobody else owns this category yet. Meilisearch, Typesense, and Elasticsearch are all server processes. There is no established "embeddable search for Go" library the way there is an established embeddable database. ZENITH is trying to be the first real answer to that gap.

---

## The Python sidecar that had to go

The earliest version of ZENITH used a Python process called `nerve` for ML inference and PDF extraction. The pitch at the time was "pure Go, single binary, drop it anywhere." The reality was that before the binary would do anything useful, you needed Python 3.10+, a pip or uv installation, a virtualenv, and about 600 MB of packages. That setup step could take up to 20 minutes on a cold machine.

This is what a credibility gap looks like. The code directly contradicts what the README promises. Any engineer who reads "single binary, zero dependencies" and then opens the repo to find a `requirements.txt` immediately discounts the entire project. It doesn't matter how good the rest of it is. The first thing they see disproves the claim.

The operational problems compounded it. Three things were wrong simultaneously:

First, two runtimes had to stay alive at the same time. The Go process and the `nerve` Python process both had to be running. If `nerve` crashed for any reason, ZENITH would either degrade silently or fail entirely, and the error would be confusing.

Second, every embedding call crossed a gRPC boundary. That's 5-15ms of IPC latency per call, paid for no reason except that inference was running in a different process. The network hop existed purely as an artifact of the architecture, not because the task required it.

Third, deployment was genuinely painful. Running ZENITH meant managing two runtimes, two sets of dependencies, and a sidecar process in addition to your actual application. The "drop it in and it works" promise died at the first `zenith setup` command.

The fix was moving everything in-process. Embeddings now run via an ONNX model loaded with `yalue/onnxruntime_go`. PDF extraction uses `ledongthuc/pdf`, which is pure Go. Cold start dropped from 30-90 seconds to under a second. The credibility gap closed.

---

## Technical decisions

### 1. No Python sidecar — everything in-process

The embedding model is `all-MiniLM-L6-v2`, the int8 quantized ONNX export (~22 MB). It runs in-process via the `yalue/onnxruntime_go` CGo binding. PDF text extraction uses `ledongthuc/pdf`. Image indexing decomposes file paths into tokens. There is no Python interpreter involved anywhere.

No Python interpreter. No virtualenv. No IPC. Cold start under one second.

### 2. ONNX Runtime over llama.cpp

`all-MiniLM-L6-v2` is a BERT-style encoder-only model. It takes text in and produces a fixed-size 384-dimensional embedding vector in a single forward pass. ONNX Runtime is purpose-built for exactly this kind of workload — sub-millisecond inference on CPU, the int8 quantized export exists on HuggingFace, and the total model size is about 22 MB.

llama.cpp is designed for autoregressive decoder models — the kind that generate text token by token. Using it for an encoder-only embedding model is the wrong tool. The CGo build is heavier. The GGUF format adds about 45 MB vs 22 MB for the ONNX int8 export. Inference is slower for this architecture. And the GGUF conversion pipeline is itself Python-based, which brings back the exact dependency we were trying to eliminate.

If ZENITH ever adds native image captioning, llama.cpp with a small GGUF vision model (LLaVA-style) would be the right call for that. It's an additive change that doesn't require revisiting this decision.

Both approaches use the same distribution strategy: `go generate` downloads the model from HuggingFace, the file is in `.gitignore`, and CI runs `go generate` before `go build`. Model size isn't a differentiator between the two.

### 3. go:embed over download-on-first-run

The ONNX model (~22 MB) is embedded directly in the binary via `go:embed`. It's not downloaded on first run.

A download-on-first-run approach keeps the binary smaller but breaks the embeddable library pitch entirely. If you import ZENITH as a library in your application, `zenith.Open()` has to just work. There cannot be a "first run setup ceremony" where the library phones home to download a model file. A library that requires network access before it functions is not embeddable in the SQLite sense.

22 MB is the right tradeoff for zero-setup semantics. SQLite's amalgamation source is 237,000 lines of C — nobody complains about it. The concern with embedded resources is always self-containedness, not size.

### 4. CGo is acceptable; Python is not

CGo means the binary requires a C runtime and must be compiled for its target platform. That's the same constraint as `mattn/go-sqlite3`. The binary is still a single compiled artifact. Library consumers don't manage a separate runtime.

Python as a runtime dependency is a different category of problem. Users need an interpreter, a package manager, a virtualenv, and hundreds of megabytes of packages before the software works at all. That's not a dependency; that's a second software stack.

On Windows, CGo requires MinGW-w64 via MSYS2:

```
winget install -e --id MSYS2.MSYS2
# then in the MSYS2 terminal:
pacman -S --noconfirm mingw-w64-ucrt-x86_64-gcc
```

Add `C:\msys64\ucrt64\bin` to PATH and `CGO_ENABLED=1 go build ./...` works natively.

There's also a `model_nocgo.go` stub with the `!cgo` build tag. When CGo is unavailable, `localembedder.New()` returns an error and the engine falls back to deterministic hash-based embeddings. Lexical search still works; only neural semantic search is degraded. This means `go install` without MinGW still produces a usable binary.

### 5. Pure-Go WordPiece tokenizer over daulet/tokenizers

The alternative was `daulet/tokenizers`, a CGo binding to HuggingFace's Rust tokenizer library. It gives byte-for-byte tokenization parity with Python SentenceTransformers, but it adds a second CGo dependency, a Rust toolchain requirement at build time, and significantly more static linking complexity.

WordPiece for `all-MiniLM-L6-v2` is not complicated: lowercase, split on punctuation, look up in the BERT vocabulary, fall back to `[UNK]`. For English and Latin-script text the pure-Go implementation produces identical output to HuggingFace. Edge cases exist for CJK and some Unicode combining sequences, but those don't materially affect search quality for the expected user base.

If exact Python parity ever becomes a requirement, swapping `internal/localembedder/tokenizer.go` for a `daulet/tokenizers`-backed implementation is a one-file change. The interface is internal and stable.

### 6. ledongthuc/pdf over PyMuPDF

PyMuPDF (`fitz`) is a battle-hardened C library that handles malformed and exotic PDFs that few other tools can process. But it's accessed through Python, which brings back the entire sidecar problem that was already eliminated.

`ledongthuc/pdf` is pure Go. A small fraction of PDFs that `fitz` handles silently will produce a clear error here instead. That's an acceptable tradeoff given the constraints.

If PDF handling quality becomes a real issue, a CGo binding to `mupdf` (the C library behind PyMuPDF) can slot in at `internal/pdf/` without touching anything outside that package. The interface is stable.

### 7. Filename-based image indexing over BLIP

BLIP image captioning (`Salesforce/blip-image-captioning-base`) was in ZENITH briefly but was off by default, requiring `ZENITH_BLIP_CAPTIONS=1` to activate. Getting it to run in Go without Python requires either a complex ONNX export of the full encoder-decoder architecture or a GGUF vision model via llama.cpp. Neither was worth the complexity for a feature nobody was turning on.

The replacement is path decomposition: `~/Photos/2024/vacation/portrait_sunset_beach.jpg` becomes `"portrait sunset beach vacation 2024 photo jpeg"`. This is the same strategy macOS Spotlight and Windows Search use. It works well for most real-world image collections.

If proper image understanding becomes worth the complexity, a small GGUF vision model via llama.cpp Go bindings is the right path. It's additive — only `internal/image/indexer.go` changes.

---

## Library API design

### 8. database/sql style with functional options

Three API shapes were considered seriously.

Struct-based only (`db.Add(ctx, id, text)`) is clean on day one. But adding any new option later — a second embedder, custom BM25 weights, a memory limit — requires a breaking change. You'd either need a new function or modify the struct, and both break callers.

Functional options only (`zenith.Open("search.db", zenith.WithEmbedder(...))`) is extensible but verbose on line one. For the default case, which is the majority of users, the line becomes cluttered with options they don't need. It contradicts the "zero friction" pitch.

Generics (`zenith.New[Article](...)`) turns it into an ORM. The SQLite analogy breaks immediately the moment the library requires type parameters. It also adds Go 1.18+ familiarity as a prerequisite for understanding the API.

The right answer is both A and B together: a dead-simple default path and functional options for power users who want to tune behavior. This is the pattern used by `bbolt`, `badger`, `zap`, and `grpc-go`. It's the de facto standard for production Go libraries for a reason.

```go
// default path — everything works out of the box
db, err := zenith.Open("search.db")   // like sql.Open
db.Add(ctx, "id", "text content")     // like db.Exec
results, _ := db.Search(ctx, "query") // like db.Query
db.Delete(ctx, "id")
db.Close()
```

Same surface area as `database/sql`. Anyone who has written Go for a week understands this shape immediately.

```go
// power user path — no breaking changes
db, err := zenith.Open("search.db",
    zenith.WithEmbedder(myEmbedder),
    zenith.WithBM25Only(),
)
```

The pattern was popularized by Dave Cheney's 2014 post on functional options. Reference implementations: `bbolt` and `badger` use struct API with options on `Open`. `zap` uses `zap.New(core, zap.Option...)`. `grpc-go` uses `grpc.Dial(addr, grpc.WithTransportCredentials(...))`. `database/sql` uses `sql.Open(driver, dsn)`. There's no reason to invent something different.

### 9. Persistent by default, in-memory for tests

`zenith.Open("search.db")` persists to disk. `zenith.Open(":memory:")` doesn't. Same function, one argument, mirrors SQLite's behavior exactly.

The in-memory mode is not a "lightweight" alternative to disk — it's a testing feature. Every engineering team that operates at any serious scale tests with in-memory databases because it makes tests fast, isolated, and parallel-safe with no cleanup. A library that can't be tested in-memory doesn't get adopted by serious teams. The ability to write `zenith.Open(":memory:")` in a test is a major adoption signal.

Persistent mode is for production: data survives pod restarts in Kubernetes, process crashes, and deploys. In-memory is for tests: fast, isolated, no cleanup needed.

No magic paths. No library-managed directories. The caller decides where the data lives. In a containerized deployment where you control exactly which volumes are mounted, this is the pattern that works.

### 10. Two products, one engine, one repository

ZENITH ships as both a CLI tool (`cmd/zenith`) and a Go library (`pkg/zenith`), both built on top of the same internal engine at `internal/index/engine.go`.

The CLI is a standalone program. `go install` puts it in `$GOPATH/bin`. You run it from the terminal. It's a tool.

The library is a dependency. `go get` puts it in `go.mod`. It runs inside your application in the same process. No separate binary, no `zenith` command, no setup. Just a function call.

```go
import "github.com/shramanb113/ZENITH/pkg/zenith"

db, _ := zenith.Open("search.db")
db.Add(ctx, "id", text)
results, _ := db.Search(ctx, "query")
```

These are two different products sharing one engine. The CLI is ZENITH the tool. The library is ZENITH the platform — the thing you embed in your application. Same search logic, same embeddings, same storage engine, two entry points.

`pkg/zenith/` is a thin public facade over `internal/index/engine.go`. The facade translates the five-method public API into engine operations. Internal types (`BKTree`, `VectorStore`, `InvertedIndex`) never appear in any public signature. This is the same pattern `database/sql` uses: a clean `DB` type exposed to users while every driver implementation lives behind an interface. Users of `database/sql` have never seen a `btree.Node`. That's the goal here.

### 11. Why "SQLite of search" is the right wedge

Meilisearch, Typesense, Quickwit, and Elasticsearch are all server processes. You run them separately and talk to them over HTTP. That means ops overhead to deploy and maintain, a network hop away from your code on every query, and an external dependency your application cannot ship without.

ZENITH is a library. It compiles into your binary and runs in your process. That's a completely different product category. You're not trying to be faster than Meilisearch. You're trying to be to search what SQLite is to databases: the thing you reach for when you want search inside your application, not next to it.

SQLite didn't win by out-featuring PostgreSQL. It won by being embeddable, zero-config, and single-file — a category nobody else owned. It now runs in every iPhone, every Android device, every browser. ZENITH is claiming that same unclaimed category for search. `go get` and you have production-quality hybrid search in your application with no infrastructure to manage and nothing to deploy.

---

## Library engineering: edge cases, failure modes, and breaking criteria

These are the things that will surface in production and come up in technical interviews. Every one of them has been thought through. The fixes are in the code. This is the reasoning behind them.

### FNV-32 hash collision

The early engine assigned each document an internal ID by hashing its string ID with a 32-bit FNV-32a hash:

```go
h := fnv.New32a()
h.Write([]byte(originalID))
internalID := uint32(h.Sum32()) // 2^32 possible values
```

The birthday problem makes this dangerous at scale. With 2^32 possible values, you have a 1% chance of at least one collision at just 92,000 documents. A collision means document B silently overwrites document A. A's content disappears from search results with no error, no warning, and no way to detect it from outside the engine.

The fix is FNV-64a. The internal ID space is now 2^64 values. At any practical corpus size the collision probability is negligible.

### Gob file corruption on crash

The engine saves its entire state to a single gob file. If the process is killed mid-write — OOM kill, power loss, `kill -9` — the result is a partially-written file: valid gob header, truncated body. The next `Open` fails with a confusing decode error. Users think they lost their index.

The fix is writing to `zenith.db.tmp` and then atomically renaming to `zenith.db`. `os.Rename` is atomic at the filesystem level on all major operating systems. The reader always sees either the old complete file or the new complete file, never a partial write.

### No format version header

The original gob file had no version marker. When an internal struct changed after an upgrade, `Open` failed with `gob: type mismatch in decoder: want struct type ...` — a message that means nothing to anyone.

Now every saved file starts with a 4-byte magic (`ZNTH`) and a 2-byte version number. On `Open`, the magic and version are read before any gob decoding begins. An incompatible version returns `ErrIncompatibleVersion` with a human-readable message instead of a cryptic decode failure.

### BM25 statistics not serialized

BM25 scoring depends on corpus-level statistics that live only in memory: total document count, total corpus length, per-term document frequency. If these aren't saved, every `Open` starts BM25 from zero. All previously indexed documents have no term frequency records and produce meaningless scores until every document is re-indexed from scratch.

The fix is explicitly serializing BM25 scorer state as part of `Save` and restoring it in `Load`. There's a round-trip test: save, load, verify that BM25 scores are identical before and after.

### Internal chunk IDs leaking out

PDF indexing creates internal chunk IDs in the format `"docID||p3||c1||text||10.00,20.00,400.00,15.00"`. Without the library facade layer, these raw internal IDs surface to the caller who passed `"report.pdf"` and expected `"report.pdf"` back. They have to parse `||`-separated implementation details to recover their original document ID.

The library layer strips chunk suffixes before returning results. The caller always gets their original document ID. Page and position information is returned in a separate `Chunk` struct on the `Result` type for callers who need it for PDF highlighting.

### Empty and whitespace-only inputs

```go
db.Add(ctx, "", "content")      // ErrInvalidID
db.Add(ctx, "id", "")           // ErrEmptyDocument
db.Add(ctx, "id", "   \t\n ")   // ErrEmptyDocument — whitespace-only counts as empty
```

Silently accepting empty documents pollutes the index with zero-vector entries that can never be meaningfully retrieved. Named sentinel errors let callers handle these cases with `errors.Is`.

### String length bounds

IDs are stored in a map that lives in process memory. Unbounded ID lengths are a memory attack surface. The cap is 512 bytes. Beyond that, `ErrIDTooLong`.

Document text has no length limit for lexical indexing — the full text goes into the inverted index regardless of length. But the embedding tokenizer truncates at 256 tokens, which is roughly 1,000 characters. Semantic search only covers that portion of the document. This is silent data loss for semantic search on long documents, and the godoc says so explicitly: semantic search covers only the first ~1,000 characters, lexical search covers the full document.

### Returning nil vs an empty slice

`Search` with no matches returns `[]Result{}`, never `nil`. Returning `nil` breaks JSON serialization — `json.Marshal(nil)` produces `null`, not `[]`. It also surprises callers who check `len(results) == 0`. Every slice return value from the public API is allocated and empty, not nil.

`AddBatch` with a nil or empty map is a no-op, no error.

### Invalid UTF-8 and null bytes

IDs must be valid UTF-8 with no null bytes and no control characters (0x00-0x1F). These corrupt log output, break JSON serialization of results, and cause subtle downstream bugs in code that processes the IDs. They're rejected at input with `ErrInvalidID`.

Document text is different. Real-world content from web scraping, OCR output, and legacy encodings frequently contains garbage bytes. Rejecting it would break too many legitimate use cases. The library sanitizes text silently via `strings.ToValidUTF8(text, "")` before indexing.

### ID characters colliding with the internal separator

The chunk ID format uses `||` as a separator. A user ID containing `||` silently corrupts the format — the library's chunk ID stripping logic would misparse it. The fix is validating at input and returning `ErrInvalidID` with an explanation. Leaky internal abstractions have to be exposed as explicit errors, not silent corruption.

### Option values out of range

```go
zenith.WithLimit(-5)          // ErrInvalidOption
zenith.WithFuzzyDistance(-1)  // ErrInvalidOption
zenith.WithFuzzyDistance(100) // clamped to 5 with a log warning, not an error
zenith.WithCacheSize(0)       // valid — disables the cache
```

Negative values for limits are always errors. `FuzzyDistance` above 5 is clamped rather than rejected because high values cause O(n) BK-tree scans that would hang the caller, and clamping silently is preferable to letting the caller set a value that degrades performance invisibly.

### Score normalization

Raw RRF scores are reciprocal sums — they're not bounded, not intuitive, and not comparable across different result sets. The library normalizes all scores to `[0.0, 1.0]` by dividing by the maximum score in each result set.

NaN and Inf scores (which can come from zero-magnitude vectors in dot product calculations) are clamped to 0.0. The `Result.Score` field is guaranteed to always be a valid float64 in range.

```go
type Result struct {
    ID    string
    Score float64 // always in [0.0, 1.0], never NaN, never Inf
}
```

### Duplicate IDs in results

The hybrid pipeline (lexical + fuzzy + vector) can reach the same document from multiple passes. RRF accumulates scores across passes by design. But a bug could produce duplicate IDs in the final slice. Before returning, results are deduplicated by ID. The caller never sees the same ID twice in one result set.

### Non-deterministic ordering for equal scores

When two results have identical normalized scores, the ordering is determined by ID lexicographically as a tiebreaker. Without this, any test that checks result ordering is flaky, and production result ordering shifts unpredictably between deployments for queries where multiple documents score identically.

### Error messages exposing internals

Raw errors from the engine contain file paths and internal type names that mean nothing to users:

```
// bad: "index: fst build: open /home/user/.zenith/data/terms.fst: permission denied"
// good: "zenith: failed to update search index"
```

All errors returned through the public API are wrapped with `fmt.Errorf("zenith: %w", err)`. Still unwrappable via `errors.Is` and `errors.As`, but with a consistent prefix that identifies the source and keeps internal paths out of user-visible messages.

### Use-after-close

Every method checks an atomic `closed` flag before doing any work:

```go
func (db *DB) Add(ctx context.Context, id, text string) error {
    if db.closed.Load() {
        return ErrClosed
    }
    // ...
}
```

Calling any method after `Close` returns `ErrClosed`. It never panics. The flag is atomic so the check itself is safe from any goroutine.

### Double-close safety

`Close` uses `sync.Once` internally. Calling it twice returns `nil` on the second call. `defer db.Close()` is always correct even if the caller also closes explicitly on an error path.

### Close racing with Add or Search

`Close` sets `closed = true` atomically before acquiring the write lock. If an `Add` is in progress when `Close` is called, `Add` completes first. `Close` then acquires the write lock, saves the index to disk, and tears down. Documents are either fully indexed or not. There's no partial state. Subsequent `Add` calls return `ErrClosed`.

### Panic recovery

If the internal engine panics (this is a bug — it shouldn't happen, but can), a `recover()` wrapper in each public method catches it, marks the DB as permanently closed, and returns a structured error. A panicking `DB` becomes `ErrClosed` rather than corrupting the goroutine stack of whatever called into it.

### ONNX session pool exhaustion

The ONNX model runs in a pool of sessions, capped at `runtime.NumCPU()`. If all sessions are busy, goroutines block waiting for one to free up. The internal context is passed to this wait. If the context is cancelled — including by `Close` — the waiting goroutine unblocks and returns without hanging forever.

### Nil-safe methods

```go
func (db *DB) Add(ctx context.Context, id, text string) error {
    if db == nil {
        return errors.New("zenith: Add called on nil DB")
    }
    // ...
}
```

Every exported method handles a nil receiver with a clean error. No nil pointer panics from caller mistakes.

### Non-determinism in AddBatch

`AddBatch` accepts `map[string]string`. Go's map iteration order is deliberately randomized. Two identical `AddBatch` calls produce documents indexed in different orders, which produces different FST structures and different fuzzy match behavior across runs. The fix is sorting document IDs before processing. Same input always produces the same index.

### Goroutine leak from warmup

The engine starts a goroutine on open to warm the ONNX model with a dummy input, which prevents the first real query from paying the JIT compilation cost. If `Close` is called before warmup finishes — common in short-lived tests — that goroutine outlives the `DB`, holds a reference to the ONNX session, and prevents garbage collection. Over many test runs this accumulates into a real memory leak. The warmup goroutine uses the DB's internal context, so `Close` cancels it immediately.

### WAL overhead for library use

The storage engine's WAL was designed for a long-running server that needs crash recovery across restarts. A library used in short-lived web request handlers pays WAL write overhead on every `Add` with no benefit — if a request handler crashes mid-index, that request fails and nothing else is at risk. A `WithNoWAL()` option lets users skip it. In-memory mode disables WAL automatically.

### Memory scaling in :memory: mode

```
1,000,000 documents × 384 dimensions × 2 bytes (float16) = 768 MB for vectors alone
```

Plus postings lists, BK-tree nodes, and ID mappings. Nothing is evicted — everything stays in RAM. GC pressure grows linearly with index size, causing latency spikes in web handlers. Users who run `:memory:` in production with a large index will hit OOM kills. The godoc documents this formula explicitly. A `WithMemoryLimit(bytes int64)` option returns `ErrIndexFull` when the configured limit is exceeded rather than letting the process grow until the OS kills it.

### Two :memory: opens are not the same database

```go
db1, _ := zenith.Open(":memory:")
db2, _ := zenith.Open(":memory:")
// db1 and db2 share NO state — completely independent databases
```

Engineers coming from Redis or SQLite shared-memory mode expect `:memory:` to be a shared in-process singleton. It isn't. This is the first thing the `:memory:` godoc says.

### File locking

Two processes calling `Open("search.db")` simultaneously would both read the file, both make modifications in memory, and both write back — silently corrupting each other's work. The library acquires an exclusive advisory lock at `path + ".lock"` during `Open`. A second caller gets `ErrLocked` immediately. Two `*DB` instances in the same process pointing at the same path are detected via an in-process `sync.Map` registry and also return `ErrLocked`. Stale lock files from crashed processes are detected by reading the stored PID from the lock file and checking whether that process is still alive.

### Silent quality degradation from embedding failures

Embedding is non-fatal by design. If the ONNX model fails for a specific document, that document is indexed lexically but not semantically. It shows up in keyword and fuzzy results but never in pure semantic search. If the embedder fails intermittently and then recovers, the index ends up with a mixed population of vectored and vectorless documents that behave differently. `SemanticScore float64` on `Result` (0.0 when no vector exists) lets callers detect and handle this split.

### Float16 magnitude cache desync

Vectors are stored as float16 to halve memory usage. Magnitudes are cached separately as float32 for use in dot product normalization. When a document is re-indexed, both the vector and its magnitude have to be updated together. If one updates and the other doesn't — which could happen if there's a write failure partway through — cosine similarity scores for that document are computed against a stale magnitude and produce silently wrong ranking. The fix is updating vector and magnitude atomically under the write lock, in the same struct assignment.

### Synonym expansion cycles

If the synonym map contains a bidirectional entry (A maps to B, B maps to A, which can happen when loading a thesaurus without deduplication), expansion loops until stack overflow. The fix is tracking visited terms in a `map[string]bool` during expansion and capping depth at 10 terms.

---

## Concurrency audit

These are the data races and correctness issues found in `internal/index/engine.go` before the library launched. Each one was either fixed or accepted with a documented reason. The fixes apply to both the `go get` library and the `go install` gRPC server.

### idMapping data race

The engine maintains a `map[uint64]string` called `idMapping` that converts internal uint64 IDs back to the original string IDs that callers provided. Writes happen inside the inverted index write lock in `addInternal`. Reads happen in `rankAndFuse` — after all read locks are released — because the scorer receives a live reference to the map.

Concurrent `Add + Search` was a data race. `go test -race` caught it. A read happening at the same time as a write could return a partially written or zeroed-out string ID, so results could come back with empty ID fields.

The root cause was that fine-grained per-sub-index locks covered each sub-index independently but left engine-level fields (`idMapping`, `fstSize`, `fst`, `bkTree`) unprotected across method boundaries.

The fix was adding `mu sync.RWMutex` to `Engine` as a top-level gate. `Add`, `AddWithVector`, `AddBatch`, `Remove`, `Load`, and the public `RebuildFST` take `e.mu.Lock()`. `Search` and `Save` take `e.mu.RLock()`. With the lock held for the full duration of each public call, `idMapping` is always accessed consistently. The per-sub-index locks remain as defense-in-depth.

### Concurrent RebuildFST race

Every `Add` call checked whether new terms had been added to the vocabulary since the last FST build, and if so triggered a rebuild. This check-and-rebuild had no mutual exclusion. Two concurrent `Add` goroutines could both pass the `currentSize > fstSize` check and call `RebuildFST` simultaneously. Inside `RebuildFST`, both goroutines would write `e.fstSize` and call `e.fst.Build()` at the same time — a data race on both fields.

The fix was splitting `RebuildFST` into two versions: a public method that takes `Engine.mu.Lock()` for external callers, and an internal `rebuildFSTLocked` that assumes the engine write lock is already held. `Add` calls the internal version while it already holds the write lock. Only one FST rebuild can ever run at a time.

### SetFST race on the analyzer

`StandardAnalyzer.SetFST()` assigns `a.fst = fst` — just a pointer assignment, with no lock. `Analyze` and `AnalyzeQuery` read `a.fst` on every call. Concurrent `Add` (which calls `SetFST` after rebuilding the FST) and `Search` (which calls `Analyze`) raced on that pointer.

This is fixed by the same engine write lock. `SetFST` is only called inside `rebuildFSTLocked` while `Engine.mu.Lock()` is held. `Analyze` is only called inside `Search` while `Engine.mu.RLock()` is held. The engine-level lock ensures these never overlap.

### BKTree not cleaned on re-index

When a document is re-indexed with different content, `addInternal` correctly removes old posting-list entries from the inverted index, phonetic index, BM25, and TF-IDF. But old tokens stay in the BKTree indefinitely.

BK-trees don't support deletion as a structural property. The tree is organized by edit distances between nodes. There's no efficient way to remove a word without rebuilding the entire tree from scratch, which costs O(n log n) and would be more expensive than the problem it solves.

In practice this is harmless. A BKTree lookup for a stale token finds no matching postings and contributes nothing to results. The tree just grows monotonically under heavy re-indexing. If this becomes a production concern, a periodic rebuild can be triggered when the tree size grows significantly beyond the number of live tokens.

### globalSeen never shrinking

The `globalSeen` map tracked every term ever indexed for FST vocabulary management, but `Remove` never touched it. Terms from deleted documents stayed in the FST vocabulary permanently. The FST would suggest completions for terms that no longer existed in any document.

The fix was changing `globalSeen` from `map[string]bool` to `map[string]int` — reference counts instead of a presence flag. Each `Add` increments the count for each new token. Each `Remove` decrements and deletes terms that reach zero. A new field `docTokens map[uint64][]string` was added to `InvertedIndex` to track which raw tokens belong to each document, so `Remove` knows which counts to decrement. The FST is rebuilt from surviving keys only. The serialization format was bumped from v2 to v3 because the type of `globalSeen` changed in the gob stream.

### O(n) brute-force vector scan

Every search iterates over every document vector to compute dot product scores:

```go
for id, entry := range e.vectors.GetVectors() {
    scores[id] = ranking.DotProduct(queryVec, Float16ToFloats(entry.Vector))
}
```

At 1 million documents, each with 384-dimensional float16 vectors (768 bytes each), this is a 768 MB memory scan per query. On the benchmark hardware that takes 25-40ms. Meilisearch's HTTP round-trip is about 2ms. With vector search enabled, ZENITH is slower than Meilisearch at 1M documents. The latency story inverts completely.

For the benchmark, the latency numbers use `WithBM25Only()` to avoid this problem. Recall numbers use the full hybrid mode on 100K documents where the vector scan is about 77 MB and takes 2-3ms, which is acceptable.

The long-term fix is HNSW, which reduces approximate nearest-neighbor search from O(n) to O(log n) with tunable recall. That's a roadmap item. Adding it only changes `vectorPass` and how the graph is stored alongside the existing gob state.

### getSemanticNeighbors truncating without sorting

The function that finds semantically similar words for neural query expansion iterated over a Go map, collected candidates above a similarity threshold, and then truncated to the top N:

```go
if topN > 0 && len(candidates) > topN {
    candidates = candidates[:topN]
}
```

Go's map iteration order is deliberately randomized at runtime. "Top 5 semantic neighbors" was actually "first 5 neighbors encountered in random iteration order." Same query, potentially different expansion results across runs and goroutines.

The fix was collecting candidates as `{word, score}` pairs, sorting descending by dot product score, then truncating to the top N. Results are now deterministic.

### Negative dot products escaping normalization

`DotProduct` can return negative values when query and document vectors point in opposite directions. When the library normalizes scores by dividing by the maximum score in the result set, a negative raw score produces a negative normalized score — which violates the `[0.0, 1.0]` guarantee on `Result.Score`.

The fix was clamping dot products to 0.0 in `vectorPass` before they enter the ranking pipeline. A document with negative cosine similarity to the query has zero semantic relevance. Returning a negative score implies it's actively harmful, which isn't meaningful.

---

## Benchmark design decisions

Using a Go-native test harness (`go test ./bench/...`) would make the numbers trivially dismissable as self-reported. CI regression graphs signal engineering discipline but aren't shareable in any format that gets attention. A Docker Compose setup with a single entry point (`mage benchQuick`) is what gets shared on Hacker News and X because anyone can clone the repo and reproduce the exact same numbers on their own machine. The reproducibility is the credibility signal.

MS MARCO Passage Retrieval v1 is the standard IR evaluation corpus used in academic papers for the past decade. Engineers at search and vector database companies know it by name. Using it means ZENITH's recall numbers are directly comparable to published benchmarks. A proprietary or synthetic corpus invites "you cherry-picked your test data."

Taking the first N lines of the collection rather than a random sample means the dataset is deterministic without needing a random seed. Two people running the benchmark get identical data and identical results.

Recall@10 asks a simple question: did the relevant passage appear in the top 10 results? That's the clearest version of "does this engine find the right answer." NDCG weights by position within the top results and requires graded relevance judgments that MS MARCO doesn't provide in its standard form. MRR only looks at whether the single most relevant result was retrieved. Recall@10 is the simplest metric that matches what users actually care about.

Indexing throughput and concurrent queries-per-second are both excluded because both disadvantage ZENITH for structural reasons unrelated to search quality. ONNX inference makes bulk indexing slower than BM25-only engines regardless of everything else. The single-writer model is appropriate for embedded library use, not for a standalone server competing on QPS. Measuring either would be honest numbers that tell the wrong story. The benchmark output footnotes explain what wasn't measured and why.

Magefile instead of Makefile because `make` is not installed on standard Windows machines. The target audience already has Go. `go install github.com/magefile/mage@latest` and `mage benchQuick` works everywhere Go works.

The four benchmark columns — ZENITH lib, ZENITH gRPC, Meilisearch BM25, Typesense BM25 — tell a complete story. Embed ZENITH and get the lowest possible latency (in-process, no network). Run it as a server and it's still faster than the competition because gRPC is more efficient than HTTP/1.1 + JSON. ZENITH lib and ZENITH gRPC produce identical Recall@10 because they're the same engine. That's an important result: it proves the library and the server are not different products with different quality — they're the same search logic in two deployment shapes.

---

## A note on how this was built

Every component in ZENITH was written from scratch. The WAL, the skip-list, the SSTable compactor, the BK-tree, the FST dictionary, the RRF ranker, the ONNX tokenizer, the WordPiece implementation, the mean pooling step — all of it. Nothing was outsourced to an embedded key-value store or a vector database library.

That wasn't the pragmatic choice. The goal was to understand how search engines actually work at the component level, not to assemble one from off-the-shelf parts. If you're reading this to understand the internals, every component has a clear boundary, a reason to exist at that boundary, and the ability to be replaced or improved without touching anything else.

The consequence of building from scratch is that the internals are well understood and improvable. The storage engine can be swapped. The ranker is tunable. The embedder is an interface. The public API is stable because nothing internal leaks through it.
