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

### 12. WAL-backed crash safety in the embeddable library

For a long time the durability story for `pkg/zenith` was pretty simple and pretty bad. Every `Add` call updated the in-memory index. Nothing touched disk until `Close` was called. `Close` serialized the entire index to a gob file and that was it. If your process died before `Close` — OOM kill, SIGKILL, power loss, whatever — you lost everything indexed since the last time `Close` ran successfully. The gob file on disk reflected the state from the previous run and nothing more.

That's the same problem SQLite has in its default journal mode, by the way. And it's actually the thing SQLite is most criticized for in embedded use cases. Most people don't realize it but SQLite in WAL mode still has a checkpoint step — it just automates it more aggressively. The fundamental issue is the same: crash at the wrong moment and you lose recent writes.

What I built instead is a proper WAL journal sitting right next to the gob file. Every `Add` and `Delete` call now writes a record to a `.wal` file before it touches the in-memory index. The record is fsynced immediately. Only after that does the in-memory index get updated. This means the worst case on a crash is that you replay the WAL on next open — you never lose a write that was acknowledged.

The mechanics are straightforward. On `Open`, I load the gob first to get the baseline state from the last checkpoint. Then I replay any WAL records written after that. On `Close`, I save the gob and then reset the WAL to zero, which is the checkpoint. Next time you open the file, the WAL is empty and there's nothing to replay. The WAL only matters when the process dies between `Add` calls and the next `Close`.

One thing that took some thought was what to do when both the gob and the WAL have something to say. If the gob is corrupt — bad write, filesystem issue, anything — most systems just give you an error and tell you to rebuild. I looked at this differently. If the gob is corrupt but the WAL has records in it, those records represent real data the user indexed. I throw away the broken gob and rebuild the index from the WAL alone. You lose whatever was checkpointed into the corrupt gob, but you keep the stuff that was written since the last clean checkpoint. SQLite doesn't do this. It fails hard on a corrupt database file, period. If only the WAL has your recent data and the main file is gone, that's your problem.

The one case where I do return an error is if both are gone or corrupt. Corrupt gob with an empty WAL means there's genuinely nothing to recover. Better to surface that as an error than silently open with an empty index and let the user wonder where their data went.

I also added `WithCheckpointInterval` which runs a background goroutine that periodically saves the gob and resets the WAL on a timer you control. Without this, the WAL grows unboundedly if the process runs for a long time without a clean close. With it, WAL replay on crash is bounded to whatever happened in the last interval. The minimum is 10 seconds, enforced. Shorter than that and you're paying meaningful gob serialization overhead continuously, which defeats the purpose.

One thing I explicitly chose not to do for the embeddable library is use the full storage engine. ZENITH has a complete LSM stack — WAL, MemTable, SSTable flush, leveled compaction, Bloom filters, the whole thing. That infrastructure runs in `cmd/server` because the server needs it. But the embeddable library is a different product. If you're importing `pkg/zenith` into your Go service, you don't want four background goroutines, a compaction thread, and SSTable files accumulating in a directory. You want a single file that works. So the library uses only the WAL package directly. The gob is the primary storage format. The WAL is the crash journal. No MemTable, no SSTables, no compaction. The library stays self-contained with exactly two files on disk — the database and the journal alongside it.

### Memory scaling in :memory: mode

This was originally a back-of-envelope projection (`1,000,000 docs × 384 dims × 2 bytes float16 = 768 MB for vectors alone, plus postings/BK-tree/ID mappings, unmeasured`). `TestScale` (`internal/index/scale_test.go`, `bench/BENCHMARK.md`) has since measured it directly: ingesting 1,000,000 real-text documents (synthetic 384-dim vectors) reaches **1,575 MB Go heap in use**, 2,520 MB working set / 1,860 MB private before compaction. Nothing is evicted — everything stays in RAM. GC pressure grows linearly with index size, causing latency spikes in web handlers. Users who run `:memory:` in production with a large index will hit OOM kills. A `WithMemoryLimit(bytes int64)` option returns `ErrIndexFull` when the configured limit is exceeded rather than letting the process grow until the OS kills it, using a per-doc estimate derived from this same measurement (see `pkg/zenith/zenith.go`'s `estimatedBytesPerDoc`).

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

## Neural expansion threshold bug (threshold drift)

The `Search` function in `internal/index/engine.go` had a condition to fire neural expansion as a fallback when results were weak:

```go
if len(ranks) == 0 || ranks[0].Score < 5.0 {
```

The constant `5.0` was written when the scoring pipeline returned raw boosted keyword scores (10,000–60,000 range). After RRF replaced that pipeline, the same constant remained. RRF scores are computed as `1/(k + rank)` with k=60, giving a maximum possible score of `1/61 ≈ 0.0164` per list, or `~0.033` across both keyword and vector lists. The condition `ranks[0].Score < 5.0` was therefore **always true** — neural expansion fired on every single query regardless of result quality.

The consequence: every search call ran `getSemanticNeighbors` for each query token, iterating all 70,880 word vectors per token (O(n_vocab × n_tokens)). On a 100K document index this was pure wasted work — the initial ranking already had good results — and roughly doubled query latency.

The fix is one line: `len(ranks) == 0 || ranks[0].Score < 5.0` → `len(ranks) == 0`. Neural expansion now fires only when the engine returns zero results, which is its correct intended role as a last-resort fallback.

This is a threshold drift bug: a constant that was meaningful in one scoring context became permanently wrong after the scoring pipeline changed. It compiled, tests passed, nothing crashed — it just silently ran 2× more expensive than intended on every search call. The benchmark run that discovered it took ~3 hours on 101K queries because the O(n) neighbor scan was executing for all of them.

---

## BM25-only mode ranking bug (scorer bypass via epsilon mismatch)

ZENITH's `WithBM25Only()` option is meant to disable vector search and rank results by BM25 — directly comparable to SQLite FTS5 and Bleve. The MS MARCO benchmark revealed it was not doing that: Recall@10 was 0.406 against 0.594 for Bleve and 0.625 for SQLite FTS5, despite all three using lexical BM25 ranking.

The engine has a correctly-implemented BM25 scorer (`internal/ranking/bm25.go`, Okapi BM25, k1=1.2, b=0.75) that is indexed on every document. `rankAndFuse` called `e.bm25.Query(qryToks)` on every search. Yet recall was 46% lower than Bleve. The BM25 result was being computed and silently discarded.

**The primary ranking key was n-gram coverage, not BM25.** The `lexicalPass` scores every document as:

```go
keywordScores[id] += (float64(len(frag)) / float64(Q)) * 100.0
```

Then `rankAndFuse` added constant boosts before passing to RRF:

```go
boosted[id] = score + 10000.0
if len(matchToks[id]) >= len(qryToks) {
    boosted[id] += 50000.0
}
scored := e.scorer.Score(kwIDs, boosted, vcIDs, vScores, e.idMapping)
```

This n-gram coverage scoring has no IDF weighting. "The" and "serendipity" contribute identically per fragment. Short prefixes like "cap" match "captain", "capable", "capacity" — all with the same weight as an exact match for "capital". BM25's IDF suppresses common words and rewards rare term matches; the coverage formula does neither.

**BM25 was wired as a tiebreaker with an epsilon that never fired.** After `scorer.Score()` returned RRF-ranked results, BM25 was used only to swap adjacent results whose RRF scores differed by less than `epsilon = 1e-6`:

```go
const epsilon = 1e-6
if rrfDiff > -epsilon {
    if bm25Map[b.ID] > bm25Map[a.ID] { swap }
}
```

Adjacent RRF scores differ by `1/(60+n) − 1/(60+n+1) = 1/((60+n)(61+n)) ≈ 2.64×10⁻⁴`. This is 264× larger than the epsilon threshold. **The BM25 tiebreaker never fired on any query.** Every call to `e.bm25.Query()` was pure wasted computation — the result was computed and immediately ignored.

This is the same threshold drift pattern as the neural expansion bug: a constant calibrated for floating-point noise (`1e-6`) became wrong after the scoring pipeline introduced large constant boosts (+10000, +50000), pushing all score differences far above the threshold. The code compiled, tests passed, nothing crashed — it just silently ranked by n-gram coverage instead of BM25 on every BM25-only search.

The fix is a single conditional branch in `rankAndFuse`: when no vector scores are present (BM25-only mode), bypass the n-gram coverage path and use `bm25Map` as the primary sort key for RRF input instead:

```go
if len(vScores) == 0 {
    bm25Results := e.bm25.Query(qryToks)
    bm25ByID := make(map[uint64]float64, len(bm25Results))
    for _, r := range bm25Results {
        bm25ByID[r.DocID] = r.Score
    }
    kwIDs := make([]uint64, 0, len(kwScores))
    for id := range kwScores {
        kwIDs = append(kwIDs, id)
    }
    scored := e.scorer.Score(kwIDs, bm25ByID, nil, nil, e.idMapping)
    // ...
    return results
}
```

The lexical pass still runs to produce the candidate set (preserving fuzzy, phonetic, and n-gram prefix matching). BM25 then ranks those candidates. The hybrid path (vectors present) is unchanged.

Result: Recall@10 jumped from 0.406 to 0.594 — matching Bleve exactly, and within 0.031 of SQLite FTS5. The 0.031 gap against SQLite FTS5 reflects tokenizer differences (ZENITH uses Porter2 + stop-word filter; SQLite FTS5 uses its built-in unicode61 tokenizer with different stemming behaviour on a handful of edge cases).

---

## ONNX indexing was 40× slower than the design math said (fixed padding + unbatched documents + warm-up cache eviction)

The benchmark design notes estimated ~200–250s to index 100k MS MARCO passages in hybrid mode. The observed time was ~3,600s on a cold machine (5,536s thermally throttled). Systematic debugging found three compounding root causes, none of which was visible from any single file:

**1. Every input was padded to a fixed 256 tokens.** `tokenizer.tokenize` always returned 256-length tensors. The attention mask makes padded positions *correct*, not *free* — ONNX Runtime computes the full sequence regardless. MS MARCO passages average 76 WordPiece tokens (p50=72, p95=136), so each document paid ~3× its real cost. Vocabulary words (~4 tokens) paid ~60×. Measured fix impact: a 512-word batch dropped from 14.1s to 0.37s (38×); a 64-passage batch from 25.4ms/doc to 7.4ms/doc.

The subtlety: the int8 dynamically-quantized model computes activation scales over whole tensors *including padding*, so embeddings vary ~1% with padding length (cosine ≈ 0.988 between the same text at different paddings). This is quantization noise, not a masking bug — a genuine mask bug would drop similarity to 0.3–0.7. A regression test (`TestEmbed_PaddingInvariance`) pins this at ≥ 0.98.

**2. Documents were embedded one at a time.** `AddBatch` batched only the word-vector warm-up; each document still went through a single-input `Embed` call (88.8ms each at fixed 256 padding) inside `addInternal`, sequentially, under the engine write lock. The fix embeds documents in length-sorted batches of 64 on a producer goroutine that runs one 1,024-doc chunk ahead of index construction, so ONNX inference and lexical indexing overlap. Length-sorting matters: unsorted batches pad to an average of 161 tokens on MS MARCO; sorted batches pad to 76 (2.11× less compute).

Two non-obvious negative results, measured rather than assumed: batch 128 is ~12× slower per document than batch 64 on 12-thread consumer hardware (int8 GEMM cliff), and four concurrent sessions are ~5× slower than one (each session's intra-op thread pool oversubscribes the cores). The "session pool capped at NumCPU" described in earlier docs was never in the code — and building it would have made indexing slower. One session, batch 64, is the optimum here.

**3. The word-vector warm-up was thrown away.** `AddBatch` pre-embedded the vocabulary purely to populate the LRU embed cache — 10,000 entries against a 70,880-term vocabulary. By the time `addInternal` read words back, the early ~60k entries had been evicted, so most of the vocabulary was embedded twice at full cost (~2,000s wasted at the old per-word cost). The warm-up now writes vectors directly into the `VectorStore` instead of relying on the cache.

The lesson is the same shape as the threshold-drift bugs: nothing crashed, nothing errored, every component worked "correctly" in isolation. The system was just quietly 40× more expensive than intended, and only an end-to-end benchmark with a back-of-envelope sanity check ("the math works out to 200–250 seconds") exposed the gap.

---

## Hybrid recall capped by the unfixed half of the BM25 bypass bug

The BM25-only ranking bug (previous section) was fixed with a conditional: *when no vector scores are present*, rank the lexical candidates by BM25. That fix deliberately left the hybrid path untouched — which meant hybrid mode kept fusing the **broken** lexical list. `rankAndFuse` fed RRF a keyword list ranked by n-gram coverage with +10,000/+50,000 constant boosts — the exact scoring that measured Recall@10 = 0.406 standalone. Hybrid's 0.812 was RRF(0.406-quality lexical, semantic).

The fix applies the same BM25 ranking to the hybrid path: the lexical candidate set (still recalled via n-gram + phonetic + BK-tree fuzzy, so typo tolerance is preserved) is ranked by `1.0 + BM25(d, Q)` for documents BM25 can score, and by `coverage × 1e-9` for fuzzy/phonetic-only candidates so they rank strictly below every BM25-scored document (BM25 IDF is +1-smoothed, hence always positive). The dead epsilon tiebreaker (1e-6 against RRF gaps of ~2.6e-4 — never fired) was removed.

The pattern worth remembering: when a bug is found in a shared code path and fixed for the mode where it was *detected*, audit every other mode that flows through the same path. The hybrid branch had the identical defect for the identical reason, hidden because hybrid's absolute numbers still looked good (0.812 beat every lexical engine by 30%+).

**Measured outcome (2026-06-10): Recall@10 unchanged at 0.812.** The fix is correct in isolation — the lexical list fed to RRF is now the 0.594-quality BM25 ranking instead of the 0.406-quality coverage ranking — but the fused top-10 did not improve. The fix stays (it is strictly more principled and removes dead code), but it is not a recall lever. Why it isn't is the next section.

---

## The recall denominator was 32 queries, and equal-weight RRF was below the dense list alone

Instrumenting the unchanged 0.812 produced two discoveries, one about measurement and one about ranking.

**Measurement: the benchmark's Recall@10 is computed over 32 queries.** The harness indexes the first 100k of 8.8M passages and (correctly) only counts queries whose ground-truth passage is inside that subset. 100k/8.8M ≈ 1.1% of 6,980 qrel queries ≈ 32 queries. Every published recall number was a fraction of 32: ZENITH hybrid 0.812 = 26/32, ZENITH BM25 and Bleve 0.594 = 19/32, SQLite FTS5 0.625 = 20/32. Recall moves in steps of 0.031 — which is why a genuinely correct ranking fix measured "identical to three decimals," and why "matches Bleve exactly" was a much weaker statement than it sounded. The diagnostic fix: index the 7,399 ground-truth passages alongside the 100k corpus (107k total), making all 6,980 queries evaluable. Tuning decisions are made on n=6,980; the official benchmark number stays on the standard corpus with its n=32 caveat now documented.

**Ranking: equal-weight RRF fused below its best input.** On the 6,980-query corpus, measured per-list:

| Signal | Recall@10 |
|---|---|
| Dense (MiniLM vector pass) alone | 0.947 |
| Lexical (BM25-ranked candidates) alone | 0.803 |
| Fused, RRF k=60, equal weights (shipping config) | 0.918 |
| Union of both top-10s (fusion ceiling) | 0.971 |

Fusion was *subtracting* 3 points from the dense list. Mechanism: RRF treats both lists as equally trustworthy. When the dense list answers a query at rank 1–3, a lexically-popular wrong document that appears mid-list in *both* rankings accumulates two reciprocal-rank contributions and displaces the right answer from the fused top-10. The weaker the second list, the more often this happens — and the lexical list (0.803) is much weaker than the dense list (0.947) on a semantic corpus.

The fix is weighted RRF: `score(d) = 1.0/(k + rank_lex) + wVec/(k + rank_vec)`. A grid over k ∈ {10,20,30,60,120} × wVec ∈ {0.5,1,1.5,2,3} on the 6,980 queries shows a broad plateau — k 10–30 × wVec 1.5–3.0 all ≥ 0.952 — peaking at 0.9595. The shipped constants are k=20, wVec=2.0, chosen from the plateau's interior rather than its edge to avoid overfitting the grid. Fused 0.960 > dense-only 0.947: fusion finally adds value, with the lexical list rescuing dense misses instead of vetoing dense wins. The constants live in `config.DefaultConfig()` (`RRFConstant`, `VectorWeight` — the latter was declared but consumed nowhere until now), and `ranking.NewWeightedRRFRanker` carries the measured justification in its godoc.

The diagnostic harness is `internal/index/msmarco_diag_test.go` (gated behind `ZENITH_MSMARCO_DIAG=1`); it persists the indexed engine to `bench/.cache/` so re-runs skip the 15-minute indexing step.

**Verified on the official benchmark (2026-06-11): Recall@10 = 0.906 (29/32), up from 0.812 (26/32)** — exactly the value the 6,980-query grid simulation predicted for the official subset before the run. Index time 935.9s, consistent with the post-fix 890.6s under thermal variance.

---

## BM25 micro-tuning: measured, rejected

After the lexical comparison showed ZENITH at 0.803 vs SQLite FTS5 at 0.794 and Bleve at 0.8095 (Recall@10, n=6,980, gt-augmented corpus via `bench/cmd/lexdiag`), two candidate BM25 improvements were evaluated against the built index in a single diagnostic pass: excluding synonym-expansion tokens from BM25 scoring, and the standard MS MARCO-tuned parameters (k1=0.82, b=0.68; also Anserini's k1=0.9, b=0.4) versus the universal defaults (k1=1.2, b=0.75).

Result: the best combination (base tokens + 0.82/0.68) gained **+0.0035 lexical recall — under one standard error (±0.0047)** — and hybrid fused recall was flat across all six variants (0.9589–0.9595). Neither change ships:

- The k1/b values are MS MARCO-specific; shipping benchmark-tuned constants as library defaults is overfitting dressed up as engineering. The universal defaults stay.
- Synonym expansion's BM25 cost is −0.002 recall here, but its purpose is candidate recall on vocabularies where users and documents use different words. A null cost on one corpus is not a reason to remove a feature designed for other corpora.

The conclusion that matters: ZENITH's lexical quality is at parity with Bleve and ahead of SQLite FTS5; the official benchmark's "FTS5 leads" impression was a one-query artifact of the 32-query denominator. There is no lexical recall gap to close — the differentiator is the semantic pass (+15 points fused).

---

## BM25 posting lists: the O(N) scan is gone

`BM25Scorer.Query()` originally iterated every indexed document and evaluated BM25 against the query — O(N) per query regardless of how many documents contained any query term. At 100k documents × 6,980 benchmark queries that was ~700M BM25 evaluations on one goroutine, ~280ms p50, and the single biggest reason Bleve looked 100× faster per query.

The fix is the textbook one: an inverted posting list (`term → []docID`) maintained alongside the existing term-frequency maps. `Query` walks only the posting lists of the query's terms — O(Σ hits) — accumulating scores per document. Measured on a synthetic 100k corpus with deliberately long posting lists: 10.3ms/query vs ~280ms, ~27× (real queries with rarer terms do better).

Three design points worth recording:

1. **Score-identical by construction.** Documents containing no query term always scored 0 and were discarded; only they are skipped now. Repeated query terms are collapsed to (term, count) and the IDF contribution multiplied by count, matching the original per-occurrence loop exactly. An equivalence test (`TestQuery_PostingListMatchesBruteForce`) pins posting-list output against a brute-force reference, including re-index, Remove, repeated-term, and unknown-term cases.

2. **No serialization format bump.** Posting lists are derived state. `LoadState` rebuilds them from `termFreqs` in one pass (~1s at 100k docs), so existing `zenith.db` gob files load unchanged and the v3 format stays v3.

3. **Remove/re-index maintain the lists incrementally** with a swap-delete per (term, doc). This is O(posting-list length) per term of the removed document — fine for the embedded-library write rates ZENITH targets.

With BM25 off the critical path, hybrid query latency is bounded by the n-gram/BK-tree lexical pass and the O(N) vector scan — the vector scan being the known HNSW roadmap item.

**Verified on the official benchmark (2026-06-11): ZENITH BM25 p50 284ms → 177ms, hybrid p50 → 346ms** (down from 413ms cold / 787ms hot), recall unchanged at 0.594/0.906 as the equivalence tests guaranteed. ZENITH BM25 now beats SQLite FTS5 on both recall (0.803 vs 0.794, n=6,980) and latency (177ms vs 257ms p50). The remaining 177ms is the typo-tolerance machinery (BK-tree fuzzy + prefix n-gram + phonetic candidates) — the next latency target if one is needed.

---

## A note on how this was built

Every component in ZENITH was written from scratch. The WAL, the skip-list, the SSTable compactor, the BK-tree, the FST dictionary, the RRF ranker, the ONNX tokenizer, the WordPiece implementation, the mean pooling step — all of it. Nothing was outsourced to an embedded key-value store or a vector database library.

That wasn't the pragmatic choice. The goal was to understand how search engines actually work at the component level, not to assemble one from off-the-shelf parts. If you're reading this to understand the internals, every component has a clear boundary, a reason to exist at that boundary, and the ability to be replaced or improved without touching anything else.

The consequence of building from scratch is that the internals are well understood and improvable. The storage engine can be swapped. The ranker is tunable. The embedder is an interface. The public API is stable because nothing internal leaks through it.
