# Query-serving layer: result caching + gap fixes — design

Status: **proposed, not yet implemented.** Written 2026-10-08, companion to `ROADMAP.md`
and `DECISIONS.md` (same "why things are built the way they are" convention), covering
the next body of work: a real query-result cache plus six smaller gaps found while
investigating it.

## Why this exists

`zenith search`, `zenith serve` (gRPC and HTTP), and `pkg/zenith`'s `DB.Search` all
eventually call `internal/index.Engine.SearchFilteredWeighted`
(`internal/index/search.go:133`) — directly (gRPC server, CLI), or through
`pkg/zenith.DB.Search` (`pkg/zenith/zenith.go:418,427`), which the HTTP sidecar and
persistent collections sit on top of. That one function is the entire query-serving
layer's choke point today, and it has no result-level cache: `internal/embedding/cache.go`
only caches the *embedding vector* for repeated query text, so an identical repeated
search still redoes the full lexical pass, vector pass, and RRF fusion every time.

Investigating that gap surfaced six more, found by reading the actual code, not assumed:

1. No result-level cache at all (above).
2. No per-mutation invalidation signal — `Engine.manifestGen` only bumps on a committed
   flush, but `Add`/`AddBatch`/`Delete` mutate the live delta under `e.mu.Lock()` and are
   immediately searchable before any flush.
3. No thundering-herd protection at the result level — item K's `singleflight` only
   covers embedding computation; concurrent identical full searches still each redo the
   whole pipeline independently.
4. `SearchFilteredWeighted` only reads `ctx` to embed the query; `lexicalPass`,
   `vectorPass`, `rankAndFuse`, `neuralExpand` take no `ctx` at all, so a canceled or
   timed-out request on a large exact-scan corpus runs to completion anyway.
5. Cache-key correctness hazard: filter, per-query weights, sort, and limit/offset must
   not be forgotten from the key, or an override would silently return another call's
   cached results.
6. No hit/miss observability — unmeasurable the same way item K's query-embedding cache
   almost was.
7. No cross-process cache sharing, so multiple `zenith serve` replicas behind a load
   balancer each run cold and never help each other — flagged by the user as a possible
   blocker, not a nice-to-have.

Two previously-suspected gaps turned out not to be gaps on closer inspection and are
**not** part of this design: reranking already exists (`internal/reranker`, opt-in
cross-encoder, wired into `zenith search --rerank`), and there is already an implicit
query planner (the ANN-vs-exact threshold switch, the hybrid-fusion shortcut) — this
design does not touch query planning.

## Scope

**In scope:** `internal/index` (the `Engine` cache itself, the write-generation counter,
the `ctx` cancellation fix), a new `internal/querycache` package (the cache
implementations), `internal/config` (new `Config` fields), `internal/metrics` (new
counters), `cmd/zenith/serve.go` and `cmd/server/main.go` (flags/wiring),
`internal/collections` (automatic per-tenant Redis namespacing), `DECISIONS.md` and
`README.md` (documentation).

**Out of scope, explicitly:** `pkg/zenith`'s public API gets no new `SearchOption` or
exported method this round — the cache is an internal implementation detail of `Engine`,
invisible to callers except as a (hopefully faster) `Search`. Semantic/near-duplicate
cache matching (e.g. "weather in SF" hitting a cached "SF weather") is out of scope —
this is an exact-match cache only. Query planning, rate limiting/admission control, and
any `pkg/zenith` changes are separate future work, not folded in here.

## Decision: two-tier cache (L1 in-process, optional L2 Redis)

Three approaches were considered: in-process only; a pluggable interface with an
optional Redis backend, used directly; and a two-tier cache (in-process L1 in front of
the same optional Redis L2). The two-tier design is chosen because it is the only one
that doesn't leave gap #7 half-solved while staying fast for the dominant case: item K's
own measurement found that the realistic traffic pattern is many clients hitting a
*popular, repeated* query against the *same* process — L1 serves that at zero network
cost, and L2 only has to earn its keep for the genuinely cross-replica case. L2 is pure
addition — with it unconfigured, the system behaves exactly like the in-process-only
approach, at zero extra cost.

## Architecture

```
SearchFilteredWeighted(ctx, query, filter, weights)
  │
  ├─ filter has no structured Spec (raw Predicate closure)? ─→ bypass cache entirely,
  │                                                             compute as today
  │
  ├─ key = hash(namespace, writeGen, query, filter.Spec JSON, weights)
  │
  ├─ Tiered[[]SearchResponse].Get(key)
  │     ├─ L1 (in-process LRU, native value, no decode) hit  → return (clone)
  │     ├─ L2 (Redis, gob-decoded, if configured) hit → backfill L1, return (clone)
  │     └─ miss on both
  │
  ├─ searchSF.Do(key, func() { <existing lexical+vector+fusion+neural-expansion logic,
  │                              unchanged> })
  │
  └─ Tiered.Set(key, result); return result (clone)
```

Every mutating method that already takes `e.mu.Lock()` (`Add`, `AddWithVectorAttrs`,
`AddBatch`, `Delete`) bumps a new `e.writeGen uint64` field before returning. Because the
generation is part of the key, a write makes every previously-cached key for that engine
permanently unreachable without touching the cache at all — no enumeration, no reverse
index, no active invalidation logic to get wrong. Stale entries are reclaimed by the L1
LRU's normal eviction and an L2 TTL (default 5 minutes, since Redis has no inherent LRU
bound unless the operator configures `maxmemory-policy`, and a cache that can only grow
is a worse failure mode for a shared resource than one that occasionally recomputes a
still-fresh value a few minutes early).

`Filter` already distinguishes a structured `FilterSpec` (JSON-serializable, used for
index-accelerated filtering) from an arbitrary raw `Predicate` closure
(`internal/index/search.go`'s `f.pred()`/`f.spec()`). Per CLAUDE.md, `pkg/zenith/filter.go`
sometimes builds raw predicates directly rather than through `FilterSpec`. A closure
can't be hashed into a key, so: **a filtered query only participates in the cache when
`f.spec() != nil`; a raw-closure filter transparently bypasses the cache (computes fresh,
exactly like today)** — a documented limit, not a silent gap, and it costs nothing (no
behavior change) for callers that hit it.

## Components

**New package `internal/querycache`** (generic over the cached value type, so the L1 path
never pays a serialization cost it doesn't need; no dependency on `internal/index` types,
so the package stays reusable and has no cyclic-import risk):

- `Cache[V any]` interface: `Get(key string) (V, bool)`, `Set(key string, val V)`. This is
  the shape L1 uses directly.
- `MemCache[V]`: wraps `github.com/hashicorp/golang-lru/v2` (already a dependency, same
  library `internal/embedding/cache.go` already uses) — plain LRU of native Go values, no
  TTL (the generation-in-key scheme makes staleness self-resolving), no (de)serialization
  on the hot path.
- `BytesCache` (non-generic): wraps `github.com/redis/go-redis/v9` (new dependency — same
  category of addition as `prometheus/client_golang` for item N: always linked once
  added, opt-in at runtime via config) — Redis fundamentally stores bytes, so this is the
  only tier that deals in `[]byte`. `Get`/`Set` take a `context.Context` and respect it;
  any Redis error (unreachable, timeout) is treated as a miss, never surfaced as a search
  error, logged at `Warn` — matching the existing "degrade gracefully" precedent for an
  unreachable embedder in `SearchFilteredWeighted`.
- `Tiered[V]`: a `Cache[V]` composed of an L1 `MemCache[V]` and an optional L2
  `BytesCache` (nil when Redis isn't configured), constructed with an `encode func(V)
  ([]byte, error)` / `decode func([]byte) (V, error)` pair supplied by the caller. `Get`
  checks L1 first (no decode); on an L1 miss with L2 configured, it decodes an L2 hit,
  backfills L1 with the decoded value, and returns it. `Set` writes L1 directly and
  encodes once for L2. `internal/index` instantiates this as
  `querycache.Tiered[[]SearchResponse]`, passing `encoding/gob` as the encode/decode pair
  — so the serialization cost is paid only on an L2 round-trip, never on an L1 hit.

**`internal/index` additions:**

- `Engine.cache querycache.Cache` (always a `Tiered` with at least an L1; nil only when
  `Config.QueryCacheSize <= 0` disables caching entirely — a nil check short-circuits
  `Get`/`Set` to zero overhead).
- `Engine.writeGen uint64`, protected by the existing `e.mu` (no new lock, no atomic
  needed — all mutators already hold `e.mu.Lock()`, all readers already hold
  `e.mu.RLock()`, and Go's `sync.RWMutex` already gives the needed visibility).
- `Engine.searchSF singleflight.Group`, dedicated to full-query computation, mirroring
  `CachingEmbedder.embedSF`/`querySF`'s existing pattern exactly.
- `ctx` threaded into `lexicalPhase`, `vectorPass`, `rankAndFuse`: checked once at the
  start of each phase, and inside the unbounded exact-scan loop in `vectorPass`'s
  `eachVector` callback every 2048 documents (frequent enough to cancel promptly on a
  large corpus, infrequent enough that the check's own cost is negligible). On
  cancellation, return `ctx.Err()` immediately instead of finishing the scan.

**`internal/config.Config` additions** (all zero-value-safe, so every existing
construction site — `cmd/server/main.go`, `cmd/zenith/engine.go`, `pkg/zenith/zenith.go`'s
`Open` — needs no code change and gets the new default automatically):

- `QueryCacheSize int` (default **1,000** — smaller than the embedding caches' 10,000,
  since a result-list value is larger and less bounded in size than a fixed-dim
  `[]float32`). `<= 0` disables the cache entirely.
- `QueryCacheTTL time.Duration` (L2 only; default 5 minutes; `0` means "use the default,"
  not "no TTL" — an unbounded shared Redis cache is the one failure mode worth refusing
  by default).
- `QueryCacheRedisAddr string` (default `""` = L2 disabled).
- `QueryCacheNamespace string` (default `""`; see below).

**Redis namespacing:** a shared Redis used by multiple collections must not let one
tenant's cached results collide with another's. `internal/collections.Manager` already
knows each collection's own ID, and passes it automatically as `QueryCacheNamespace` when
constructing that collection's `*zenith.DB` — no new user-facing configuration for the
common case. A raw-`Engine` user outside `collections` (direct `pkg/zenith` library use,
or `cmd/server`'s gRPC path) who wants Redis sharing across their own multiple processes
must set the namespace explicitly; the default (`""`, with Redis unconfigured) is always
safe since L1-only mode has no cross-process visibility to collide in the first place.

**`internal/metrics` additions**, following the existing `metrics.InstrumentEmbedder`
pattern (`internal/index` does not import `internal/metrics` today, and this design keeps
that separation): `Engine` exposes cache-hit/-miss counts through a small optional
observer hook (default no-op), and `cmd/zenith/serve.go`/`cmd/server/main.go` wire a real
`internal/metrics`-backed observer the same way they already wire
`metrics.InstrumentEmbedder`. New series: `zenith_query_cache_hits_total{tier="l1"|"l2"}`,
`zenith_query_cache_misses_total`.

**New CLI flags** on `zenith serve` (both `--http` and gRPC modes) and `cmd/server`:
`--query-cache-size`, `--query-cache-ttl`, `--query-cache-redis-addr`. (No
`--query-cache-namespace` flag — `collections` sets it automatically; a raw-engine user
needing it is advanced enough to be out of this design's CLI-ergonomics scope and can be
revisited if real demand shows up.)

## Error handling

- Redis unreachable, timeout, or any `go-redis` error: logged at `Warn`, treated as an L2
  miss, never a search error.
- `gob` decode failure on an L2 entry (e.g. a binary upgrade leaves stale-shaped entries
  in Redis from an older version): treated as a miss, overwritten on the next `Set`.
- `QueryCacheSize <= 0`: cache is `nil`; every `Get`/`Set` is a no-op; behavior is
  byte-for-byte identical to the engine today.
- A filter with no structured `Spec`: bypasses the cache (see Architecture), never an
  error.

## Testing

- `internal/querycache`: `MemCache` eviction correctness; `RedisCache` against a real
  local Redis in CI or a pure-Go fake (`github.com/alicebob/miniredis/v2`, test-only
  dependency, never shipped in the binary); `Tiered`'s L1-preferred-over-L2 and
  L2-hit-backfills-L1 behavior.
- `internal/index`, new table-driven tests alongside the existing
  `search_weights_test.go`/`sort_attribute_test.go` style:
  - A cache hit returns results identical to a fresh computation (differential test, same
    spirit as `layers_test.go`'s existing result-identical tests).
  - A write (`Add`/`Delete`) bumps `writeGen`; the next identical query recomputes rather
    than returning a stale hit.
  - N concurrent identical queries trigger exactly one real computation
    (`searchSF` proof, mirroring `cache_test.go`'s existing embedding-dedup test).
  - `WithWeights`/`WithFilter`/plain queries produce distinct cache keys — directly tests
    gap #5 (no cross-contamination between differently-parameterized calls).
  - A raw-`Predicate` filter (via `SearchWithFilter`) never touches the cache and behaves
    identically to today.
  - `ctx` cancellation on a large synthetic exact-scan corpus returns promptly with
    `ctx.Err()` instead of finishing the scan.
- `-race` on `internal/index`, `internal/querycache`.
- `gofmt -l`, `go build`/`go vet`/`go test` under both `CGO_ENABLED=0` and `=1` (CGO is
  unrelated to this subsystem, but kept as the project's standing verification bar).

## Rollout

The cache defaults **on** (1,000-entry in-process L1, no L2) for every existing
construction site, with no code change required at any of them — this is a deliberate,
stated behavior change (extra RAM for the LRU, previously-identical repeated queries now
return faster) rather than a silent one. Redis L2 defaults **off**; enabling it links
`go-redis` into the binary unconditionally once the dependency is added (same binary-size
trade-off already accepted for `prometheus/client_golang` in item N), but has zero runtime
effect until `--query-cache-redis-addr` (or `Config.QueryCacheRedisAddr`) is actually set.

## Documentation to update once implemented

- `DECISIONS.md`: why generation-in-key beats active invalidation; why Redis is optional
  and namespaced per-collection; why the `ctx`-cancellation fix rode along with this
  change instead of shipping separately.
- `README.md`: new "Query result caching" section, same structure as the existing
  "Metadata filtering" section — defaults, flags, and the explicit non-goal (exact-match
  only, no semantic/near-duplicate matching).
- `ROADMAP.md`: a new backlog item once shipped, in the existing item format, with real
  measured hit-rate/latency numbers (not projected ones) the same way every other item in
  that document is backed by a real run.
