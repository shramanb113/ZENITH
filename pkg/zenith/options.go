package zenith

import (
	"fmt"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/internal/index"
)

type options struct {
	embedder           Embedder
	bm25Only           bool
	cacheSize          int
	fuzzyDistance      int
	limit              int
	checkpointInterval time.Duration
	noWordVectors      bool
	memoryLimitBytes   int64
	annMinDocs         int    // -1 = engine default
	model              string // registered embedding model id; "" = the bundled one
	modelsDir          string
	rerank             bool
	rerankModel        string // registered reranker model id; "" = the default

	queryCacheSize              int // -1 = engine/Config default
	queryCacheTTL               time.Duration
	queryCacheRedisAddr         string
	queryCacheNamespace         string
	queryCacheSemanticThreshold float64
	annThresholdBandPct         float64
	queryCacheObserver          index.CacheObserver
}

func defaultOptions() *options {
	return &options{
		annMinDocs:     -1,
		cacheSize:      10_000,
		fuzzyDistance:  2,
		limit:          10,
		queryCacheSize: -1,
	}
}

// Option configures a DB at Open time.
type Option func(*options) error

// SearchOption configures a single Search call without changing DB state.
type SearchOption func(*searchOptions)

type searchOptions struct {
	limit     int
	explain   bool
	filter    *Filter
	weights   index.Weights
	sortField string
	sortDesc  bool
	// err is the first invalid SearchOption argument seen; Search returns it
	// instead of running the query (SearchOption itself cannot return one).
	err error
}

// setErr records the first invalid-option error.
func (o *searchOptions) setErr(err error) {
	if o.err == nil {
		o.err = err
	}
}

// WithEmbedder replaces the default embedded ONNX embedder with a custom one.
func WithEmbedder(e Embedder) Option {
	return func(o *options) error {
		if e == nil {
			return ErrInvalidOption
		}
		o.embedder = e
		return nil
	}
}

// WithModel selects a registered embedding model by ID (see localembedder's
// registry: all-MiniLM-L6-v2, gte-small, bge-small-en-v1.5). The bundled model
// is used directly; any other must first be installed with `zenith models pull <id>`
// (or placed at <models dir>/<id>/model.onnx). Unlike the default embedder, an
// explicitly requested model that cannot be loaded is an error from Open, not a
// silent fall back to lexical-only search.
//
// The model's identity is recorded in the index, so an index built with one
// model refuses to open with another (ErrEmbedderMismatch) rather than mixing
// vector spaces. Requires a CGO build.
func WithModel(id string) Option {
	return func(o *options) error {
		if strings.TrimSpace(id) == "" {
			return ErrInvalidOption
		}
		o.model = id
		return nil
	}
}

// WithModelsDir sets where non-bundled models are looked up
// (default: ~/.zenith/models).
func WithModelsDir(dir string) Option {
	return func(o *options) error {
		if dir == "" {
			return ErrInvalidOption
		}
		o.modelsDir = dir
		return nil
	}
}

// WithBM25Only disables vector search, eliminating the CGo build dependency.
func WithBM25Only() Option {
	return func(o *options) error {
		o.bm25Only = true
		return nil
	}
}

// WithCacheSize sets the LRU embedding cache size. Default: 10,000.
// Pass 0 to disable caching.
func WithCacheSize(n int) Option {
	return func(o *options) error {
		if n < 0 {
			return ErrInvalidOption
		}
		o.cacheSize = n
		return nil
	}
}

// WithFuzzyDistance sets the BK-tree edit distance threshold. Default: 2.
// Clamped to [0, 5] — values above 5 produce O(n) BKTree scans.
func WithFuzzyDistance(n int) Option {
	return func(o *options) error {
		if n < 0 {
			return ErrInvalidOption
		}
		if n > 5 {
			n = 5
		}
		o.fuzzyDistance = n
		return nil
	}
}

// WithLimit sets the default maximum results returned by Search. Default: 10.
func WithLimit(n int) Option {
	return func(o *options) error {
		if n <= 0 {
			return ErrInvalidOption
		}
		o.limit = n
		return nil
	}
}

// Limit overrides the result limit for a single Search call.
func Limit(n int) SearchOption {
	return func(o *searchOptions) {
		if n > 0 {
			o.limit = n
		}
	}
}

// WithoutWordVectors skips per-token embeddings at index time. Semantic search still uses one vector
// per document; only zero-result neural expansion is disabled. Recommended for small, short-lived
// indexes where indexing latency matters.
func WithoutWordVectors() Option {
	return func(o *options) error {
		o.noWordVectors = true
		return nil
	}
}

// Explain makes Search return, for every document in the hybrid candidate list (at most
// Config.MaxResults) with a term hit or a positive semantic score, the raw per-signal evidence in
// Result.Signals. Only those candidates are examined, never the whole index, so the cost is bounded
// by the candidate list. Results are ordered by evidence (most query terms matched, then BM25, then
// cosine). Score keeps its usual meaning (normalised hybrid score) and must not be used as a
// threshold.
func Explain() SearchOption {
	return func(o *searchOptions) { o.explain = true }
}

// WithWeights overrides the per-list ranking weights used to fuse this one
// Search call's results, instead of the engine's configured defaults
// (DefaultConfig: VectorWeight 2.0, PhoneticWeight 0.3, RRFConstant 20.0).
// Pass 0 for any argument to keep that one at its engine default — the
// override applies only to this call; concurrent and later searches are
// unaffected.
//
//   - vector is the RRF weight of the semantic/vector result list, relative
//     to a fixed keyword/lexical weight of 1.0.
//   - phonetic is the per-match weight added for a Soundex phonetic hit
//     during the lexical pass.
//   - rrfConstant is the k in RRF's score(d) = Σ weight/(k + rank(d)); a
//     smaller k rewards a top rank more steeply, a larger k flattens the
//     advantage of ranking higher.
//
// A NaN, infinite or negative argument makes Search return an error wrapping
// ErrInvalidOption instead of running the query.
func WithWeights(vector, phonetic, rrfConstant float64) SearchOption {
	return func(o *searchOptions) {
		w := index.Weights{Vector: vector, Phonetic: phonetic, RRF: rrfConstant}
		if err := w.Validate(); err != nil {
			o.setErr(fmt.Errorf("%w: WithWeights: %v", ErrInvalidOption, err))
			return
		}
		o.weights = w
	}
}

// SortBy replaces score ordering for this one Search call with ordering by
// the named document attribute's value instead — not a secondary tiebreak.
// The sort is stable, so documents tied on the attribute's value (including
// every document missing it) keep their relative score order. A document
// missing field, or holding an array value for it, always sorts after every
// document with a comparable scalar (string/number/bool) value, regardless
// of desc. Documents holding differently-typed values for field (e.g. one
// string, one number) are incomparable to each other and keep their
// relative score order too.
func SortBy(field string, desc bool) SearchOption {
	return func(o *searchOptions) {
		o.sortField = field
		o.sortDesc = desc
	}
}

// WithMemoryLimit rejects new documents (Add/AddBatch return ErrIndexFull)
// once the index's estimated heap usage would exceed limitBytes, instead of
// growing unbounded until the OS kills the process.
//
// The estimate is approximate, not exact accounting: it multiplies the
// document count by a fixed per-document cost (~1.6KB) derived from
// TestScale's 1M-doc run in bench/BENCHMARK.md (1,575MB Go heap in use /
// 1,000,000 docs). Actual usage varies with document length, vocabulary
// overlap, and embedder mode — treat limitBytes as a safety margin, not a
// precise cap.
func WithMemoryLimit(limitBytes int64) Option {
	return func(o *options) error {
		if limitBytes <= 0 {
			return ErrInvalidOption
		}
		o.memoryLimitBytes = limitBytes
		return nil
	}
}

// WithCheckpointInterval sets how often the DB automatically saves a gob
// snapshot and resets the WAL in a background goroutine. Default: 0 (disabled).
// This bounds WAL growth so crash recovery only needs to replay a small delta.
// Minimum enforced: 10 seconds. Has no effect on :memory: databases.
func WithCheckpointInterval(d time.Duration) Option {
	return func(o *options) error {
		const minInterval = 10 * time.Second
		if d > 0 && d < minInterval {
			d = minInterval
		}
		o.checkpointInterval = d
		return nil
	}
}

// WithReranker enables cross-encoder reranking of the hybrid top candidates
// before Search applies its result limit. It reorders quality, not recall: a
// document absent from the hybrid candidate list is never added by reranking.
// Off by default. Requires a CGO build and the reranker model to be installed
// (see WithRerankerModel / `zenith models pull`); if enabled but the model
// cannot be loaded, Open returns an error rather than silently disabling it.
func WithReranker(enabled bool) Option {
	return func(o *options) error {
		o.rerank = enabled
		return nil
	}
}

// WithRerankerModel selects a registered cross-encoder by ID (see
// localembedder's reranker registry). Only meaningful together with
// WithReranker(true); "" uses the default (ms-marco-MiniLM-L-6-v2).
func WithRerankerModel(id string) Option {
	return func(o *options) error {
		if strings.TrimSpace(id) == "" {
			return ErrInvalidOption
		}
		o.rerankModel = id
		return nil
	}
}

// WithANNThreshold sets the document count at which vector search switches
// from an exact scan to an approximate HNSW graph (default 20,000). Pass 0 to
// always search exactly. The graph is rebuilt from stored vectors on Open, so
// very large indexes pay that build time at startup.
func WithANNThreshold(n int) Option {
	return func(o *options) error {
		if n < 0 {
			return ErrInvalidOption
		}
		o.annMinDocs = n
		return nil
	}
}

// WithQueryCacheSize sets the in-process (L1) query-result cache's max
// entry count. Default: 1,000. Pass 0 to disable the query-result cache
// entirely.
func WithQueryCacheSize(n int) Option {
	return func(o *options) error {
		if n < 0 {
			return ErrInvalidOption
		}
		o.queryCacheSize = n
		return nil
	}
}

// WithQueryCacheTTL bounds how long an L2 (Redis) query-cache entry
// survives. Default: 5 minutes. Has no effect unless WithQueryCacheRedisAddr
// is also set.
func WithQueryCacheTTL(d time.Duration) Option {
	return func(o *options) error {
		if d <= 0 {
			return ErrInvalidOption
		}
		o.queryCacheTTL = d
		return nil
	}
}

// WithQueryCacheRedisAddr enables an optional L2 query-result cache tier
// backed by Redis at addr (e.g. "localhost:6379"), shared across processes.
// Unset (default) keeps the cache in-process (L1) only.
func WithQueryCacheRedisAddr(addr string) Option {
	return func(o *options) error {
		if addr == "" {
			return ErrInvalidOption
		}
		o.queryCacheRedisAddr = addr
		return nil
	}
}

// WithQueryCacheNamespace prefixes every query-cache key, so a Redis L2
// shared by multiple DBs doesn't let one's cached results collide with
// another's. internal/collections.Manager sets this automatically per
// collection; set it explicitly only when sharing Redis across your own
// multiple processes outside collections.
func WithQueryCacheNamespace(ns string) Option {
	return func(o *options) error {
		o.queryCacheNamespace = ns
		return nil
	}
}

// WithQueryCacheSemanticThreshold enables near-duplicate query-cache
// matching: on an exact cache-key miss, compares the incoming query's
// embedding against recently cached queries sharing the same filter and
// weights, returning the closest one at or above this cosine similarity.
// Default: 0 (disabled) — a wrong near-duplicate hit is a silently wrong
// answer, not just a slow one, so this is opt-in. 0.97 is a conservative
// starting point.
func WithQueryCacheSemanticThreshold(threshold float64) Option {
	return func(o *options) error {
		if threshold < 0 || threshold > 1 {
			return ErrInvalidOption
		}
		o.queryCacheSemanticThreshold = threshold
		return nil
	}
}

// WithANNThresholdBand makes the ANN-vs-exact vector-search choice near the
// WithANNThreshold boundary follow measured rolling-average latency instead
// of always taking the static side of the line. pct is the band's half-width
// as a fraction of the threshold (e.g. 0.2 = within ±20%). Default: 0
// (disabled) — the static threshold alone decides, exactly as without this
// option.
func WithANNThresholdBand(pct float64) Option {
	return func(o *options) error {
		if pct < 0 || pct > 1 {
			return ErrInvalidOption
		}
		o.annThresholdBandPct = pct
		return nil
	}
}

// WithQueryCacheObserver installs o to receive this DB's query-cache
// hit/miss events (see internal/index.CacheObserver). Unset (default) means
// no observer — the engine's own no-op default applies, so this call has no
// effect on behavior, only observability. Internal callers that already
// build on an engine directly (internal/collections, internal/sidecar) use
// this to get the same metrics.NewQueryCacheObserver() wiring that
// cmd/zenith/engine.go and cmd/server/main.go apply to a raw --db engine,
// since zenith.Open gives them no other seam to reach the engine after
// construction.
func WithQueryCacheObserver(o index.CacheObserver) Option {
	return func(opt *options) error {
		opt.queryCacheObserver = o
		return nil
	}
}

// optionsSnapshot exposes an options value's fields for cross-package tests
// (internal/collections) that need to verify a []Option slice's effect
// without a real Open call. Test-only; not part of the supported API.
type optionsSnapshot struct {
	QueryCacheNamespace         string
	QueryCacheRedisAddr         string
	QueryCacheSize              int
	QueryCacheTTL               time.Duration
	QueryCacheSemanticThreshold float64
	ANNThresholdBandPct         float64
}

// SnapshotOptions applies opts to a fresh default options value and returns
// the fields relevant to query-cache namespacing. Exported for
// internal/collections' tests only.
func SnapshotOptions(opts ...Option) (optionsSnapshot, error) {
	o := defaultOptions()
	for _, fn := range opts {
		if err := fn(o); err != nil {
			return optionsSnapshot{}, err
		}
	}
	return optionsSnapshot{
		QueryCacheNamespace:         o.queryCacheNamespace,
		QueryCacheRedisAddr:         o.queryCacheRedisAddr,
		QueryCacheSize:              o.queryCacheSize,
		QueryCacheTTL:               o.queryCacheTTL,
		QueryCacheSemanticThreshold: o.queryCacheSemanticThreshold,
		ANNThresholdBandPct:         o.annThresholdBandPct,
	}, nil
}
