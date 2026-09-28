package zenith

import "time"

type options struct {
	embedder           Embedder
	bm25Only           bool
	cacheSize          int
	fuzzyDistance      int
	limit              int
	checkpointInterval time.Duration
	noWordVectors      bool
	memoryLimitBytes   int64
	annMinDocs         int // -1 = engine default
}

func defaultOptions() *options {
	return &options{
		annMinDocs:    -1,
		cacheSize:     10_000,
		fuzzyDistance: 2,
		limit:         10,
	}
}

// Option configures a DB at Open time.
type Option func(*options) error

// SearchOption configures a single Search call without changing DB state.
type SearchOption func(*searchOptions)

type searchOptions struct {
	limit   int
	explain bool
	filter  *Filter
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

// Explain makes Search return, for every document with a term hit or a positive semantic score,
// the raw per-signal evidence in Result.Signals. Results are ordered by evidence (most query terms
// matched, then BM25, then cosine). Score keeps its usual meaning (normalised hybrid score, 0 when
// the document is not in the hybrid result list) and must not be used as a threshold.
func Explain() SearchOption {
	return func(o *searchOptions) { o.explain = true }
}

// WithMemoryLimit rejects new documents (Add/AddBatch return ErrIndexFull)
// once the index's estimated heap usage would exceed limitBytes, instead of
// growing unbounded until the OS kills the process.
//
// The estimate is approximate, not exact accounting: it multiplies the
// document count by a fixed per-document cost (~11KB) derived from the
// measured hybrid-mode heap delta in bench/BENCHMARK.md (1,127MB / 100,000
// docs). Actual usage varies with document length, vocabulary overlap, and
// embedder mode — treat limitBytes as a safety margin, not a precise cap.
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
