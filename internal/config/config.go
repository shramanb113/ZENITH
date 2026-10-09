package config

import "time"

type Config struct {
	// Search
	// MaxResults is the internal candidate cap the RRF ranker fuses over —
	// not a user-facing page size. Callers (pkg/zenith's WithLimit/Limit,
	// gRPC's SearchRequest.limit) truncate the returned slice themselves;
	// this only needs to be large enough not to clip before that happens.
	MaxResults   int
	FuzzyMaxDist int
	// PrefixFragmentCap bounds candidate generation from short edge-n-gram
	// prefixes: a prefix (shorter than the whole query token) whose posting list
	// holds more documents than this stops adding candidates. Such a prefix
	// ("h", "th") matches a large share of the corpus without meaning anything,
	// and on MS MARCO turned a ~4k-document real match set into ~73k candidates
	// per query. The full-token fragment is never capped, so every document that
	// contains a query term is still found. 0 = automatic (see
	// index.Engine.prefixCap); negative disables the cap (exhaustive expansion).
	PrefixFragmentCap int
	// FuzzyByLength scales the allowed edit distance with word length instead of
	// applying FuzzyMaxDist to every word: <=3 letters exact, 4-5 letters one
	// edit, longer words FuzzyMaxDist. See index.Engine.fuzzyDist.
	FuzzyByLength  bool
	RRFConstant    float64
	PhoneticWeight float64
	VectorWeight   float64

	// WordVectors enables per-token embeddings at index time (used only by zero-result neural
	// expansion). Small per-request namespaces turn it off: it is the dominant indexing cost.
	WordVectors bool

	// QueryCacheSize is the in-process (L1) query-result cache's max entry
	// count. <= 0 disables the query-result cache entirely (every Get/Set
	// becomes a no-op in internal/index.Engine, byte-for-byte identical to
	// the engine with no cache). Smaller than the embedding caches' 10,000
	// default, since a result-list value is larger and less bounded in size
	// than a fixed-dim []float32.
	QueryCacheSize int
	// QueryCacheTTL bounds how long an L2 (Redis) entry survives. 0 means
	// "use the 5-minute default," not "no TTL" — an unbounded shared Redis
	// cache is the one failure mode worth refusing by default. Has no effect
	// when QueryCacheRedisAddr is unset (L1 has no TTL; staleness is handled
	// by writeGen making old keys unreachable, reclaimed by LRU eviction).
	QueryCacheTTL time.Duration
	// QueryCacheRedisAddr is the L2 Redis address (e.g. "localhost:6379").
	// "" (default) disables L2; the cache is then in-process (L1) only.
	QueryCacheRedisAddr string
	// QueryCacheNamespace prefixes every cache key, so a shared Redis used by
	// multiple engines/collections doesn't let one tenant's cached results
	// collide with another's. "" is always safe when QueryCacheRedisAddr is
	// also unset (L1-only mode has no cross-process visibility to collide
	// in). internal/collections.Manager sets this automatically per
	// collection; a raw Engine user sharing Redis across their own multiple
	// processes must set it explicitly.
	QueryCacheNamespace string
	// QueryCacheSemanticThreshold, when > 0, enables near-duplicate query
	// matching: on an exact cache-key miss, the engine compares the
	// incoming query's embedding against recently cached queries sharing the
	// same filter+weights, returning the closest one above this cosine
	// similarity. 0 (default) disables it — a wrong near-duplicate hit is a
	// silently wrong answer, not just a slow one, so this is opt-in. A
	// suggested starting point is 0.97 (deliberately conservative); tune
	// from the zenith_query_cache_hits_total{tier="semantic"} counter.
	QueryCacheSemanticThreshold float64
	// ANNThresholdBandPct, when > 0, makes the ANN-vs-exact vector search
	// choice near the corpus-size threshold follow measured rolling-average
	// latency instead of always taking the static side of the line. E.g.
	// 0.20 means: within ±20% of the configured ANN threshold, pick
	// whichever path (ANN graph or exact scan) is currently faster. 0
	// (default) disables this — the static threshold alone decides, exactly
	// as today.
	ANNThresholdBandPct float64
}

func DefaultConfig() *Config {
	return &Config{
		MaxResults:   1000,
		FuzzyMaxDist: 2,
		// Measured on MS MARCO dev (300 queries, 107k docs, lexical mode):
		// length-scaled fuzziness + the automatic prefix cap took p50 201ms -> 43ms
		// with Recall@10 unchanged (0.830) and typo Recall@10 0.550 -> 0.553. A fixed
		// distance of 2 on every word made 2-3 letter words match hundreds of
		// unrelated terms: ~70k candidates per query, 6% of them with any BM25 hit.
		FuzzyByLength: true,
		// RRFConstant and VectorWeight were tuned on MS MARCO dev with all
		// 6,980 ground truths indexed: k=20/wVec=2.0 reached Recall@10 0.960
		// vs 0.918 for the previous k=60/equal-weight fusion (which scored
		// below the dense list alone at 0.947). Both sit on a broad plateau:
		// k 10–30 × wVec 1.5–3.0 all measured ≥ 0.952.
		// Re-checked with gte-small on 2026-10-08: still on the plateau (ROADMAP P1-6).
		RRFConstant:    20.0,
		PhoneticWeight: 0.3,
		VectorWeight:   2.0, // RRF semantic-list weight (lexical list weight is 1.0)
		WordVectors:    true,

		QueryCacheSize:              1000,
		QueryCacheTTL:               5 * time.Minute,
		QueryCacheRedisAddr:         "",
		QueryCacheNamespace:         "",
		QueryCacheSemanticThreshold: 0,
		ANNThresholdBandPct:         0,
	}
}
