package config

import "time"

type Config struct {
	// Storage
	WALDir          string
	DataDir         string
	MemTableMaxSize int64

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
	BloomFPRate    float64
	RRFConstant    float64
	PhoneticWeight float64
	VectorWeight   float64
	NeuralWeight   float64

	// WordVectors enables per-token embeddings at index time (used only by zero-result neural
	// expansion). Small per-request namespaces turn it off: it is the dominant indexing cost.
	WordVectors bool

	// Performance
	EmbeddingWorkers int
	CompactionInterv time.Duration
	WALSyncInterval  time.Duration

	// Nerve
	NerveGRPCAddr string
}

func DefaultConfig() *Config {
	return &Config{
		WALDir:          "./data/wal",
		DataDir:         "./data/sst",
		MemTableMaxSize: 67108864, // 64MB
		MaxResults:      1000,
		FuzzyMaxDist:    2,
		// Measured on MS MARCO dev (300 queries, 107k docs, lexical mode):
		// length-scaled fuzziness + the automatic prefix cap took p50 201ms -> 43ms
		// with Recall@10 unchanged (0.830) and typo Recall@10 0.550 -> 0.553. A fixed
		// distance of 2 on every word made 2-3 letter words match hundreds of
		// unrelated terms: ~70k candidates per query, 6% of them with any BM25 hit.
		FuzzyByLength: true,
		BloomFPRate:   0.01,
		// RRFConstant and VectorWeight were tuned on MS MARCO dev with all
		// 6,980 ground truths indexed: k=20/wVec=2.0 reached Recall@10 0.960
		// vs 0.918 for the previous k=60/equal-weight fusion (which scored
		// below the dense list alone at 0.947). Both sit on a broad plateau:
		// k 10–30 × wVec 1.5–3.0 all measured ≥ 0.952.
		// Re-checked with gte-small on 2026-10-08: still on the plateau (ROADMAP P1-6).
		RRFConstant:      20.0,
		EmbeddingWorkers: 4,
		CompactionInterv: 30 * time.Second,
		WALSyncInterval:  100 * time.Millisecond,
		NerveGRPCAddr:    "localhost:8000",
		PhoneticWeight:   0.3,
		VectorWeight:     2.0, // RRF semantic-list weight (lexical list weight is 1.0)
		NeuralWeight:     1.0,
		WordVectors:      true,
	}
}
