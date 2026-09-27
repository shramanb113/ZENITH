package config

import "time"

type Config struct {
	// Storage
	WALDir          string
	DataDir         string
	MemTableMaxSize int64

	// Search
	MaxResults      int
	FuzzyMaxDist    int
	BloomFPRate     float64
	RRFConstant     float64
	PhoneticWeight  float64
	VectorWeight    float64
	NeuralWeight    float64

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
		WALDir:           "./data/wal",
		DataDir:          "./data/sst",
		MemTableMaxSize:  67108864, // 64MB
		MaxResults:       10,
		FuzzyMaxDist:     2,
		BloomFPRate:      0.01,
		// RRFConstant and VectorWeight were tuned on MS MARCO dev with all
		// 6,980 ground truths indexed: k=20/wVec=2.0 reached Recall@10 0.960
		// vs 0.918 for the previous k=60/equal-weight fusion (which scored
		// below the dense list alone at 0.947). Both sit on a broad plateau:
		// k 10–30 × wVec 1.5–3.0 all measured ≥ 0.952.
		RRFConstant:      20.0,
		EmbeddingWorkers: 4,
		CompactionInterv: 30 * time.Second,
		WALSyncInterval:  100 * time.Millisecond,
		NerveGRPCAddr: "localhost:8000",
		PhoneticWeight:   0.3,
		VectorWeight:     2.0, // RRF semantic-list weight (lexical list weight is 1.0)
		NeuralWeight:     1.0,
		WordVectors:      true,
	}
}
