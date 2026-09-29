//go:build cgo

package index

import (
	"context"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestMSMARCOHybridWeightSweep sweeps VectorWeight, RRFConstant, and
// PhoneticWeight over the real MS MARCO hybrid index cached by
// TestMSMARCOHybrid, to check whether the current defaults (VectorWeight=2.0,
// RRFConstant=20.0, PhoneticWeight=0.3 — see config.DefaultConfig's comment,
// tuned with all-MiniLM-L6-v2 before gte-small became the default) are still
// best with the current default embedding model.
//
// VectorWeight and RRFConstant are baked into the ranking.Scorer at
// construction (see ranking.NewWeightedRRFRanker), not read dynamically from
// *config.Config like PhoneticWeight is, so each variant swaps eng.scorer
// rather than mutating cfg for those two — but reuses the same loaded index
// throughout, since reloading the ~107k-doc corpus costs minutes.
//
// Run: ZENITH_MSMARCO_WEIGHTSWEEP=1 ZENITH_MSMARCO_HYBRID_INDEX=<path> \
//
//	go test ./internal/index -run TestMSMARCOHybridWeightSweep -v -timeout 1800s
//
// Optional: ZENITH_MSMARCO_QUERIES=<n> (default 500).
func TestMSMARCOHybridWeightSweep(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_WEIGHTSWEEP") == "" {
		t.Skip("set ZENITH_MSMARCO_WEIGHTSWEEP=1 to run (needs a cached index from TestMSMARCOHybrid)")
	}
	idxPath := os.Getenv("ZENITH_MSMARCO_HYBRID_INDEX")
	if idxPath == "" {
		t.Skip("set ZENITH_MSMARCO_HYBRID_INDEX to the index cached by TestMSMARCOHybrid")
	}
	if _, err := os.Stat(idxPath); err != nil {
		t.Skipf("cached index not found at %s — run TestMSMARCOHybrid first", idxPath)
	}

	const cacheDir = "../../bench/.cache"
	nQueries := 500
	if n, _ := strconv.Atoi(os.Getenv("ZENITH_MSMARCO_QUERIES")); n > 0 {
		nQueries = n
	}
	_, _, queries, qrels := loadMSMARCO(t, cacheDir, 100_000)
	if nQueries < len(queries) {
		queries = queries[:nQueries]
	}

	localEmb, err := localembedder.NewByID(os.Getenv("ZENITH_MODEL"), cacheDir+"/models")
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}
	emb, err := embedding.NewCachingEmbedder(localEmb, 10_000)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	defer eng.Close()
	if err := eng.Load(idxPath); err != nil {
		t.Fatalf("loading cached index: %v", err)
	}
	t.Logf("corpus: %d live docs, %d segment(s)", eng.Count(), eng.SegmentCount())

	type variant struct {
		name                                      string
		vectorWeight, rrfConstant, phoneticWeight float64
	}
	dv, dr, dp := cfg.VectorWeight, cfg.RRFConstant, cfg.PhoneticWeight
	variants := []variant{
		{"default", dv, dr, dp},
		{"vectorWeight=1.0", 1.0, dr, dp},
		{"vectorWeight=1.5", 1.5, dr, dp},
		{"vectorWeight=3.0", 3.0, dr, dp},
		{"vectorWeight=4.0", 4.0, dr, dp},
		{"rrfConstant=10", dv, 10.0, dp},
		{"rrfConstant=40", dv, 40.0, dp},
		{"rrfConstant=60", dv, 60.0, dp},
		{"phoneticWeight=0", dv, dr, 0},
		{"phoneticWeight=0.6", dv, dr, 0.6},
	}

	ctx := context.Background()
	type result struct {
		name         string
		recall, ndcg float64
	}
	var out []result
	for _, v := range variants {
		eng.scorer = ranking.NewWeightedRRFRanker(v.rrfConstant, cfg.MaxResults, 1.0, v.vectorWeight)
		cfg.PhoneticWeight = v.phoneticWeight

		var hits int
		var ndcgSum float64
		for _, q := range queries {
			res, err := eng.Search(ctx, q.text)
			if err != nil {
				t.Fatal(err)
			}
			for i, x := range res {
				if i >= 10 {
					break
				}
				if x.ID == qrels[q.id] {
					hits++
					ndcgSum += 1 / math.Log2(float64(i)+2)
					break
				}
			}
		}
		n := float64(len(queries))
		r := result{name: v.name, recall: float64(hits) / n, ndcg: ndcgSum / n}
		out = append(out, r)
		t.Logf("%-22s Recall@10=%.4f  nDCG@10=%.4f", r.name, r.recall, r.ndcg)
		fmt.Fprintf(os.Stderr, "WEIGHTSWEEP %s | recall=%.4f ndcg=%.4f\n", r.name, r.recall, r.ndcg)
	}

	sort.SliceStable(out, func(i, j int) bool { return out[i].ndcg > out[j].ndcg })
	t.Logf("best by nDCG@10: %s (%.4f)", out[0].name, out[0].ndcg)
}
