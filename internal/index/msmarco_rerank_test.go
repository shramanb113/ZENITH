//go:build cgo

// package index_test (external): internal/reranker imports internal/index,
// so a test that needs both must live outside package index to avoid an
// import cycle (see MSMARCOQuery / LoadMSMARCOForTest in msmarco_data_test.go).
package index_test

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/reranker"
)

// TestMSMARCORerank checks the cross-encoder reranker (internal/reranker)
// against real MS MARCO hybrid search results. Per the plan's acceptance
// criterion, reranking must not lower Recall@10 and should raise nDCG@10.
//
// It reuses the index cached by TestMSMARCOHybrid (run that first) instead
// of re-embedding the corpus — this test measures only the reranking layer.
//
// Run: ZENITH_MSMARCO_RERANK=1 ZENITH_MSMARCO_HYBRID_INDEX=<path> \
//
//	go test ./internal/index -run TestMSMARCORerank -v -timeout 1800s
//
// Optional: ZENITH_MSMARCO_QUERIES=<n> (default 200 — cross-encoder
// inference is much slower per query than the hybrid pass alone) and
// ZENITH_RERANKER_MODEL (default: modelspec.DefaultRerankerID), read from
// <cacheDir>/models like `zenith models pull` installs.
func TestMSMARCORerank(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_RERANK") == "" {
		t.Skip("set ZENITH_MSMARCO_RERANK=1 to run (needs a cached index from TestMSMARCOHybrid)")
	}
	idxPath := os.Getenv("ZENITH_MSMARCO_HYBRID_INDEX")
	if idxPath == "" {
		t.Skip("set ZENITH_MSMARCO_HYBRID_INDEX to the index cached by TestMSMARCOHybrid")
	}
	if _, err := os.Stat(idxPath); err != nil {
		t.Skipf("cached index not found at %s — run TestMSMARCOHybrid first", idxPath)
	}

	const cacheDir = "../../bench/.cache"
	nQueries := 200
	if n, _ := strconv.Atoi(os.Getenv("ZENITH_MSMARCO_QUERIES")); n > 0 {
		nQueries = n
	}
	_, _, queries, qrels := index.LoadMSMARCOForTest(t, cacheDir, 100_000)
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
	eng := index.NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	defer eng.Close()
	if err := eng.Load(idxPath); err != nil {
		t.Fatalf("loading cached index: %v", err)
	}
	t.Logf("corpus: %d live docs, %d segment(s)", eng.Count(), eng.SegmentCount())

	rr, err := reranker.New(os.Getenv("ZENITH_RERANKER_MODEL"), cacheDir+"/models")
	if err != nil {
		t.Fatalf("reranker.New: %v (run: zenith models pull ms-marco-MiniLM-L-6-v2)", err)
	}
	defer rr.Close()

	ctx := context.Background()
	ndcgAt10 := func(ids []string, relevant string) float64 {
		for i, id := range ids {
			if i >= 10 {
				break
			}
			if id == relevant {
				return 1 / math.Log2(float64(i)+2)
			}
		}
		return 0
	}
	topIDs := func(raw []index.SearchResponse, n int) []string {
		if n > len(raw) {
			n = len(raw)
		}
		ids := make([]string, n)
		for i := 0; i < n; i++ {
			ids[i] = raw[i].ID
		}
		return ids
	}
	contains := func(ids []string, target string) bool {
		for _, id := range ids {
			if id == target {
				return true
			}
		}
		return false
	}

	var baseHits, rerankHits int
	var baseNDCG, rerankNDCG float64
	for _, q := range queries {
		raw, err := eng.Search(ctx, q.Text)
		if err != nil {
			t.Fatal(err)
		}
		baseIDs := topIDs(raw, 10)
		if contains(baseIDs, qrels[q.ID]) {
			baseHits++
		}
		baseNDCG += ndcgAt10(baseIDs, qrels[q.ID])

		reranked := rr.Rerank(ctx, q.Text, raw, eng.GetText)
		rerankIDs := topIDs(reranked, 10)
		if contains(rerankIDs, qrels[q.ID]) {
			rerankHits++
		}
		rerankNDCG += ndcgAt10(rerankIDs, qrels[q.ID])
	}

	n := float64(len(queries))
	baseRecall, rerankRecall := float64(baseHits)/n, float64(rerankHits)/n
	baseNDCG /= n
	rerankNDCG /= n

	t.Logf("baseline (hybrid only):   Recall@10=%.4f  nDCG@10=%.4f", baseRecall, baseNDCG)
	t.Logf("reranked (cross-encoder): Recall@10=%.4f  nDCG@10=%.4f", rerankRecall, rerankNDCG)
	fmt.Fprintf(os.Stderr, "RESULT rerank base_recall=%.4f base_ndcg=%.4f rerank_recall=%.4f rerank_ndcg=%.4f queries=%d\n",
		baseRecall, baseNDCG, rerankRecall, rerankNDCG, len(queries))

	if rerankRecall < baseRecall {
		t.Errorf("reranking lowered Recall@10: %.4f -> %.4f", baseRecall, rerankRecall)
	}
	if rerankNDCG < baseNDCG {
		t.Errorf("reranking lowered nDCG@10: %.4f -> %.4f (expected an improvement)", baseNDCG, rerankNDCG)
	}
}
