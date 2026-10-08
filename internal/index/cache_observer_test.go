package index

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

type fakeCacheObserver struct {
	hits   map[string]int
	misses int
}

func (f *fakeCacheObserver) ObserveQueryCacheHit(tier string) {
	if f.hits == nil {
		f.hits = map[string]int{}
	}
	f.hits[tier]++
}
func (f *fakeCacheObserver) ObserveQueryCacheMiss() { f.misses++ }

func TestCacheObserver_MissThenHit(t *testing.T) {
	ctx := context.Background()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()

	obs := &fakeCacheObserver{}
	e.SetCacheObserver(obs)

	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	if obs.misses != 1 {
		t.Fatalf("misses = %d after first (uncached) search, want 1", obs.misses)
	}
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	if obs.hits["l1"] != 1 {
		t.Fatalf("hits[l1] = %d after second (cached) search, want 1; hits=%v", obs.hits["l1"], obs.hits)
	}
}

func TestCacheObserver_DefaultIsNoOp(t *testing.T) {
	// No SetCacheObserver call: must not panic.
	ctx := context.Background()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()

	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
}
