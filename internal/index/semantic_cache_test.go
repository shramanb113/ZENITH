package index

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// semanticEmbedder returns a per-text vector from a fixed table, so a test
// can control exactly how similar two different query strings' embeddings
// are — real embedders would make "near-duplicate" setups flaky.
type semanticEmbedder struct {
	vecs map[string][]float32
}

func (s *semanticEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if v, ok := s.vecs[text]; ok {
		return v, nil
	}
	return []float32{0, 0, 1}, nil // orthogonal to every configured vector
}
func (s *semanticEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i], _ = s.Embed(ctx, t)
	}
	return out, nil
}
func (s *semanticEmbedder) Dimensions() int { return 3 }

func semanticTestEngine(threshold float64) (*Engine, *semanticEmbedder) {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSemanticThreshold = threshold
	emb := &semanticEmbedder{vecs: map[string][]float32{
		"weather in SF": {1, 0, 0},
		"SF weather":    {0.99, 0.01, 0}, // near-duplicate of "weather in SF"
		"restaurants":   {0, 1, 0},       // unrelated
	}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	return e, emb
}

// Review Finding (code review, 2026-10-08): the original version of this
// test only compared len(first) == len(second), which would pass even with
// semantic matching entirely removed (every call here hits the same single
// document regardless). Asserting on the cache observer's "semantic" hit
// count is the only way to actually prove the near-duplicate match fired,
// rather than merely that both searches returned a result.
func TestSemanticCache_NearDuplicateAboveThresholdHits(t *testing.T) {
	ctx := context.Background()
	e, _ := semanticTestEngine(0.97)
	defer e.Close()
	obs := &fakeCacheObserver{}
	e.SetCacheObserver(obs)
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}

	first, err := e.SearchFiltered(ctx, "weather in SF", nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.SearchFiltered(ctx, "SF weather", nil) // near-duplicate, exact-key miss
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(second) {
		t.Fatalf("semantic hit returned a different result shape: first=%v second=%v", first, second)
	}
	if obs.hits["semantic"] != 1 {
		t.Fatalf("hits[semantic] = %d, want 1 — the near-duplicate must be served from the semantic scan, not recomputed", obs.hits["semantic"])
	}
}

func TestSemanticCache_BelowThresholdMissesAndRecomputes(t *testing.T) {
	ctx := context.Background()
	e, _ := semanticTestEngine(0.999) // "SF weather" (sim ~0.99) no longer clears this
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := e.SearchFiltered(ctx, "weather in SF", nil); err != nil {
		t.Fatal(err)
	}
	before := e.semanticScans.Load()
	if _, err := e.SearchFiltered(ctx, "SF weather", nil); err != nil {
		t.Fatal(err)
	}
	if e.semanticScans.Load() <= before {
		t.Fatal("semantic scan never ran even though threshold > 0")
	}
}

// Filter/weights must stay exact-only: two calls with identical query text
// but different weights must never semantically match each other.
func TestSemanticCache_DifferentWeightsNeverMatchSemantically(t *testing.T) {
	ctx := context.Background()
	e, _ := semanticTestEngine(0.5) // generous threshold
	defer e.Close()
	obs := &fakeCacheObserver{}
	e.SetCacheObserver(obs)
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := e.SearchFilteredWeighted(ctx, "weather in SF", nil, Weights{}); err != nil {
		t.Fatal(err)
	}
	// Same exact text, different weights bucket: must be an exact-key miss
	// AND a semantic miss (the only cached entry is in a different bucket),
	// so this must still succeed by recomputing, not by panicking or
	// returning the other bucket's entry.
	if _, err := e.SearchFilteredWeighted(ctx, "weather in SF", nil, Weights{Vector: 5}); err != nil {
		t.Fatal(err)
	}
	if obs.hits["semantic"] != 0 {
		t.Fatalf("hits[semantic] = %d, want 0 — a different weights bucket must never be served by the semantic scan despite a generous (0.5) threshold", obs.hits["semantic"])
	}
	if obs.misses != 2 {
		t.Fatalf("misses = %d, want 2 — both calls must be genuine recomputations, each in its own bucket", obs.misses)
	}
}

// Same text, different structured filters: the spec's own gap list calls
// this out explicitly alongside the weights case above, and it was missing
// from this file entirely before this fix.
func TestSemanticCache_DifferentFiltersNeverMatchSemantically(t *testing.T) {
	ctx := context.Background()
	e, _ := semanticTestEngine(0.5) // generous threshold
	defer e.Close()
	obs := &fakeCacheObserver{}
	e.SetCacheObserver(obs)
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0, 0}, Attrs{"lang": {Kind: AttrString, S: "en"}}); err != nil {
		t.Fatal(err)
	}

	specA := &FilterSpec{Op: "eq", Field: "lang", Value: &SpecValue{AttrValue{Kind: AttrString, S: "en"}}}
	specB := &FilterSpec{Op: "eq", Field: "lang", Value: &SpecValue{AttrValue{Kind: AttrString, S: "fr"}}}
	fA, err := specA.Compile()
	if err != nil {
		t.Fatal(err)
	}
	fB, err := specB.Compile()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := e.SearchFilteredWeighted(ctx, "weather in SF", fA, Weights{}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFilteredWeighted(ctx, "weather in SF", fB, Weights{}); err != nil {
		t.Fatal(err)
	}
	if obs.hits["semantic"] != 0 {
		t.Fatalf("hits[semantic] = %d, want 0 — a different filter bucket must never be served by the semantic scan despite a generous (0.5) threshold", obs.hits["semantic"])
	}
	if obs.misses != 2 {
		t.Fatalf("misses = %d, want 2 — both calls must be genuine recomputations, each in its own filter bucket", obs.misses)
	}
}

// Default off (threshold == 0): zero scan calls, not just zero hits.
func TestSemanticCache_DefaultOffNeverScans(t *testing.T) {
	ctx := context.Background()
	e, _ := semanticTestEngine(0) // default
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0, 0}, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := e.SearchFiltered(ctx, "weather in SF", nil); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFiltered(ctx, "SF weather", nil); err != nil {
		t.Fatal(err)
	}
	if n := e.semanticScans.Load(); n != 0 {
		t.Fatalf("semanticScans = %d with threshold 0, want 0 (no scan should ever run)", n)
	}
}
