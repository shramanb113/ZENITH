package index

import (
	"context"
	"reflect"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// fixedQueryEmbedder always returns the same vector, regardless of text. It
// stands in for a real embedder in tests that need an exact, known query
// vector; document vectors are supplied directly via AddWithVectorAttrs, so
// this is only ever consulted for the query side of a search.
type fixedQueryEmbedder struct{ vec []float32 }

func (f fixedQueryEmbedder) Embed(context.Context, string) ([]float32, error) { return f.vec, nil }
func (f fixedQueryEmbedder) EmbedBatch(_ context.Context, t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i := range t {
		out[i] = f.vec
	}
	return out, nil
}
func (f fixedQueryEmbedder) Dimensions() int { return len(f.vec) }

func weightTestEngine(queryVec []float32) *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, fixedQueryEmbedder{vec: queryVec}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	return e
}

// Weights{} must be a true no-op: SearchFilteredWeighted with a zero value
// behaves exactly like SearchFiltered.
func TestSearchFilteredWeighted_ZeroWeightsMatchesSearchFiltered(t *testing.T) {
	ctx := context.Background()
	e := weightTestEngine([]float32{1, 0})
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "lexDoc", "lexxx filler words", []float32{0, 1}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "vecDoc", "vecyyy filler words", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	want, err := e.SearchFiltered(ctx, "lexxx", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := e.SearchFilteredWeighted(ctx, "lexxx", nil, Weights{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("Weights{} changed behavior: default=%v, zero-weights=%v", want, got)
	}
}

// Overriding the vector-list weight below the keyword weight flips which of
// a lexical-only and a vector-only document ranks first, relative to the
// engine's configured default (VectorWeight=2.0 > keyword weight 1.0, so the
// vector-only document wins by default). A later plain SearchFiltered call
// must still return the default order — the override must not leak into the
// engine's shared scorer.
func TestSearchFilteredWeighted_VectorOverrideFlipsOrder(t *testing.T) {
	ctx := context.Background()
	e := weightTestEngine([]float32{1, 0})
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "lexDoc", "lexxx filler words", []float32{0, 1}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "vecDoc", "vecyyy filler words", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	def, err := e.SearchFiltered(ctx, "lexxx", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(def) != 2 || def[0].ID != "vecDoc" {
		t.Fatalf("default weights: got %v, want vecDoc first (VectorWeight=2.0 > keyword weight 1.0)", def)
	}

	ov, err := e.SearchFilteredWeighted(ctx, "lexxx", nil, Weights{Vector: 0.1})
	if err != nil {
		t.Fatal(err)
	}
	if len(ov) != 2 || ov[0].ID != "lexDoc" {
		t.Fatalf("Weights{Vector: 0.1}: got %v, want lexDoc first", ov)
	}

	def2, err := e.SearchFiltered(ctx, "lexxx", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(def2) != 2 || def2[0].ID != "vecDoc" {
		t.Fatalf("default weights after override: got %v, want vecDoc first — the override leaked into the engine's shared scorer", def2)
	}
}

// The RRF constant override reaches the per-call scorer: with both list
// weights equalised to 1.0 (Vector: 1.0 overrides the configured default of
// 2.0), a lexical-only and a vector-only document each rank 1st in their own
// list, so each one's fused score is exactly 1/(k+1) for the overridden k.
func TestSearchFilteredWeighted_RRFConstantOverrideChangesScoreMagnitude(t *testing.T) {
	ctx := context.Background()
	e := weightTestEngine([]float32{1, 0})
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "lexDoc", "lexxx filler words", []float32{0, 1}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "vecDoc", "vecyyy filler words", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	const k = 5.0
	got, err := e.SearchFilteredWeighted(ctx, "lexxx", nil, Weights{Vector: 1.0, RRF: k})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("want 2 results, got %d: %v", len(got), got)
	}
	want := 1.0 / (k + 1.0)
	for _, r := range got {
		if diff := r.Score - want; diff > 1e-12 || diff < -1e-12 {
			t.Fatalf("score %v for %s, want %v (RRF: %v override not applied)", r.Score, r.ID, want, k)
		}
	}
}

// The phonetic weight override only matters among documents with no literal
// BM25 hit (buildKwRank ranks any real hit above every coverage-only
// candidate regardless of coverage magnitude): "ba" is reached only via a
// Soundex match on the query token "bo" (coverage = phoneticWeight), "cat" is
// reached only via its own (shorter) word being one of the edge-n-gram
// prefixes generated from the longer query token "catalog" (coverage =
// 100*3/7 ≈ 42.86, phoneticWeight-independent). The query token is
// deliberately longer than the matched document word ("cat" rather than a
// longer word like "catalog" in the document): the reverse — a document word
// extending the query token — would also make AnalyzeQuery's FST prefix
// resolution add the full document word as an extra literal query token,
// promoting the document into the BM25-hit bucket and making the comparison
// moot. The engine default (PhoneticWeight 0.3) ranks the fragment match
// first; overriding PhoneticWeight above ~42.86 flips it.
func TestSearchFilteredWeighted_PhoneticOverrideFlipsCoverageOnlyOrder(t *testing.T) {
	ctx := context.Background()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	defer e.Close()

	if err := e.AddWithVectorAttrs(ctx, "phon", "ba", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "frag", "cat", nil, nil); err != nil {
		t.Fatal(err)
	}

	def, err := e.SearchFiltered(ctx, "catalog bo", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(def) != 2 || def[0].ID != "frag" || def[1].ID != "phon" {
		t.Fatalf("default phonetic weight: got %v, want frag before phon", def)
	}

	ov, err := e.SearchFilteredWeighted(ctx, "catalog bo", nil, Weights{Phonetic: 150})
	if err != nil {
		t.Fatal(err)
	}
	if len(ov) != 2 || ov[0].ID != "phon" || ov[1].ID != "frag" {
		t.Fatalf("Weights{Phonetic: 150}: got %v, want phon before frag", ov)
	}
}
