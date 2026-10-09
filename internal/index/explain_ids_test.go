package index

import (
	"context"
	"fmt"
	"math/rand"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// explainEngine is diffEngine with the production ranker (MaxResults candidate
// cap rather than RRFRanker's default of 10), so a ranked search returns every
// candidate on a corpus smaller than MaxResults.
func explainEngine() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = true
	e := NewEngine(cfg, bagEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetANNThreshold(0) // exact vector scan: the ranked candidates are exact
	e.SetAutoCompact(false)
	return e
}

// ExplainIDs over a ranked search's result IDs must equal the full-scan
// ExplainFiltered restricted to those IDs — same documents, same order, and
// bit-identical signals — across every layer (delta, frozen, segments), with
// and without filters. On a corpus below MaxResults with exact vector search the
// ranked candidates also cover every document the full scan explains, so the
// two are equal outright.
func TestExplainIDs_EqualsFullScanRestrictedToCandidates(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(11))
	path := filepath.Join(t.TempDir(), "explain.db")
	e := explainEngine()
	defer e.Close()

	queries := append(append([]string(nil), diffQueries...),
		"latency", "containr networking", "the", "kubernetes kubernetes cluster")

	add := func(i int) {
		text := diffText(r)
		if err := e.AddWithVectorAttrs(ctx, fmt.Sprintf("d%d", i), text, e.EmbedText(ctx, text), randAttrs(r)); err != nil {
			t.Fatal(err)
		}
	}

	compare := func(stage string, eng *Engine) {
		t.Helper()
		filters := []*Filter{nil, {Pred: tenantPred("a")}}
		for i := 0; i < 4; i++ {
			spec := randSpec(r, 2)
			f, err := spec.Compile()
			if err != nil {
				t.Fatal(err)
			}
			filters = append(filters, f)
		}
		compared := 0
		for _, q := range queries {
			for fi, f := range filters {
				wantTerms, full, err := eng.ExplainFiltered(ctx, q, f)
				if err != nil {
					t.Fatal(err)
				}
				ranked, err := eng.SearchFiltered(ctx, q, f)
				if err != nil {
					t.Fatal(err)
				}
				ids := make([]string, 0, len(ranked)+2)
				inRanked := make(map[string]bool, len(ranked))
				for _, res := range ranked {
					ids = append(ids, res.ID)
					inRanked[res.ID] = true
				}
				// Unknown and duplicate IDs are ignored.
				ids = append(ids, "no-such-doc")
				if len(ranked) > 0 {
					ids = append(ids, ranked[0].ID)
				}

				gotTerms, got, err := eng.ExplainIDs(ctx, q, ids, f)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(gotTerms, wantTerms) {
					t.Fatalf("%s: %q filter#%d: query terms %v, full scan %v", stage, q, fi, gotTerms, wantTerms)
				}
				want := make([]ExplainHit, 0, len(full))
				for _, h := range full {
					if inRanked[h.ID] {
						want = append(want, h)
					}
				}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("%s: %q filter#%d: restricted explain differs from the full scan restricted to the candidates\n got: %+v\nwant: %+v",
						stage, q, fi, got, want)
				}
				// Below MaxResults with an exact vector pass the candidates
				// cover everything the full scan explains — except a query
				// that analyses to no terms, which Search answers with nothing.
				if len(wantTerms) > 0 && len(want) != len(full) {
					t.Fatalf("%s: %q filter#%d: ranked candidates cover %d of the %d explained documents",
						stage, q, fi, len(want), len(full))
				}
				compared += len(got)
			}
		}
		if compared < 200 {
			t.Fatalf("%s: only %d explained documents compared — corpus too sparse to prove anything", stage, compared)
		}
	}

	for i := 0; i < 300; i++ {
		add(i)
	}
	compare("all in delta", e)

	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 60; i++ { // replacements and removals across the flushed segment
		add(r.Intn(300))
	}
	for i := 0; i < 40; i++ {
		e.Remove(ctx, fmt.Sprintf("d%d", r.Intn(300)))
	}
	compare("delta over a segment", e)

	// A frozen layer (mid-checkpoint) with deletions and replacements in it.
	if _, _, err := e.BeginCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 20; i++ {
		e.Remove(ctx, fmt.Sprintf("d%d", r.Intn(300)))
	}
	for i := 0; i < 20; i++ {
		add(r.Intn(300))
	}
	compare("frozen layer", e)
	if err := e.FinishCheckpoint(path); err != nil {
		t.Fatal(err)
	}
	compare("after checkpoint", e)

	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	compare("after compaction", e)
	reloaded := explainEngine()
	defer reloaded.Close()
	if err := reloaded.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}
	compare("after reload", reloaded)
}

// A stop-word-only query analyses to no terms: Search returns nothing, so
// there is nothing to explain, and ExplainIDs on an empty ID list is empty
// (non-nil).
func TestExplainIDs_EmptyIDs(t *testing.T) {
	e := explainEngine()
	defer e.Close()
	ctx := context.Background()
	if err := e.AddWithVectorAttrs(ctx, "a", "kubernetes cluster", e.EmbedText(ctx, "kubernetes cluster"), nil); err != nil {
		t.Fatal(err)
	}
	_, hits, err := e.ExplainIDs(ctx, "kubernetes", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if hits == nil || len(hits) != 0 {
		t.Fatalf("want empty non-nil hits, got %#v", hits)
	}
}
