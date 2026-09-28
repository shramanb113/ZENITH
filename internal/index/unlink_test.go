package index

import (
	"context"
	"hash/fnv"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

func hashID(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// postingsMention reports whether id appears in any lexical posting list.
func postingsMention(e *Engine, id uint64) bool {
	for _, ids := range e.inverted.GetData() {
		for _, x := range ids {
			if x == id {
				return true
			}
		}
	}
	for _, ids := range e.phonetics.GetData() {
		for _, x := range ids {
			if x == id {
				return true
			}
		}
	}
	return false
}

// Per-document fragments are no longer stored; Remove and re-index must
// recompute them from the document's tokens and leave no stale postings.
func TestRemoveAndReindex_LeaveNoStalePostings(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	ctx := context.Background()

	for id, text := range map[string]string{"a": "apple pie recipe", "b": "apple tart"} {
		if err := eng.Add(ctx, id, text); err != nil {
			t.Fatal(err)
		}
	}
	if !postingsMention(eng, hashID("a")) {
		t.Fatal("precondition: doc a should be in the postings")
	}

	if err := eng.Remove(ctx, "a"); err != nil {
		t.Fatal(err)
	}
	if postingsMention(eng, hashID("a")) {
		t.Fatal("removed doc a still referenced by a posting list")
	}
	if !postingsMention(eng, hashID("b")) {
		t.Fatal("removing a must not disturb doc b's postings")
	}

	// Re-index b with different text: the old terms must disappear.
	if err := eng.Add(ctx, "b", "banana bread"); err != nil {
		t.Fatal(err)
	}
	if res, _ := eng.Search(ctx, "tart"); len(res) != 0 {
		t.Fatalf("search for a term only the old b text had returned %d results", len(res))
	}
	if res, _ := eng.Search(ctx, "banana"); len(res) != 1 || res[0].ID != "b" {
		t.Fatalf("expected b for 'banana', got %+v", res)
	}
	if got := eng.Count(); got != 1 {
		t.Fatalf("Count = %d, want 1", got)
	}
}
