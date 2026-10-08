package index

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

func sortAttrTestEngine(t *testing.T) *Engine {
	t.Helper()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	t.Cleanup(func() { e.Close() })
	return e
}

func idsOfResponses(rs []SearchResponse) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	return out
}

func assertOrder(t *testing.T, got []SearchResponse, want []string) {
	t.Helper()
	gotIDs := idsOfResponses(got)
	if len(gotIDs) != len(want) {
		t.Fatalf("got %v, want %v", gotIDs, want)
	}
	for i := range want {
		if gotIDs[i] != want[i] {
			t.Fatalf("got %v, want %v", gotIDs, want)
		}
	}
}

// Numeric attribute: ascending/descending order, and a document missing the
// field always sorts last regardless of direction.
func TestEngine_SortByAttribute_Numeric(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)

	e.AddWithVectorAttrs(ctx, "a", "filler", nil, Attrs{"year": {Kind: AttrNumber, N: 2020}})
	e.AddWithVectorAttrs(ctx, "b", "filler", nil, Attrs{"year": {Kind: AttrNumber, N: 2010}})
	e.AddWithVectorAttrs(ctx, "c", "filler", nil, nil)

	results := []SearchResponse{{ID: "a", Score: 1}, {ID: "b", Score: 2}, {ID: "c", Score: 3}}
	e.SortByAttribute(results, "year", false)
	assertOrder(t, results, []string{"b", "a", "c"})

	results = []SearchResponse{{ID: "a", Score: 1}, {ID: "b", Score: 2}, {ID: "c", Score: 3}}
	e.SortByAttribute(results, "year", true)
	assertOrder(t, results, []string{"a", "b", "c"})
}

// String attribute: lexicographic ascending/descending.
func TestEngine_SortByAttribute_String(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)

	e.AddWithVectorAttrs(ctx, "a", "filler", nil, Attrs{"lang": {Kind: AttrString, S: "en"}})
	e.AddWithVectorAttrs(ctx, "b", "filler", nil, Attrs{"lang": {Kind: AttrString, S: "fr"}})
	e.AddWithVectorAttrs(ctx, "c", "filler", nil, nil)

	results := []SearchResponse{{ID: "c", Score: 1}, {ID: "b", Score: 2}, {ID: "a", Score: 3}}
	e.SortByAttribute(results, "lang", false)
	assertOrder(t, results, []string{"a", "b", "c"})

	results = []SearchResponse{{ID: "c", Score: 1}, {ID: "b", Score: 2}, {ID: "a", Score: 3}}
	e.SortByAttribute(results, "lang", true)
	assertOrder(t, results, []string{"b", "a", "c"})
}

// A document with an array value for the sort field is treated like a
// missing field: it always sorts last.
func TestEngine_SortByAttribute_ArrayValueSortsLast(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)

	e.AddWithVectorAttrs(ctx, "a", "filler", nil, Attrs{"tags": {Kind: AttrNumber, N: 5}})
	e.AddWithVectorAttrs(ctx, "b", "filler", nil, Attrs{"tags": {Kind: AttrArray, Arr: []AttrValue{{Kind: AttrNumber, N: 1}}}})

	results := []SearchResponse{{ID: "b", Score: 1}, {ID: "a", Score: 2}}
	e.SortByAttribute(results, "tags", false)
	assertOrder(t, results, []string{"a", "b"})
}

// Two documents with a mismatched attribute type (string vs number) for the
// same field are treated as incomparable — their relative order is left
// alone (stable), rather than one arbitrarily "winning".
func TestEngine_SortByAttribute_TypeMismatchIsStable(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)

	e.AddWithVectorAttrs(ctx, "a", "filler", nil, Attrs{"v": {Kind: AttrString, S: "x"}})
	e.AddWithVectorAttrs(ctx, "b", "filler", nil, Attrs{"v": {Kind: AttrNumber, N: 1}})

	results := []SearchResponse{{ID: "a", Score: 1}, {ID: "b", Score: 2}}
	e.SortByAttribute(results, "v", false)
	assertOrder(t, results, []string{"a", "b"})

	results = []SearchResponse{{ID: "b", Score: 1}, {ID: "a", Score: 2}}
	e.SortByAttribute(results, "v", false)
	assertOrder(t, results, []string{"b", "a"})
}

// Documents tied on the attribute's value keep their relative input order
// (the sort is stable, not re-ordered by score or ID).
func TestEngine_SortByAttribute_TiesAreStable(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)

	e.AddWithVectorAttrs(ctx, "a", "filler", nil, Attrs{"tier": {Kind: AttrNumber, N: 1}})
	e.AddWithVectorAttrs(ctx, "b", "filler", nil, Attrs{"tier": {Kind: AttrNumber, N: 1}})

	results := []SearchResponse{{ID: "b", Score: 1}, {ID: "a", Score: 2}}
	e.SortByAttribute(results, "tier", false)
	assertOrder(t, results, []string{"b", "a"})
}
