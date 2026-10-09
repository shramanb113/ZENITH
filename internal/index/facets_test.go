package index

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func str(s string) AttrValue  { return AttrValue{Kind: AttrString, S: s} }
func numV(n float64) AttrValue { return AttrValue{Kind: AttrNumber, N: n} }
func arr(vs ...AttrValue) AttrValue {
	return AttrValue{Kind: AttrArray, Arr: vs}
}

// facetSeed indexes five documents: d1..d4 with lang/tags/year attributes
// (d2's tags repeat an element), d5 with none.
func facetSeed(t *testing.T, e *Engine) {
	t.Helper()
	ctx := context.Background()
	for _, d := range []struct {
		id    string
		attrs Attrs
	}{
		{"d1", Attrs{"lang": str("en"), "tags": arr(str("go"), str("search")), "year": numV(2024)}},
		{"d2", Attrs{"lang": str("en"), "tags": arr(str("go"), str("go")), "year": numV(2023)}},
		{"d3", Attrs{"lang": str("fr"), "tags": arr(str("search"))}},
		{"d4", Attrs{"lang": str("de"), "year": numV(2024)}},
		{"d5", nil},
	} {
		if err := e.AddWithVectorAttrs(ctx, d.id, "filler text "+d.id, nil, d.attrs); err != nil {
			t.Fatal(err)
		}
	}
}

func facetMap(list []FacetCount) map[attrKey]int {
	m := make(map[attrKey]int, len(list))
	for _, c := range list {
		m[keyOf(c.Value)] = c.Count
	}
	return m
}

func TestFacetCounts_CountsCandidatesAndArrayElementsOncePerDoc(t *testing.T) {
	e := sortAttrTestEngine(t)
	facetSeed(t, e)

	// d1 twice and an unknown id: duplicates and unknowns contribute nothing extra.
	f, err := e.FacetCounts([]string{"d1", "d2", "d3", "d5", "d1", "nope"}, []string{"lang", "tags", "missing"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := facetMap(f["lang"]), map[attrKey]int{keyOf(str("en")): 2, keyOf(str("fr")): 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("lang = %v, want %v", got, want)
	}
	// d2's [go, go] counts go once; d1 adds go and search; d3 adds search.
	if got, want := facetMap(f["tags"]), map[attrKey]int{keyOf(str("go")): 2, keyOf(str("search")): 2}; !reflect.DeepEqual(got, want) {
		t.Fatalf("tags = %v, want %v", got, want)
	}
	if l, ok := f["missing"]; !ok || len(l) != 0 {
		t.Fatalf("missing field = %v (present %v), want an empty slice", l, ok)
	}
	// Highest count first, ties broken by value.
	if f["lang"][0].Value.S != "en" {
		t.Fatalf("lang order = %v, want en first", f["lang"])
	}
	if f["tags"][0].Value.S != "go" || f["tags"][1].Value.S != "search" {
		t.Fatalf("tags tie order = %v, want go, search", f["tags"])
	}
}

func TestFacetCounts_TopK(t *testing.T) {
	e := sortAttrTestEngine(t)
	facetSeed(t, e)
	f, err := e.FacetCounts([]string{"d1", "d2", "d3", "d4"}, []string{"lang"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(f["lang"]) != 1 || f["lang"][0].Value.S != "en" || f["lang"][0].Count != 2 {
		t.Fatalf("topK=1 lang = %v, want [en:2]", f["lang"])
	}
}

func TestFacetCounts_EmptyFieldRejected(t *testing.T) {
	e := sortAttrTestEngine(t)
	if _, err := e.FacetCounts(nil, []string{""}, 0); !errors.Is(err, ErrEmptyFacetField) {
		t.Fatalf("FacetCounts err = %v, want ErrEmptyFacetField", err)
	}
	if _, err := e.CorpusFacetCounts([]string{"ok", ""}, 0); !errors.Is(err, ErrEmptyFacetField) {
		t.Fatalf("CorpusFacetCounts err = %v, want ErrEmptyFacetField", err)
	}
}

// CorpusFacetCounts answers from the attribute index's postings; it must equal
// FacetCounts over every live document id — including after a replace and a
// remove (dead postings), and after a Save/Load round trip (segment docs).
func TestCorpusFacetCounts_EqualsFacetCountsOverAllLiveDocs(t *testing.T) {
	ctx := context.Background()
	// TempDir first: cleanups run LIFO, so both engines unmap their segment
	// files before the directory is removed (Windows refuses otherwise).
	path := filepath.Join(t.TempDir(), "facets.db")
	e := sortAttrTestEngine(t)
	facetSeed(t, e)
	// Replace d3 (old postings go dead) and remove d4.
	if err := e.AddWithVectorAttrs(ctx, "d3", "filler", nil, Attrs{"lang": str("en"), "tags": arr(str("db"))}); err != nil {
		t.Fatal(err)
	}
	if err := e.Remove(ctx, "d4"); err != nil {
		t.Fatal(err)
	}
	live := []string{"d1", "d2", "d3", "d5"}
	fields := []string{"lang", "tags", "year"}

	check := func(name string, e *Engine) {
		t.Helper()
		want, err := e.FacetCounts(live, fields, 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := e.CorpusFacetCounts(fields, 0)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: corpus %v != over all live ids %v", name, got, want)
		}
		if facetMap(got["lang"])[keyOf(str("en"))] != 3 || facetMap(got["lang"])[keyOf(str("de"))] != 0 {
			t.Fatalf("%s: lang = %v, want en:3 and no de", name, got["lang"])
		}
	}
	check("in memory", e)

	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	re := sortAttrTestEngine(t)
	if err := re.Load(path); err != nil {
		t.Fatal(err)
	}
	check("reloaded", re)
}

func TestWeightsValidate(t *testing.T) {
	ok := []Weights{{}, {Vector: 2, Phonetic: 0.3, RRF: 20}}
	for _, w := range ok {
		if err := w.Validate(); err != nil {
			t.Fatalf("%+v: unexpected error %v", w, err)
		}
	}
	bad := []Weights{
		{Vector: math.NaN()}, {Phonetic: math.Inf(1)}, {RRF: math.Inf(-1)}, {Vector: -1}, {RRF: -0.5},
	}
	e := sortAttrTestEngine(t)
	for _, w := range bad {
		if err := w.Validate(); !errors.Is(err, ErrInvalidWeights) {
			t.Fatalf("%+v: err = %v, want ErrInvalidWeights", w, err)
		}
		if _, err := e.SearchFilteredWeighted(context.Background(), "x", nil, w); !errors.Is(err, ErrInvalidWeights) {
			t.Fatalf("SearchFilteredWeighted %+v: err = %v, want ErrInvalidWeights", w, err)
		}
	}
}

func TestSuggest_PrefixLowercasedStemmedVocabulary(t *testing.T) {
	ctx := context.Background()
	e := sortAttrTestEngine(t)
	for id, text := range map[string]string{
		"a": "kubernetes networking guide",
		"b": "kubectl reference",
		"c": "running the network",
	} {
		if err := e.Add(ctx, id, text); err != nil {
			t.Fatal(err)
		}
	}
	got, err := e.Suggest("  KUB ", 10)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"kubectl", "kubernet"}) {
		t.Fatalf("Suggest(kub) = %v, want the stemmed terms [kubectl kubernet] in lexicographic order", got)
	}
	if got, _ := e.Suggest("run", 10); !reflect.DeepEqual(got, []string{"run"}) {
		t.Fatalf("Suggest(run) = %v, want [run] (running is stemmed)", got)
	}
	if got, _ := e.Suggest("kub", 1); len(got) != 1 {
		t.Fatalf("Suggest(kub, 1) = %v, want one term", got)
	}
	if got, _ := e.Suggest("   ", 10); got == nil || len(got) != 0 {
		t.Fatalf("Suggest(blank) = %#v, want empty non-nil", got)
	}
	if got, _ := e.Suggest("zzz", 10); got == nil || len(got) != 0 {
		t.Fatalf("Suggest(zzz) = %#v, want empty non-nil", got)
	}
}

func TestAttrValueFromAnyRoundTrip(t *testing.T) {
	for _, v := range []any{"s", true, false, float64(3.5), []any{"a", float64(1), true}} {
		av, err := AttrValueFromAny(v)
		if err != nil {
			t.Fatalf("%v: %v", v, err)
		}
		if got := av.Any(); !reflect.DeepEqual(got, v) {
			t.Fatalf("round trip %#v -> %#v", v, got)
		}
	}
	if av, err := AttrValueFromAny(int32(7)); err != nil || av.Kind != AttrNumber || av.N != 7 {
		t.Fatalf("int32: %+v %v", av, err)
	}
	for _, v := range []any{nil, map[string]any{}, []any{[]any{"x"}}, math.NaN()} {
		if _, err := AttrValueFromAny(v); err == nil {
			t.Fatalf("%#v: want error", v)
		}
	}
}
