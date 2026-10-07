package index

import (
	"context"
	"fmt"
	"math/rand"
	"path/filepath"
	"sort"
	"testing"
)

var (
	idxFields  = []string{"tenant", "lang", "year", "tier", "public", "tags", "path"}
	idxTenants = []string{"acme", "globex", "initech", "umbrella"}
	idxLangs   = []string{"en", "hi", "fr"}
	idxTags    = []string{"go", "infra", "python", "rust", "ml"}
	idxPaths   = []string{"/docs/guide", "/docs/api", "/blog/notes", "/blog/release"}
)

func randAttrs(r *rand.Rand) Attrs {
	a := Attrs{}
	if r.Intn(10) > 0 {
		a["tenant"] = AttrValue{Kind: AttrString, S: idxTenants[r.Intn(len(idxTenants))]}
	}
	if r.Intn(3) > 0 {
		a["lang"] = AttrValue{Kind: AttrString, S: idxLangs[r.Intn(len(idxLangs))]}
	}
	if r.Intn(3) > 0 {
		a["year"] = AttrValue{Kind: AttrNumber, N: float64(2018 + r.Intn(8))}
	}
	if r.Intn(4) == 0 {
		a["tier"] = AttrValue{Kind: AttrNumber, N: float64(r.Intn(3))}
	}
	if r.Intn(2) == 0 {
		b := 0.0
		if r.Intn(2) == 0 {
			b = 1
		}
		a["public"] = AttrValue{Kind: AttrBool, N: b}
	}
	if r.Intn(2) == 0 {
		n := 1 + r.Intn(3)
		elems := make([]AttrValue, n)
		for i := 0; i < n; i++ {
			elems[i] = AttrValue{Kind: AttrString, S: idxTags[r.Intn(len(idxTags))]}
		}
		a["tags"] = AttrValue{Kind: AttrArray, Arr: elems}
	}
	if r.Intn(3) > 0 {
		a["path"] = AttrValue{Kind: AttrString, S: idxPaths[r.Intn(len(idxPaths))]}
	}
	return a
}

func randSpec(r *rand.Rand, depth int) FilterSpec {
	leaf := func() FilterSpec {
		switch r.Intn(7) {
		case 0:
			return FilterSpec{Op: "eq", Field: "tenant", Value: &SpecValue{AttrValue{Kind: AttrString, S: idxTenants[r.Intn(len(idxTenants))]}}}
		case 1:
			return FilterSpec{Op: "in", Field: "lang", Values: []SpecValue{
				{AttrValue{Kind: AttrString, S: idxLangs[r.Intn(len(idxLangs))]}},
				{AttrValue{Kind: AttrString, S: idxLangs[r.Intn(len(idxLangs))]}}}}
		case 2:
			lo, hi := float64(2018+r.Intn(8)), float64(2018+r.Intn(8))
			if lo > hi {
				lo, hi = hi, lo
			}
			s := FilterSpec{Op: "range", Field: "year", Min: &lo, Max: &hi}
			if r.Intn(3) == 0 {
				s.Max = nil
			}
			return s
		case 3:
			return FilterSpec{Op: "exists", Field: idxFields[r.Intn(len(idxFields))]}
		case 4:
			// eq/in against an array-valued field: "any element matches" in
			// both the index path (attrindex.set indexes each element) and
			// the predicate scan path (filterspec's anyMatch).
			return FilterSpec{Op: "eq", Field: "tags", Value: &SpecValue{AttrValue{Kind: AttrString, S: idxTags[r.Intn(len(idxTags))]}}}
		case 5:
			return FilterSpec{Op: "prefix", Field: "path", Value: &SpecValue{AttrValue{Kind: AttrString, S: idxPaths[r.Intn(len(idxPaths))][:5]}}}
		case 6:
			return FilterSpec{Op: "contains", Field: "path", Value: &SpecValue{AttrValue{Kind: AttrString, S: "/"}}}
		default:
			return FilterSpec{Op: "eq", Field: "public", Value: &SpecValue{AttrValue{Kind: AttrBool, N: float64(r.Intn(2))}}}
		}
	}
	if depth == 0 || r.Intn(3) == 0 {
		return leaf()
	}
	switch r.Intn(3) {
	case 0:
		return FilterSpec{Op: "and", Args: []FilterSpec{randSpec(r, depth-1), randSpec(r, depth-1)}}
	case 1:
		return FilterSpec{Op: "or", Args: []FilterSpec{randSpec(r, depth-1), randSpec(r, depth-1)}}
	default:
		return FilterSpec{Op: "not", Args: []FilterSpec{randSpec(r, depth-1)}}
	}
}

// The attribute index must never lose a matching document (candidates are a
// superset of the true matches), must be exact when it says so, and its
// estimate must be an upper bound — through inserts, replacements and removals.
func TestAttrIndex_PropertyAgainstPredicateScan(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	x := newAttrIndex()
	attrs := map[uint64]Attrs{}
	check := func(stage string) {
		t.Helper()
		for i := 0; i < 120; i++ {
			spec := randSpec(r, 3)
			f, err := spec.Compile()
			if err != nil {
				t.Fatal(err)
			}
			want := map[uint64]bool{}
			for id, a := range attrs {
				if f.Pred(a) {
					want[id] = true
				}
			}
			ords, exact, ok := x.candidates(&spec)
			est, estOK := x.estimate(&spec)
			if !ok {
				continue
			}
			ids := x.docIDs(ords)
			if !sort.SliceIsSorted(ords, func(i, j int) bool { return ords[i] < ords[j] }) {
				t.Fatalf("%s: candidates not ascending for %+v", stage, spec)
			}
			got := map[uint64]bool{}
			for _, id := range ids {
				if _, live := attrs[id]; !live {
					t.Fatalf("%s: candidate %d is not a live document (%+v)", stage, id, spec)
				}
				got[id] = true
			}
			for id := range want {
				if !got[id] {
					t.Fatalf("%s: index lost matching document %d for filter %+v", stage, id, spec)
				}
			}
			if exact {
				for id := range got {
					if !want[id] {
						t.Fatalf("%s: index claims exactness but returned non-match %d for %+v", stage, id, spec)
					}
				}
			}
			if estOK && est < len(ords) {
				t.Fatalf("%s: estimate %d below the %d candidates for %+v", stage, est, len(ords), spec)
			}
		}
	}
	for i := 0; i < 600; i++ {
		id := uint64(r.Intn(400))
		switch r.Intn(4) {
		case 0:
			x.drop(id)
			delete(attrs, id)
		default:
			a := randAttrs(r)
			x.set(id, a)
			if len(a) == 0 {
				delete(attrs, id)
			} else {
				attrs[id] = a
			}
		}
		if i%150 == 149 {
			check(fmt.Sprintf("after %d ops", i+1))
		}
	}
	// A rebuild from the attribute map gives an equivalent index.
	x = rebuildAttrIndex(attrs)
	check("after rebuild")
}

// End to end: a filter that carries its spec (index path) and the same filter
// as a bare predicate (scan path) return identical results, before and after
// flush, compaction and reload, with replacements and removals in between.
func TestFilter_IndexPathEqualsScanPath(t *testing.T) {
	ctx := context.Background()
	r := rand.New(rand.NewSource(5))
	path := filepath.Join(t.TempDir(), "filt.db")
	e := diffEngine()
	defer e.Close()

	add := func(i int) {
		text := diffText(r)
		if err := e.AddWithVectorAttrs(ctx, fmt.Sprintf("d%d", i), text, e.EmbedText(ctx, text), randAttrs(r)); err != nil {
			t.Fatal(err)
		}
	}
	compare := func(stage string, eng *Engine) {
		t.Helper()
		checked := 0
		for i := 0; i < 60; i++ {
			spec := randSpec(r, 3)
			f, err := spec.Compile()
			if err != nil {
				t.Fatal(err)
			}
			for _, q := range diffQueries[:6] {
				want, err := eng.SearchFiltered(ctx, q, &Filter{Pred: f.Pred}) // no spec: predicate scan
				if err != nil {
					t.Fatal(err)
				}
				got, err := eng.SearchFiltered(ctx, q, f) // spec: attribute index
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != len(want) {
					t.Fatalf("%s: %q with %+v: %d results via the index, %d via the scan", stage, q, spec, len(got), len(want))
				}
				for j := range got {
					if got[j].ID != want[j].ID || got[j].Score != want[j].Score {
						t.Fatalf("%s: %q with %+v: rank %d differs: %v vs %v", stage, q, spec, j, got[j], want[j])
					}
				}
				checked += len(got)
			}
		}
		if checked < 100 {
			t.Fatalf("%s: only %d results compared — the filters are too narrow to prove anything", stage, checked)
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
	if err := e.Save(path); err != nil {
		t.Fatal(err)
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	compare("after compaction", e)
	compare("after reload", mustLoad(t, path))
}

// Documents that carry no attributes are outside the index; Not and the scan
// must still treat them correctly.
func TestFilter_NotIncludesDocumentsWithoutAttributes(t *testing.T) {
	ctx := context.Background()
	e := diffEngine()
	defer e.Close()
	e.AddWithVectorAttrs(ctx, "a", "kubernetes cluster", e.EmbedText(ctx, "kubernetes cluster"), Attrs{"tenant": {Kind: AttrString, S: "acme"}})
	e.AddWithVectorAttrs(ctx, "b", "kubernetes cluster", e.EmbedText(ctx, "kubernetes cluster"), Attrs{"tenant": {Kind: AttrString, S: "globex"}})
	e.AddWithVectorAttrs(ctx, "c", "kubernetes cluster", e.EmbedText(ctx, "kubernetes cluster"), nil)
	spec := FilterSpec{Op: "not", Args: []FilterSpec{{Op: "eq", Field: "tenant", Value: &SpecValue{AttrValue{Kind: AttrString, S: "acme"}}}}}
	f, err := spec.Compile()
	if err != nil {
		t.Fatal(err)
	}
	res, err := e.SearchFiltered(ctx, "kubernetes cluster", f)
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, r := range res {
		got[r.ID] = true
	}
	if got["a"] || !got["b"] || !got["c"] || len(got) != 2 {
		t.Fatalf("Not(tenant=acme) returned %v, want b and c", got)
	}
}
