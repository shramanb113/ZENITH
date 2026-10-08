package index

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// countingLexicalEmbedder counts how many times Embed/EmbedQuery is called,
// standing in for a real embedder in tests that need to prove the cache
// actually skipped recomputation (gap #3 from QUERYCACHE.md: thundering-herd
// protection at the result level).
type countingSearchEmbedder struct {
	vec   []float32
	calls atomic.Int64
}

func (f *countingSearchEmbedder) Embed(context.Context, string) ([]float32, error) {
	f.calls.Add(1)
	return f.vec, nil
}
func (f *countingSearchEmbedder) EmbedBatch(_ context.Context, t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i := range t {
		out[i] = f.vec
	}
	return out, nil
}
func (f *countingSearchEmbedder) Dimensions() int { return len(f.vec) }

func cacheTestEngine(emb *countingSearchEmbedder) *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = 1000
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	return e
}

func TestCache_HitReturnsSameResultAsFreshComputation(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	first, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(second) {
		t.Fatalf("cached result differs in length: first=%v second=%v", first, second)
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("cached result differs at %d: first=%v second=%v", i, first[i], second[i])
		}
	}
}

// Review Focus #5: Engine.SortByAttribute mutates its argument slice in
// place (`copy(results, sorted)`). If a caller's slice aliased a cached
// entry's backing array, that in-place sort would silently corrupt every
// future hit on the same key. cloneResponses (cache_result.go) exists
// specifically to prevent this — prove it actually does.
func TestCache_ReturnedSliceIsNotAliasedWithCachedEntry(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "docA", "hello world", []float32{1, 0}, Attrs{"rank": {Kind: AttrNumber, N: 2}}); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "docB", "hello world", []float32{1, 0}, Attrs{"rank": {Kind: AttrNumber, N: 1}}); err != nil {
		t.Fatal(err)
	}

	first, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatal(err)
	}
	firstOrder := make([]string, len(first))
	for i, r := range first {
		firstOrder[i] = r.ID
	}

	// Caller sorts its own copy of the results by "rank" — a cache hit's
	// slice, if aliased with the stored entry, would be reordered in place
	// right here.
	e.SortByAttribute(first, "rank", false)

	second, err := e.SearchFiltered(ctx, "hello", nil) // cache hit
	if err != nil {
		t.Fatal(err)
	}
	secondOrder := make([]string, len(second))
	for i, r := range second {
		secondOrder[i] = r.ID
	}
	for i := range firstOrder {
		if secondOrder[i] != firstOrder[i] {
			t.Fatalf("cached entry was mutated by the caller's SortByAttribute call: before=%v after=%v", firstOrder, secondOrder)
		}
	}
}

func TestCache_WriteBumpsGenerationSoNextQueryRecomputes(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	callsAfterFirst := emb.calls.Load()

	if err := e.AddWithVectorAttrs(ctx, "doc2", "hello again", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	if emb.calls.Load() <= callsAfterFirst {
		t.Fatalf("query embedder not called again after a write: calls before=%d after=%d", callsAfterFirst, emb.calls.Load())
	}
}

// Gap #3: N concurrent identical queries must trigger exactly one real
// computation, mirroring internal/embedding/cache_test.go's existing
// ConcurrentMissesDoNotDuplicateWork test.
func TestCache_ConcurrentIdenticalQueriesComputeOnce(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	const workers = 50
	var wg sync.WaitGroup
	ready := make(chan struct{})
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-ready
			if _, err := e.SearchFiltered(ctx, "concurrent-query", nil); err != nil {
				errs <- err
			}
		}()
	}
	close(ready)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if n := emb.calls.Load(); n != 1 {
		t.Fatalf("%d concurrent identical SearchFiltered calls produced %d embedder calls, want 1", workers, n)
	}
}

// Gap #1: a raw-Predicate filter (no structured Spec) must bypass the cache
// and still work exactly as before.
func TestCache_RawPredicateFilterBypassesCache(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	pred := func(a Attrs) bool { return true }
	first, err := e.SearchWithFilter(ctx, "hello", pred)
	if err != nil {
		t.Fatal(err)
	}
	callsAfterFirst := emb.calls.Load()
	second, err := e.SearchWithFilter(ctx, "hello", pred)
	if err != nil {
		t.Fatal(err)
	}
	if emb.calls.Load() <= callsAfterFirst {
		t.Fatal("raw-Predicate search hit the cache (embedder not called a second time); it must bypass the cache entirely")
	}
	if len(first) != len(second) || (len(first) > 0 && first[0].ID != second[0].ID) {
		t.Fatalf("bypassed searches returned different results: first=%v second=%v", first, second)
	}
}

// Two concurrent raw-Predicate searches with genuinely different predicates
// must never be merged by singleflight into one answer (gap #1's correctness
// half, not just "it still works" above).
func TestCache_ConcurrentDifferentRawPredicatesNeverMerge(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "allowed", "hello world", []float32{1, 0}, Attrs{"tag": {Kind: AttrString, S: "a"}}); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "blocked", "hello world", []float32{1, 0}, Attrs{"tag": {Kind: AttrString, S: "b"}}); err != nil {
		t.Fatal(err)
	}

	allowPred := func(a Attrs) bool { return a["tag"].S == "a" }
	blockPred := func(a Attrs) bool { return a["tag"].S == "b" }

	var wg sync.WaitGroup
	var allowRes, blockRes []SearchResponse
	wg.Add(2)
	go func() {
		defer wg.Done()
		allowRes, _ = e.SearchWithFilter(ctx, "hello", allowPred)
	}()
	go func() {
		defer wg.Done()
		blockRes, _ = e.SearchWithFilter(ctx, "hello", blockPred)
	}()
	wg.Wait()

	if len(allowRes) != 1 || allowRes[0].ID != "allowed" {
		t.Fatalf("allowPred result = %v, want exactly [allowed]", allowRes)
	}
	if len(blockRes) != 1 || blockRes[0].ID != "blocked" {
		t.Fatalf("blockPred result = %v, want exactly [blocked]", blockRes)
	}
}

// QueryCacheSize <= 0 disables the cache: behavior must be identical, and
// Get/Set become no-ops (no panic on a nil *Tiered field).
func TestCache_DisabledWhenSizeIsZero(t *testing.T) {
	ctx := context.Background()
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = 0
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
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	if n := emb.calls.Load(); n != 2 {
		t.Fatalf("with cache disabled, 2 identical searches produced %d embedder calls, want 2 (no caching)", n)
	}
}

// unmarshalableSpec is a *FilterSpec whose json.Marshal fails: a Value with
// Kind == AttrArray, which SpecValue.MarshalJSON (filterspec.go) rejects
// with "value has no type" — eq/in's Value field is documented to be a
// scalar, so this can only be reached by hand-building a Filter directly
// (bypassing FilterSpec.Compile's Validate), the same way a caller handing
// Search a raw Predicate bypasses the normal construction path.
func unmarshalableSpec(field string) *FilterSpec {
	return &FilterSpec{Op: "eq", Field: field, Value: &SpecValue{AttrValue{Kind: AttrArray}}}
}

// Review Focus (code review, 2026-10-08): a Filter whose Spec fails to
// marshal must bypass the cache entirely — not silently share a cache key
// with every other unmarshalable filter via a nil/zero-value specJSON.
func TestCache_UnmarshalableFilterSpecBypassesCache(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	f := &Filter{Pred: func(Attrs) bool { return true }, Spec: unmarshalableSpec("x")}
	if _, err := e.SearchFilteredWeighted(ctx, "hello", f, Weights{}); err != nil {
		t.Fatal(err)
	}
	callsAfterFirst := emb.calls.Load()
	if _, err := e.SearchFilteredWeighted(ctx, "hello", f, Weights{}); err != nil {
		t.Fatal(err)
	}
	if emb.calls.Load() <= callsAfterFirst {
		t.Fatal("search with an unmarshalable filter spec hit the cache (embedder not called a second time); it must bypass the cache entirely")
	}
}

// The failure mode a shared zero-value key would cause: two different
// unmarshalable filters (different tenants, in practice) must never share
// a cached result just because json.Marshal failed for both the same way.
func TestCache_DifferentUnmarshalableFilterSpecsNeverShareResult(t *testing.T) {
	ctx := context.Background()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := cacheTestEngine(emb)
	defer e.Close()
	if err := e.AddWithVectorAttrs(ctx, "tenant-a-doc", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.AddWithVectorAttrs(ctx, "tenant-b-doc", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	fA := &Filter{Pred: func(a Attrs) bool { return true }, Spec: unmarshalableSpec("tenant-a")}
	fB := &Filter{Pred: func(a Attrs) bool { return true }, Spec: unmarshalableSpec("tenant-b")}

	before := emb.calls.Load()
	if _, err := e.SearchFilteredWeighted(ctx, "hello", fA, Weights{}); err != nil {
		t.Fatal(err)
	}
	if _, err := e.SearchFilteredWeighted(ctx, "hello", fB, Weights{}); err != nil {
		t.Fatal(err)
	}
	if got := emb.calls.Load() - before; got != 2 {
		t.Fatalf("two different unmarshalable filters produced %d embedder calls, want 2 (each must recompute independently, never share a key)", got)
	}
}
