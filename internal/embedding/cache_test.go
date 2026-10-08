package embedding

import (
	"context"
	"errors"
	"hash/fnv"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeVectorEmbedder returns a stable, text-dependent vector. Unlike the
// production DeterministicEmbedder (which must always fail so the engine
// never fuses meaningless noise into ranking — see ErrEmbeddingUnavailable),
// these caching tests need an embedder that actually succeeds so cache
// hit/miss call-counting is meaningful.
type fakeVectorEmbedder struct{ dims int }

func newFakeVectorEmbedder(dims int) *fakeVectorEmbedder { return &fakeVectorEmbedder{dims: dims} }

func (f *fakeVectorEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	h := fnv.New64a()
	h.Write([]byte(text))
	seed := h.Sum64()
	out := make([]float32, f.dims)
	for i := range out {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		out[i] = float32(int64(seed)) / float32(1<<63)
	}
	return out, nil
}

func (f *fakeVectorEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v, _ := f.Embed(ctx, t)
		out[i] = v
	}
	return out, nil
}

func (f *fakeVectorEmbedder) Dimensions() int { return f.dims }

// countingEmbedder wraps another Embedder and counts Embed calls.
type countingEmbedder struct {
	inner Embedder
	calls atomic.Int64
}

func (c *countingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	c.calls.Add(1)
	return c.inner.Embed(ctx, text)
}

func (c *countingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	c.calls.Add(int64(len(texts)))
	return c.inner.EmbedBatch(ctx, texts)
}

func (c *countingEmbedder) Dimensions() int { return c.inner.Dimensions() }

// asymmetricEmbedder distinguishes Embed (document) from EmbedQuery (query)
// input, like BGE's query instruction prefix — the reason queryCache must be
// a separate cache from cache, not a second lookup into the same one.
type asymmetricEmbedder struct {
	inner      *fakeVectorEmbedder
	embedCalls atomic.Int64
	queryCalls atomic.Int64

	// delay, when set, is slept before computing the vector. Singleflight
	// only de-dups callers that are still in flight when a later caller
	// arrives; with an effectively instant base call and 50 unsynchronized
	// goroutines, a fast CI runner's scheduler can let the first caller's
	// Do finish (and its singleflight entry get cleaned up) before some of
	// the other 49 are even scheduled, splitting them into extra base
	// calls — observed as a flake on the macOS/cgo=0 CI lane. A small delay
	// here widens that window so every concurrent caller reliably arrives
	// while the one real call is still in flight.
	delay time.Duration
}

func (a *asymmetricEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	a.embedCalls.Add(1)
	if a.delay > 0 {
		time.Sleep(a.delay)
	}
	return a.inner.Embed(ctx, "doc:"+text)
}
func (a *asymmetricEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return a.inner.EmbedBatch(ctx, texts)
}
func (a *asymmetricEmbedder) Dimensions() int { return a.inner.Dimensions() }
func (a *asymmetricEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	a.queryCalls.Add(1)
	if a.delay > 0 {
		time.Sleep(a.delay)
	}
	return a.inner.Embed(ctx, "query:"+text)
}

// errorEmbedder always returns an error.
type errorEmbedder struct{}

func (e *errorEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, errors.New("embed error")
}
func (e *errorEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	return nil, errors.New("batch error")
}
func (e *errorEmbedder) Dimensions() int { return 4 }

func TestCachingEmbedder_CacheHit(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	const text = "hello world"

	_, _ = cached.Embed(ctx, text)
	_, _ = cached.Embed(ctx, text) // second call must hit cache
	_, _ = cached.Embed(ctx, text) // third call must hit cache

	if got := base.calls.Load(); got != 1 {
		t.Errorf("expected 1 call to base embedder, got %d", got)
	}
}

func TestCachingEmbedder_DifferentTexts(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	texts := []string{"foo", "bar", "baz"}
	for _, txt := range texts {
		_, _ = cached.Embed(ctx, txt)
	}
	// Call all again — should all hit cache.
	for _, txt := range texts {
		_, _ = cached.Embed(ctx, txt)
	}

	if got := base.calls.Load(); got != int64(len(texts)) {
		t.Errorf("expected %d base calls, got %d", len(texts), got)
	}
}

func TestCachingEmbedder_ErrorNotCached(t *testing.T) {
	base := &countingEmbedder{inner: &errorEmbedder{}}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	_, err1 := cached.Embed(ctx, "text")
	_, err2 := cached.Embed(ctx, "text")

	if err1 == nil || err2 == nil {
		t.Error("expected errors from error embedder")
	}
	if got := base.calls.Load(); got != 2 {
		t.Errorf("error should not be cached: expected 2 calls, got %d", got)
	}
}

func TestCachingEmbedder_BatchCacheMiss(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	texts := []string{"a", "b", "c"}
	vecs, err := cached.EmbedBatch(ctx, texts)
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != len(texts) {
		t.Errorf("expected %d vectors, got %d", len(texts), len(vecs))
	}
}

func TestCachingEmbedder_BatchPartialCacheHit(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()
	// Pre-populate cache for "cached_text"
	_, _ = cached.Embed(ctx, "cached_text")
	callsBefore := base.calls.Load()

	// Batch with one cached and one new
	vecs, err := cached.EmbedBatch(ctx, []string{"cached_text", "new_text"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 {
		t.Errorf("expected 2 vectors, got %d", len(vecs))
	}

	// Only "new_text" should have triggered a base call
	newCalls := base.calls.Load() - callsBefore
	if newCalls != 1 {
		t.Errorf("expected 1 new base call for partial miss, got %d", newCalls)
	}
}

func TestCachingEmbedder_EmbedQueryIsCachedSeparatelyFromEmbed(t *testing.T) {
	base := &asymmetricEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	const text = "capital of France"

	docVec, err := cached.Embed(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	queryVec, err := cached.EmbedQuery(ctx, text)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(docVec, queryVec) {
		t.Fatal("Embed and EmbedQuery returned the same vector for the same text — queryCache must not share cache's entries")
	}
	if base.embedCalls.Load() != 1 || base.queryCalls.Load() != 1 {
		t.Fatalf("expected 1 Embed call and 1 EmbedQuery call, got %d and %d", base.embedCalls.Load(), base.queryCalls.Load())
	}

	// Repeat both — must hit their respective caches, not call base again.
	if _, err := cached.Embed(ctx, text); err != nil {
		t.Fatal(err)
	}
	if _, err := cached.EmbedQuery(ctx, text); err != nil {
		t.Fatal(err)
	}
	if base.embedCalls.Load() != 1 || base.queryCalls.Load() != 1 {
		t.Fatalf("repeat calls should hit cache: embed=%d query=%d", base.embedCalls.Load(), base.queryCalls.Load())
	}
}

func TestCachingEmbedder_SymmetricEmbedQueryStillCached(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	cached, err := NewCachingEmbedder(base, 100)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	const text = "plain query"
	if _, err := cached.EmbedQuery(ctx, text); err != nil {
		t.Fatal(err)
	}
	if _, err := cached.EmbedQuery(ctx, text); err != nil {
		t.Fatal(err)
	}
	if got := base.calls.Load(); got != 1 {
		t.Fatalf("symmetric EmbedQuery should fall through to Embed's cache: expected 1 base call, got %d", got)
	}
}

func TestCachingEmbedder_ConcurrentMissesDoNotDuplicateWork(t *testing.T) {
	ctx := context.Background()
	const workers = 50
	const text = "same text, many concurrent callers"

	for _, tc := range []struct {
		name string
		call func(c *CachingEmbedder) ([]float32, error)
	}{
		{"Embed", func(c *CachingEmbedder) ([]float32, error) { return c.Embed(ctx, text) }},
		{"EmbedQuery", func(c *CachingEmbedder) ([]float32, error) { return c.EmbedQuery(ctx, text) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := &asymmetricEmbedder{inner: newFakeVectorEmbedder(8), delay: 20 * time.Millisecond}
			cached, err := NewCachingEmbedder(base, 100)
			if err != nil {
				t.Fatal(err)
			}
			var wg sync.WaitGroup
			ready := make(chan struct{})
			results := make([][]float32, workers)
			for i := 0; i < workers; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					<-ready // all workers race into the first call together
					v, err := tc.call(cached)
					if err != nil {
						t.Error(err)
						return
					}
					results[i] = v
				}(i)
			}
			close(ready)
			wg.Wait()

			if n := base.embedCalls.Load() + base.queryCalls.Load(); n != 1 {
				t.Fatalf("%d concurrent callers on the same uncached text produced %d base calls, want 1", workers, n)
			}
			for i, v := range results {
				if !reflect.DeepEqual(v, results[0]) {
					t.Fatalf("result[%d] differs from result[0]: %v vs %v", i, v, results[0])
				}
			}
			// Mutating one caller's result must never corrupt another's or
			// the cache's own stored copy (cloneVec's invariant).
			results[0][0] = 99999
			v2, err := tc.call(cached)
			if err != nil {
				t.Fatal(err)
			}
			if v2[0] == 99999 {
				t.Fatal("mutating a returned vector corrupted a later cache hit — missing a clone")
			}
		})
	}
}

func TestCachingEmbedder_Dimensions(t *testing.T) {
	dims := 128
	base := NewDeterministicEmbedder(dims)
	cached, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatal(err)
	}
	if cached.Dimensions() != dims {
		t.Errorf("Dimensions() = %d, want %d", cached.Dimensions(), dims)
	}
}

// fakePersistentCache is a minimal in-memory stand-in for *storage.Engine's
// GetEmbedding/PutEmbedding, used so this package's tests don't need to
// import internal/storage (keeping the decoupling the production code
// relies on honest in the tests too).
type fakePersistentCache struct {
	mu      sync.Mutex
	store   map[string][]float32
	putCall int
}

func newFakePersistentCache() *fakePersistentCache {
	return &fakePersistentCache{store: map[string][]float32{}}
}

func (f *fakePersistentCache) GetEmbedding(key []byte) ([]float32, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	v, ok := f.store[string(key)]
	return v, ok
}

func (f *fakePersistentCache) PutEmbedding(key []byte, vec []float32) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.putCall++
	f.store[string(key)] = vec
	return nil
}

func TestCachingEmbedder_PersistentCacheHitSkipsBaseEmbed(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	ce, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	persist := newFakePersistentCache()
	ce.SetPersistentCache(persist)

	// First call: LRU miss, persistent miss, real embed — populates both tiers.
	if _, err := ce.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if n := base.calls.Load(); n != 1 {
		t.Fatalf("calls = %d, want 1", n)
	}

	// A second CachingEmbedder sharing the same persistent store but an
	// empty LRU simulates a fresh process reusing the same on-disk cache.
	ce2, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	ce2.SetPersistentCache(persist)
	if _, err := ce2.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("Embed (second instance): %v", err)
	}
	if n := base.calls.Load(); n != 1 {
		t.Errorf("calls = %d, want still 1 (persistent cache hit must skip base.Embed)", n)
	}
}

// TestCachingEmbedder_PersistentCacheDimensionMismatchIsTreatedAsMiss proves
// a stored vector whose length doesn't match the current embedder's
// Dimensions() is never returned to the caller as a hit — this guards
// against a reused on-disk cache directory from a differently-sized model.
// Flagged by review as missing: deleting the `len(val) == c.Dimensions()`
// check in Embed/EmbedBatch/EmbedQuery would go unnoticed without this test.
func TestCachingEmbedder_PersistentCacheDimensionMismatchIsTreatedAsMiss(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	ce, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	persist := newFakePersistentCache()
	// Pre-seed the persistent cache with a vector of the WRONG dimension
	// (8, not base's 4) under the exact key ce will look up.
	persist.store[string(ce.persistentKey('D', "hello"))] = []float32{1, 2, 3, 4, 5, 6, 7, 8}
	ce.SetPersistentCache(persist)

	vec, err := ce.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 4 {
		t.Fatalf("Embed returned a %d-dim vector, want 4 (the dimension-mismatched 8-dim cache entry must not be returned)", len(vec))
	}
	if n := base.calls.Load(); n != 1 {
		t.Errorf("calls = %d, want 1 (a dimension mismatch must fall through to the base embedder, not short-circuit as a hit)", n)
	}
}

// zeroDimButRealVectorEmbedder mimics OllamaEmbedder's actual failure shape
// for an unrecognized model name: Dimensions() is a static lookup by model
// name string, disconnected from what the API actually returns, so it can
// report 0 while Embed still returns a real, non-empty vector.
type zeroDimButRealVectorEmbedder struct{ inner *fakeVectorEmbedder }

func (z *zeroDimButRealVectorEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	return z.inner.Embed(ctx, text)
}
func (z *zeroDimButRealVectorEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return z.inner.EmbedBatch(ctx, texts)
}
func (z *zeroDimButRealVectorEmbedder) Dimensions() int { return 0 }

// TestCachingEmbedder_ZeroDimensionBaseNeverUsesPersistentCache is a
// regression test for a real review finding: an embedder reporting
// Dimensions() <= 0 while still returning real, non-empty vectors (e.g.
// OllamaEmbedder.Dimensions() for an unrecognized model name) makes every
// persistent-cache lookup's dimension check (`len(val) == c.Dimensions()`)
// impossible to satisfy — without persistUsable's guard, every embed would
// still be written to the persistent store forever while no lookup could
// ever hit it.
func TestCachingEmbedder_ZeroDimensionBaseNeverUsesPersistentCache(t *testing.T) {
	base := &zeroDimButRealVectorEmbedder{inner: newFakeVectorEmbedder(4)}
	ce, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	persist := newFakePersistentCache()
	ce.SetPersistentCache(persist)

	vec, err := ce.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != 4 {
		t.Fatalf("Embed returned a %d-dim vector, want 4 (the base embedder's real output)", len(vec))
	}
	if persist.putCall != 0 {
		t.Errorf("putCall = %d, want 0 — a zero-Dimensions() embedder must never write to a persistent cache it can never read a hit from", persist.putCall)
	}
}

func TestCachingEmbedder_EmbedAndEmbedQuery_UseDistinctPersistentKeys(t *testing.T) {
	base := &asymmetricEmbedder{inner: newFakeVectorEmbedder(4)}
	ce, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	persist := newFakePersistentCache()
	ce.SetPersistentCache(persist)

	if _, err := ce.Embed(context.Background(), "same text"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if _, err := ce.EmbedQuery(context.Background(), "same text"); err != nil {
		t.Fatalf("EmbedQuery: %v", err)
	}
	if persist.putCall != 2 {
		t.Fatalf("putCall = %d, want 2 (distinct keys for Embed vs EmbedQuery)", persist.putCall)
	}
	if len(persist.store) != 2 {
		t.Errorf("stored %d distinct keys, want 2 — Embed/EmbedQuery must not collide", len(persist.store))
	}
}

func TestCachingEmbedder_EmbedBatch_ConsultsPersistentCache(t *testing.T) {
	base := &countingEmbedder{inner: newFakeVectorEmbedder(4)}
	ce, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	persist := newFakePersistentCache()
	ce.SetPersistentCache(persist)

	if _, err := ce.EmbedBatch(context.Background(), []string{"a", "b"}); err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if n := base.calls.Load(); n != 2 {
		t.Fatalf("calls = %d, want 2", n)
	}

	ce2, err := NewCachingEmbedder(base, 10)
	if err != nil {
		t.Fatalf("NewCachingEmbedder: %v", err)
	}
	ce2.SetPersistentCache(persist)
	if _, err := ce2.EmbedBatch(context.Background(), []string{"a", "b"}); err != nil {
		t.Fatalf("EmbedBatch (second instance): %v", err)
	}
	if n := base.calls.Load(); n != 2 {
		t.Errorf("calls = %d, want still 2 (persistent cache must satisfy both texts)", n)
	}
}
