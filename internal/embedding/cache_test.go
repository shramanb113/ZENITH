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
