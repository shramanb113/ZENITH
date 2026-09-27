package embedding

import (
	"context"
	"errors"
	"hash/fnv"
	"sync/atomic"
	"testing"
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
