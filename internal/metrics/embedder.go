package metrics

import (
	"context"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// instrumentedEmbedder wraps an embedding.Embedder, observing
// zenith_embedding_duration_seconds / zenith_embedding_texts_total for every
// call. It always implements embedding.Named and embedding.QueryEmbedder
// (regardless of whether the wrapped embedder does) so wiring code can treat
// every InstrumentEmbedder result uniformly.
type instrumentedEmbedder struct {
	inner embedding.Embedder
}

// InstrumentEmbedder wraps e for metrics. Wrap it inside any caching layer
// (embedding.NewCachingEmbedder(metrics.InstrumentEmbedder(e), ...)), not
// outside it, so a cache hit is never counted as inference.
func InstrumentEmbedder(e embedding.Embedder) embedding.Embedder {
	return &instrumentedEmbedder{inner: e}
}

func (i *instrumentedEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	var v []float32
	var err error
	observeDuration(EmbeddingDuration.WithLabelValues("document"), func() {
		v, err = i.inner.Embed(ctx, text)
	})
	EmbeddingTextsTotal.WithLabelValues("document").Inc()
	return v, err
}

func (i *instrumentedEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	var v [][]float32
	var err error
	observeDuration(EmbeddingDuration.WithLabelValues("batch"), func() {
		v, err = i.inner.EmbedBatch(ctx, texts)
	})
	EmbeddingTextsTotal.WithLabelValues("batch").Add(float64(len(texts)))
	return v, err
}

func (i *instrumentedEmbedder) Dimensions() int { return i.inner.Dimensions() }

// Name forwards to the wrapped embedder when it implements embedding.Named;
// otherwise every existing index would hit ErrEmbedderMismatch against an
// unexpectedly-renamed "instrumentedEmbedder" identity.
func (i *instrumentedEmbedder) Name() string {
	if n, ok := i.inner.(embedding.Named); ok {
		return n.Name()
	}
	return "unknown"
}

// EmbedQuery goes through embedding.EmbedQuery, which itself falls back to
// Embed when the wrapped embedder has no query-specific mode — so a
// symmetric model is still counted (as "query", the call actually made).
func (i *instrumentedEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	var v []float32
	var err error
	observeDuration(EmbeddingDuration.WithLabelValues("query"), func() {
		v, err = embedding.EmbedQuery(ctx, i.inner, text)
	})
	EmbeddingTextsTotal.WithLabelValues("query").Inc()
	return v, err
}
