package embedding

import (
	"context"
	"errors"
)

// ErrEmbeddingUnavailable is returned by DeterministicEmbedder to signal that
// no real embedding model is configured. The index engine treats any
// Embed/EmbedBatch error as "degrade to lexical-only search" (see CLAUDE.md:
// "Embedding failures are non-fatal — the engine degrades to lexical-only
// search"). DeterministicEmbedder previously fabricated a hash-seeded random
// unit vector instead of erroring; that vector carried no semantic meaning
// but was still fused into ranking at full VectorWeight, silently wrecking
// result quality — a document could be missing from the top 10 for its own
// exact-match query while an unrelated document with a favourable random dot
// product took its place.
var ErrEmbeddingUnavailable = errors.New("embedding: no embedding model available (deterministic fallback)")

// DeterministicEmbedder is a placeholder Embedder used only to satisfy the
// Embedder interface when no real model (ONNX, Ollama, Nerve, local) is
// configured. It never produces a vector — every call fails, so callers
// degrade to lexical-only search instead of ranking on meaningless noise.
type DeterministicEmbedder struct {
	dims int
}

func NewDeterministicEmbedder(dims int) *DeterministicEmbedder {
	return &DeterministicEmbedder{dims: dims}
}

func (d *DeterministicEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, ErrEmbeddingUnavailable
}

func (d *DeterministicEmbedder) EmbedBatch(_ context.Context, _ []string) ([][]float32, error) {
	return nil, ErrEmbeddingUnavailable
}

func (d *DeterministicEmbedder) Dimensions() int {
	return d.dims
}

// Name identifies this embedder for index-file compatibility checks (see
// Named). It never produces a real vector (see ErrEmbeddingUnavailable
// above), so an index saved under it holds no vectors either — recorded
// distinctly from a real model so switching to one is never mistaken for a
// no-op.
func (d *DeterministicEmbedder) Name() string {
	return "none:deterministic-fallback"
}
