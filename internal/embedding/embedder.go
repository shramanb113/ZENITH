package embedding

import (
	"context"
)

type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
	Dimensions() int
}

// Named is implemented by embedders that can identify their model. The index
// engine uses it to record which model produced a saved index's vectors, and
// to refuse to load a file back with a different, incompatible model.
// Embedders that don't implement it (e.g. a caller's custom WithEmbedder type)
// are recorded as "unknown" and are not checked for mismatch.
type Named interface {
	Name() string
}
