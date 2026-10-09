// Package embedding (this file): the visual-embedding counterpart to
// Embedder. A VisualEmbedder produces vectors for images and text in one
// shared geometric space (e.g. CLIP) — unlike Embedder, which only embeds
// text, and whose vectors are not comparable to a VisualEmbedder's.
package embedding

import (
	"context"
	"image"
)

// VisualEmbedder embeds images and text into one shared vector space, so a
// text query and a matching photo (even one with no readable text in it —
// the capability this exists for) can be compared directly. Implementations
// are expected to be safe for concurrent use, same as Embedder.
type VisualEmbedder interface {
	// EmbedImage returns an embedding for a single decoded image.
	EmbedImage(ctx context.Context, img image.Image) ([]float32, error)
	// EmbedImageBatch returns embeddings for all images in as few underlying
	// inference calls as the implementation can manage — callers ingesting
	// many images (e.g. a multi-page PDF) should prefer this over repeated
	// EmbedImage calls.
	EmbedImageBatch(ctx context.Context, imgs []image.Image) ([][]float32, error)
	// EmbedText embeds a text query into the same space EmbedImage/
	// EmbedImageBatch populate — used for the query side of visual search,
	// never for indexing documents (images are always the indexed side).
	EmbedText(ctx context.Context, text string) ([]float32, error)
	// Dimensions returns the shared output size of every method above.
	Dimensions() int
}
