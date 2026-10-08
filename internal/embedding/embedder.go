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

// QueryEmbedder is implemented by embedders whose model distinguishes a
// search query from a document (e.g. BGE's query instruction prefix). The
// engine embeds queries through it when available; embedders that don't
// implement it are used symmetrically via Embed.
type QueryEmbedder interface {
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// EmbedQuery embeds text as a search query using e's query mode if it has one.
func EmbedQuery(ctx context.Context, e Embedder, text string) ([]float32, error) {
	if q, ok := e.(QueryEmbedder); ok {
		return q.EmbedQuery(ctx, text)
	}
	return e.Embed(ctx, text)
}

// PersistentEmbedCache is a crash-surviving second tier beneath
// CachingEmbedder's in-memory LRU. Satisfied by *storage.Engine
// (internal/storage) — its GetEmbedding/PutEmbedding signatures match
// exactly; no import of internal/storage is needed here, mirroring how
// DocumentJournal/Txn already decouple internal/index from internal/storage.
type PersistentEmbedCache interface {
	GetEmbedding(key []byte) ([]float32, bool)
	PutEmbedding(key []byte, vec []float32) error
}

// PersistentCacheSetter is implemented by embedders that can be given a
// PersistentEmbedCache after construction (CachingEmbedder). Callers that
// build an embedder generically (cmd/zenith, cmd/server) type-assert against
// this instead of depending on the concrete type.
type PersistentCacheSetter interface {
	SetPersistentCache(PersistentEmbedCache)
}
