package index

import (
	"context"

	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// embedForSemanticScan computes the query's embedding for a semantic
// near-duplicate comparison. This pays one embedding call up front, before
// the exact-key cache's normal (overlapped-with-lexical-phase) embedding
// happens inside searchUncached on a miss — but embedding.CachingEmbedder's
// own query cache (internal/embedding/cache.go) makes that second call a
// cheap cache hit, not a real recomputation, so the net cost is one real
// embedding call either way.
func (e *Engine) embedForSemanticScan(ctx context.Context, query string) ([]float32, bool) {
	v, err := embedding.EmbedQuery(ctx, e.embedder, query)
	if err != nil || len(v) == 0 {
		return nil, false
	}
	return normalizeVector(v), true
}

// semanticScan looks for a cached entry in the same bucket (same filter spec
// and weights — see cache_key.go's bucketKey) whose query embedding is
// within Config.QueryCacheSemanticThreshold cosine similarity of queryVec.
// Both vectors are pre-normalized (normalizeVector), so a plain dot product
// is the cosine similarity. Only ever scans L1 — L2 (Redis) is exact-match
// only, since scanning a remote store's full contents on every miss would
// defeat the point of a cache.
func (e *Engine) semanticScan(bucket string, queryVec []float32) (cacheResult, bool) {
	e.semanticScans.Add(1)
	threshold := e.config.QueryCacheSemanticThreshold
	var best cacheResult
	bestSim := -1.0
	e.cache.RangeL1(func(_ string, val cacheResult) bool {
		if val.BucketKey != bucket || len(val.QueryVec) == 0 {
			return true
		}
		if sim := ranking.DotProduct(queryVec, val.QueryVec); sim > bestSim {
			bestSim, best = sim, val
		}
		return true
	})
	return best, bestSim >= threshold
}
