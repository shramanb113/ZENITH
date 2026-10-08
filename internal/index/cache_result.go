package index

import (
	"bytes"
	"encoding/gob"
)

// cacheResult is the value type internal/querycache.Tiered caches: the
// search results plus the query embedding (used only by Task 10's semantic
// near-duplicate scan) and the bucket this entry belongs to (used by that
// same scan to restrict comparisons to entries sharing the same filter and
// weights — see cache_key.go's bucketKey).
type cacheResult struct {
	Results   []SearchResponse
	QueryVec  []float32
	BucketKey string
}

func encodeCacheResult(v cacheResult) ([]byte, error) {
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(v); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeCacheResult(b []byte) (cacheResult, error) {
	var v cacheResult
	err := gob.NewDecoder(bytes.NewReader(b)).Decode(&v)
	return v, err
}

// cloneResponses returns an independent copy of results. Every value handed
// to a Search caller from the cache is a copy, so a caller mutating it in
// place (Engine.SortByAttribute does exactly this: `copy(results, sorted)`
// directly into the slice the caller passed it) can never corrupt a cached
// entry's backing array.
func cloneResponses(results []SearchResponse) []SearchResponse {
	if results == nil {
		return nil
	}
	out := make([]SearchResponse, len(results))
	copy(out, results)
	return out
}
