package embedding

import (
	"context"
	"fmt"

	lru "github.com/hashicorp/golang-lru/v2"
)

type CachingEmbedder struct {
	base  Embedder
	cache *lru.Cache[string, []float32]
}

func NewCachingEmbedder(base Embedder, maxSize int) (*CachingEmbedder, error) {
	c, err := lru.New[string, []float32](maxSize)
	if err != nil {
		return nil, err
	}
	return &CachingEmbedder{
		base:  base,
		cache: c,
	}, nil
}

// cloneVec returns an independent copy of v. Both the cache's stored value
// and every value handed to a caller are copies, so neither the cache nor
// concurrent callers can corrupt each other's slice by mutating in place.
func cloneVec(v []float32) []float32 {
	if v == nil {
		return nil
	}
	out := make([]float32, len(v))
	copy(out, v)
	return out
}

func (c *CachingEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	if val, ok := c.cache.Get(text); ok {
		return cloneVec(val), nil
	}

	vec, err := c.base.Embed(ctx, text)
	// A nil/empty vector represents a failed embedding even when err is nil
	// (some embedders signal per-item failure that way). Never cache that —
	// it would poison the entry forever and the caller would keep getting a
	// silent "success" with nothing usable.
	if err == nil && len(vec) > 0 {
		c.cache.Add(text, cloneVec(vec))
	}
	return vec, err
}

func (c *CachingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	var missingIdx []int
	var missingTexts []string

	for i, txt := range texts {
		if val, ok := c.cache.Get(txt); ok {
			results[i] = cloneVec(val)
		} else {
			missingIdx = append(missingIdx, i)
			missingTexts = append(missingTexts, txt)
		}
	}

	if len(missingTexts) > 0 {
		missingVecs, err := c.base.EmbedBatch(ctx, missingTexts)
		if err != nil {
			return nil, err
		}
		if len(missingVecs) != len(missingTexts) {
			return nil, fmt.Errorf("embedding: base EmbedBatch returned %d vectors for %d requested texts", len(missingVecs), len(missingTexts))
		}
		for i, vec := range missingVecs {
			if len(vec) == 0 {
				// Per-item failure inside an otherwise-successful batch: leave
				// this slot nil and don't cache it, so the next call retries
				// the real embedder instead of being stuck with a fake hit.
				continue
			}
			ogIdx := missingIdx[i]
			results[ogIdx] = vec
			c.cache.Add(missingTexts[i], cloneVec(vec))
		}
	}

	return results, nil
}

func (c *CachingEmbedder) Dimensions() int {
	return c.base.Dimensions()
}

// Name forwards to the wrapped embedder if it implements Named, so callers
// see the underlying model's identity rather than the cache wrapper's.
func (c *CachingEmbedder) Name() string {
	if n, ok := c.base.(Named); ok {
		return n.Name()
	}
	return "unknown"
}
