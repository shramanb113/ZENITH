package embedding

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log/slog"

	lru "github.com/hashicorp/golang-lru/v2"
	"golang.org/x/sync/singleflight"
)

type CachingEmbedder struct {
	base  Embedder
	cache *lru.Cache[string, []float32]

	// queryCache is a second, separate LRU, not a second lookup into cache:
	// an asymmetric model (e.g. BGE's query instruction prefix) feeds
	// Embed(text) and EmbedQuery(text) different model input for the same
	// raw text, so the two must never share a cache keyed on that text.
	queryCache *lru.Cache[string, []float32]

	// embedSF/querySF de-duplicate concurrent misses for the same text: N
	// goroutines racing on the same uncached text would otherwise each pay
	// the full embedder cost independently — the dominant cost in practice
	// being the process-wide ONNX inference mutex shared by every caller
	// (internal/localembedder/model.go), not this cache. Two separate
	// singleflight.Group values rather than one keyed by a prefixed string,
	// so a cache hit (the common case) never pays a string-concatenation
	// allocation just to form a dedup key it won't use.
	embedSF, querySF singleflight.Group

	// persist is the optional second cache tier, checked on an LRU miss
	// before the base embedder, and populated on a real embed. nil (the
	// default for every existing construction site) disables it entirely —
	// zero behavior change for any caller that doesn't opt in.
	persist PersistentEmbedCache
}

// SetPersistentCache wires a crash-surviving second cache tier beneath this
// embedder's in-memory LRUs. Call once after construction; nil (the default)
// disables the tier.
func (c *CachingEmbedder) SetPersistentCache(p PersistentEmbedCache) {
	c.persist = p
}

// persistentKey builds the cache key for text under domain ('D' for
// documents via Embed/EmbedBatch, 'Q' for queries via EmbedQuery). The
// embedder identity is a literal prefix (not hashed into the digest) so a
// future per-model prune can prefix-scan without decoding every value.
func (c *CachingEmbedder) persistentKey(domain byte, text string) []byte {
	sum := sha256.Sum256([]byte(text))
	name := c.Name()
	key := make([]byte, 0, len(name)+1+1+16)
	key = append(key, []byte(name)...)
	key = append(key, 0)
	key = append(key, domain)
	key = append(key, sum[:16]...)
	return key
}

func NewCachingEmbedder(base Embedder, maxSize int) (*CachingEmbedder, error) {
	c, err := lru.New[string, []float32](maxSize)
	if err != nil {
		return nil, err
	}
	qc, err := lru.New[string, []float32](maxSize)
	if err != nil {
		return nil, err
	}
	return &CachingEmbedder{
		base:       base,
		cache:      c,
		queryCache: qc,
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
	if c.persist != nil {
		if val, ok := c.persist.GetEmbedding(c.persistentKey('D', text)); ok && len(val) == c.Dimensions() {
			c.cache.Add(text, cloneVec(val))
			return cloneVec(val), nil
		}
	}

	v, err, _ := c.embedSF.Do(text, func() (any, error) {
		vec, err := c.base.Embed(ctx, text)
		// A nil/empty vector represents a failed embedding even when err is
		// nil (some embedders signal per-item failure that way). Never cache
		// that — it would poison the entry forever and the caller would
		// keep getting a silent "success" with nothing usable.
		if err == nil && len(vec) > 0 {
			c.cache.Add(text, cloneVec(vec))
			if c.persist != nil {
				if perr := c.persist.PutEmbedding(c.persistentKey('D', text), vec); perr != nil {
					slog.Warn("embedding: persistent cache write failed", "error", perr)
				}
			}
		}
		return vec, err
	})
	if err != nil {
		return nil, err
	}
	// Every caller sharing this in-flight Do (including the one that
	// triggered it) gets the same underlying slice back from singleflight —
	// clone before returning so one caller mutating its copy can't corrupt
	// another's, the same invariant cloneVec already gives every cache hit.
	return cloneVec(v.([]float32)), nil
}

// EmbedBatch has no singleflight dedup, unlike Embed/EmbedQuery above — a
// deliberate scope decision, not an oversight. Deduping per missing text
// here would force each one through its own single-text call, destroying
// the N-texts-in-one-model-call amortization that is EmbedBatch's whole
// purpose; a correct version would need to merge missing texts *across*
// concurrent EmbedBatch calls before batching, a materially bigger feature.
// It also targets a workload (many literally-identical texts landing in
// concurrent document-ingestion batches) that TestConcurrentQPS's measured
// bottleneck — concurrent identical *queries*, via EmbedQuery above — does
// not exercise; revisit only if ingestion-side measurement shows it matters.
func (c *CachingEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	var missingIdx []int
	var missingTexts []string

	for i, txt := range texts {
		if val, ok := c.cache.Get(txt); ok {
			results[i] = cloneVec(val)
			continue
		}
		if c.persist != nil {
			if val, ok := c.persist.GetEmbedding(c.persistentKey('D', txt)); ok && len(val) == c.Dimensions() {
				results[i] = cloneVec(val)
				c.cache.Add(txt, cloneVec(val))
				continue
			}
		}
		missingIdx = append(missingIdx, i)
		missingTexts = append(missingTexts, txt)
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
			if c.persist != nil {
				if perr := c.persist.PutEmbedding(c.persistentKey('D', missingTexts[i]), vec); perr != nil {
					slog.Warn("embedding: persistent cache write failed", "error", perr)
				}
			}
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

// EmbedQuery forwards to the wrapped embedder's query mode when it has one,
// through queryCache — a separate cache/dedup path from Embed's (see
// queryCache's doc comment). Caching queries pays off specifically under
// concurrent load: many callers searching the same popular or retried query
// at once previously each paid the full embedder cost independently (the
// scenario internal/sidecar's TestConcurrentQPS measures), instead of one
// real call plus cheap cache hits. Symmetric models (no QueryEmbedder) fall
// through to the now-deduplicated Embed above, so they gain the same benefit.
func (c *CachingEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	q, ok := c.base.(QueryEmbedder)
	if !ok {
		return c.Embed(ctx, text)
	}
	if val, ok := c.queryCache.Get(text); ok {
		return cloneVec(val), nil
	}
	if c.persist != nil {
		if val, ok := c.persist.GetEmbedding(c.persistentKey('Q', text)); ok && len(val) == c.Dimensions() {
			c.queryCache.Add(text, cloneVec(val))
			return cloneVec(val), nil
		}
	}
	v, err, _ := c.querySF.Do(text, func() (any, error) {
		vec, err := q.EmbedQuery(ctx, text)
		if err == nil && len(vec) > 0 {
			c.queryCache.Add(text, cloneVec(vec))
			if c.persist != nil {
				if perr := c.persist.PutEmbedding(c.persistentKey('Q', text), vec); perr != nil {
					slog.Warn("embedding: persistent cache write failed", "error", perr)
				}
			}
		}
		return vec, err
	})
	if err != nil {
		return nil, err
	}
	return cloneVec(v.([]float32)), nil
}
