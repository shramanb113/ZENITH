// Package querycache provides a generic, two-tier cache: an in-process L1
// (MemCache) in front of an optional Redis-backed L2 (BytesCache), composed
// by Tiered. It has no dependency on internal/index — any Go value type can
// be cached, and the package stays reusable with no import-cycle risk.
package querycache

import (
	"context"

	lru "github.com/hashicorp/golang-lru/v2"
)

// MemCache is an in-process LRU cache of native Go values — no
// serialization on the hot path. ctx is accepted (to satisfy Cache[V]
// uniformly with the Redis-backed tier) but ignored: an in-process map
// lookup has nothing to cancel.
type MemCache[V any] struct {
	lru *lru.Cache[string, V]
}

// NewMemCache constructs an L1 cache holding at most size entries, evicting
// the least recently used entry once full. size must be > 0.
func NewMemCache[V any](size int) (*MemCache[V], error) {
	c, err := lru.New[string, V](size)
	if err != nil {
		return nil, err
	}
	return &MemCache[V]{lru: c}, nil
}

func (m *MemCache[V]) Get(_ context.Context, key string) (V, bool) {
	return m.lru.Get(key)
}

func (m *MemCache[V]) Set(_ context.Context, key string, val V) {
	m.lru.Add(key, val)
}

// Range calls fn for every entry currently in the cache, stopping early if
// fn returns false. Used only by Engine's semantic near-duplicate scan
// (internal/index); not part of the Cache[V] interface, since no other
// caller needs to enumerate a cache's contents.
func (m *MemCache[V]) Range(fn func(key string, val V) bool) {
	for _, k := range m.lru.Keys() {
		v, ok := m.lru.Peek(k)
		if !ok {
			continue
		}
		if !fn(k, v) {
			return
		}
	}
}
