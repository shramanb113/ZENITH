package metrics

// queryCacheObserver implements index.CacheObserver structurally — no
// import of internal/index in either direction (see index.CacheObserver's
// doc comment for why).
type queryCacheObserver struct{}

// NewQueryCacheObserver returns an index.CacheObserver (by structural
// typing) backed by the zenith_query_cache_hits_total/..._misses_total
// series. Wrap it onto an Engine the same way InstrumentEmbedder wraps an
// Embedder: unconditionally, at construction time — cheap even when nobody
// scrapes /metrics.
func NewQueryCacheObserver() queryCacheObserver {
	return queryCacheObserver{}
}

func (queryCacheObserver) ObserveQueryCacheHit(tier string) {
	QueryCacheHitsTotal.WithLabelValues(tier).Inc()
}

func (queryCacheObserver) ObserveQueryCacheMiss() {
	QueryCacheMissesTotal.Inc()
}
