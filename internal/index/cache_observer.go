package index

// CacheObserver receives query-cache tier hit/miss events. Defined here
// (not in internal/metrics) so internal/index never imports internal/metrics
// — CLAUDE.md's existing separation. internal/metrics provides a concrete
// implementation structurally (Go interfaces need no shared import for
// that); see metrics.NewQueryCacheObserver.
type CacheObserver interface {
	// ObserveQueryCacheHit is called once per cache hit, tagged by which
	// tier served it: "l1", "l2", or "semantic" (Task 10).
	ObserveQueryCacheHit(tier string)
	// ObserveQueryCacheMiss is called once per cache miss (before the real
	// computation runs).
	ObserveQueryCacheMiss()
}

// noopCacheObserver is Engine's default — every call is free.
type noopCacheObserver struct{}

func (noopCacheObserver) ObserveQueryCacheHit(string) {}
func (noopCacheObserver) ObserveQueryCacheMiss()      {}

// SetCacheObserver installs o to receive this Engine's query-cache hit/miss
// events. Safe to call at any time; nil restores the default no-op.
func (e *Engine) SetCacheObserver(o CacheObserver) {
	if o == nil {
		o = noopCacheObserver{}
	}
	e.cacheObserver = o
}
