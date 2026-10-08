package querycache

import "context"

// Tiered composes an L1 MemCache with an optional L2 BytesCache. Get checks
// L1 first (no decode cost); on an L1 miss with L2 configured, it decodes an
// L2 hit, backfills L1 with the decoded value, and returns it. Set writes L1
// directly and encodes once for L2. l2 may be nil (L2 disabled): Tiered then
// behaves exactly like L1 alone, at zero extra cost.
//
// Get/Set take a context.Context — wider than the plain Cache[V] shape L1
// satisfies alone, because L2's Redis calls need one to respect caller
// cancellation/timeouts; MemCache accepts and ignores it (see memcache.go).
type Tiered[V any] struct {
	l1     *MemCache[V]
	l2     *BytesCache
	encode func(V) ([]byte, error)
	decode func([]byte) (V, error)
}

// NewTiered composes l1 (required) and l2 (nil disables L2). encode/decode
// are used only for the L2 round-trip — never on an L1 hit.
func NewTiered[V any](l1 *MemCache[V], l2 *BytesCache, encode func(V) ([]byte, error), decode func([]byte) (V, error)) *Tiered[V] {
	return &Tiered[V]{l1: l1, l2: l2, encode: encode, decode: decode}
}

func (t *Tiered[V]) Get(ctx context.Context, key string) (V, bool) {
	if v, ok := t.l1.Get(ctx, key); ok {
		return v, true
	}
	if t.l2 == nil {
		var zero V
		return zero, false
	}
	raw, ok := t.l2.Get(ctx, key)
	if !ok {
		var zero V
		return zero, false
	}
	v, err := t.decode(raw)
	if err != nil {
		// A binary upgrade can leave stale-shaped entries in Redis from an
		// older version — treat as a miss, never an error; the next Set
		// below (from the caller recomputing) overwrites it.
		var zero V
		return zero, false
	}
	t.l1.Set(ctx, key, v)
	return v, true
}

func (t *Tiered[V]) Set(ctx context.Context, key string, val V) {
	t.l1.Set(ctx, key, val)
	if t.l2 == nil {
		return
	}
	raw, err := t.encode(val)
	if err != nil {
		return
	}
	t.l2.Set(ctx, key, raw)
}

// RangeL1 calls fn for every entry currently in the L1 tier, stopping early
// if fn returns false. L2 is never ranged — see MemCache.Range's doc comment
// for why this exists (Engine's semantic near-duplicate scan).
func (t *Tiered[V]) RangeL1(fn func(key string, val V) bool) {
	t.l1.Range(fn)
}
