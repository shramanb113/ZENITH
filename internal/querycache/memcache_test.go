package querycache

import (
	"context"
	"testing"
)

func TestMemCache_SetThenGet(t *testing.T) {
	c, err := NewMemCache[int](10)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c.Set(ctx, "a", 42)
	v, ok := c.Get(ctx, "a")
	if !ok || v != 42 {
		t.Fatalf("Get(a) = %v, %v; want 42, true", v, ok)
	}
	if _, ok := c.Get(ctx, "missing"); ok {
		t.Fatalf("Get(missing) = _, true; want false")
	}
}

func TestMemCache_EvictsOldestBeyondSize(t *testing.T) {
	c, err := NewMemCache[int](2)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c.Set(ctx, "a", 1)
	c.Set(ctx, "b", 2)
	c.Set(ctx, "c", 3) // evicts "a" (least recently used)
	if _, ok := c.Get(ctx, "a"); ok {
		t.Fatalf("Get(a) = _, true after eviction; want false")
	}
	if v, ok := c.Get(ctx, "b"); !ok || v != 2 {
		t.Fatalf("Get(b) = %v, %v; want 2, true", v, ok)
	}
	if v, ok := c.Get(ctx, "c"); !ok || v != 3 {
		t.Fatalf("Get(c) = %v, %v; want 3, true", v, ok)
	}
}

func TestMemCache_Range(t *testing.T) {
	c, err := NewMemCache[int](10)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c.Set(ctx, "a", 1)
	c.Set(ctx, "b", 2)
	seen := map[string]int{}
	c.Range(func(k string, v int) bool {
		seen[k] = v
		return true
	})
	if len(seen) != 2 || seen["a"] != 1 || seen["b"] != 2 {
		t.Fatalf("Range visited %v, want {a:1 b:2}", seen)
	}
}

func TestMemCache_RangeStopsEarly(t *testing.T) {
	c, err := NewMemCache[int](10)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	c.Set(ctx, "a", 1)
	c.Set(ctx, "b", 2)
	calls := 0
	c.Range(func(k string, v int) bool {
		calls++
		return false
	})
	if calls != 1 {
		t.Fatalf("Range called fn %d times after returning false once, want 1", calls)
	}
}

func TestNewMemCache_RejectsNonPositiveSize(t *testing.T) {
	if _, err := NewMemCache[int](0); err == nil {
		t.Fatal("NewMemCache(0) succeeded, want an error")
	}
}
