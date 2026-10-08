package querycache

import (
	"bytes"
	"context"
	"encoding/gob"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

func encodeInt(v int) ([]byte, error) {
	var buf bytes.Buffer
	err := gob.NewEncoder(&buf).Encode(v)
	return buf.Bytes(), err
}

func decodeInt(b []byte) (int, error) {
	var v int
	err := gob.NewDecoder(bytes.NewReader(b)).Decode(&v)
	return v, err
}

func TestTiered_L1OnlyWhenNoL2Configured(t *testing.T) {
	l1, err := NewMemCache[int](10)
	if err != nil {
		t.Fatal(err)
	}
	tr := NewTiered[int](l1, nil, encodeInt, decodeInt)
	ctx := context.Background()

	tr.Set(ctx, "a", 42)
	v, ok := tr.Get(ctx, "a")
	if !ok || v != 42 {
		t.Fatalf("Get(a) = %v, %v; want 42, true", v, ok)
	}
}

func TestTiered_L1PreferredOverL2(t *testing.T) {
	mr := miniredis.RunT(t)
	l1, _ := NewMemCache[int](10)
	l2 := NewBytesCache(mr.Addr(), time.Minute)
	defer l2.Close()
	tr := NewTiered[int](l1, l2, encodeInt, decodeInt)
	ctx := context.Background()

	tr.Set(ctx, "a", 1)
	// Directly poison L2 behind Tiered's back to prove a hit comes from L1,
	// not a decode of the (different) L2 value.
	encoded, _ := encodeInt(999)
	l2.Set(ctx, "a", encoded)

	v, ok := tr.Get(ctx, "a")
	if !ok || v != 1 {
		t.Fatalf("Get(a) = %v, %v; want 1 (from L1), true", v, ok)
	}
}

func TestTiered_L2HitBackfillsL1(t *testing.T) {
	mr := miniredis.RunT(t)
	l1, _ := NewMemCache[int](10)
	l2 := NewBytesCache(mr.Addr(), time.Minute)
	defer l2.Close()
	ctx := context.Background()

	// Populate L2 only (simulating another process having cached it).
	encoded, _ := encodeInt(7)
	l2.Set(ctx, "a", encoded)

	tr := NewTiered[int](l1, l2, encodeInt, decodeInt)
	v, ok := tr.Get(ctx, "a")
	if !ok || v != 7 {
		t.Fatalf("Get(a) = %v, %v; want 7 (from L2), true", v, ok)
	}

	// L1 must now hold it too: kill L2 and confirm the second Get still hits.
	l2.Close()
	v2, ok2 := tr.Get(ctx, "a")
	if !ok2 || v2 != 7 {
		t.Fatalf("second Get(a) after L2 died = %v, %v; want 7 (backfilled into L1), true", v2, ok2)
	}
}

func TestTiered_MissOnBoth(t *testing.T) {
	l1, _ := NewMemCache[int](10)
	tr := NewTiered[int](l1, nil, encodeInt, decodeInt)
	if _, ok := tr.Get(context.Background(), "missing"); ok {
		t.Fatal("Get(missing) = _, true; want false")
	}
}

func TestTiered_SetWritesBothTiers(t *testing.T) {
	mr := miniredis.RunT(t)
	l1, _ := NewMemCache[int](10)
	l2 := NewBytesCache(mr.Addr(), time.Minute)
	defer l2.Close()
	tr := NewTiered[int](l1, l2, encodeInt, decodeInt)
	ctx := context.Background()

	tr.Set(ctx, "a", 5)
	if v, ok := l1.Get(ctx, "a"); !ok || v != 5 {
		t.Fatalf("L1 after Set = %v, %v; want 5, true", v, ok)
	}
	if raw, ok := l2.Get(ctx, "a"); !ok {
		t.Fatal("L2 after Set has no entry")
	} else if v, err := decodeInt(raw); err != nil || v != 5 {
		t.Fatalf("L2 after Set decodes to %v, %v; want 5, nil", v, err)
	}
}

func TestTiered_RangeL1(t *testing.T) {
	l1, _ := NewMemCache[int](10)
	tr := NewTiered[int](l1, nil, encodeInt, decodeInt)
	ctx := context.Background()
	tr.Set(ctx, "a", 1)
	tr.Set(ctx, "b", 2)

	seen := map[string]int{}
	tr.RangeL1(func(k string, v int) bool {
		seen[k] = v
		return true
	})
	if len(seen) != 2 {
		t.Fatalf("RangeL1 visited %v, want 2 entries", seen)
	}
}
