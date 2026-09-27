package storage

import (
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage/memtable"
)

// White-box tests for Engine.Get's tombstone handling (C2 regression). These
// bypass Open()/the WAL/flush pipeline entirely and manipulate the active and
// immutable MemTable slots directly, since C2 is purely about how Get walks
// that in-memory stack.

func TestEngineGet_TombstoneInActiveStopsSearch(t *testing.T) {
	e := &Engine{active: memtable.NewMemTable(1 << 20)}

	older := memtable.NewMemTable(1 << 20)
	if err := older.Put([]byte("k"), []byte("old-value")); err != nil {
		t.Fatal(err)
	}
	e.immutable = []*memtable.MemTable{older}

	if err := e.active.Delete([]byte("k")); err != nil {
		t.Fatal(err)
	}

	if val, ok := e.Get([]byte("k")); ok {
		t.Fatalf("expected tombstone in active table to hide the older immutable value, got (%q, true)", val)
	}
}

func TestEngineGet_FallsThroughToImmutableWhenAbsentInActive(t *testing.T) {
	e := &Engine{active: memtable.NewMemTable(1 << 20)}

	older := memtable.NewMemTable(1 << 20)
	if err := older.Put([]byte("k"), []byte("old-value")); err != nil {
		t.Fatal(err)
	}
	e.immutable = []*memtable.MemTable{older}

	val, ok := e.Get([]byte("k"))
	if !ok || string(val) != "old-value" {
		t.Fatalf("expected fallthrough to immutable table value, got (%q, %v)", val, ok)
	}
}

func TestEngineGet_ActiveValueShadowsOlderImmutable(t *testing.T) {
	e := &Engine{active: memtable.NewMemTable(1 << 20)}

	older := memtable.NewMemTable(1 << 20)
	if err := older.Put([]byte("k"), []byte("old-value")); err != nil {
		t.Fatal(err)
	}
	e.immutable = []*memtable.MemTable{older}

	if err := e.active.Put([]byte("k"), []byte("new-value")); err != nil {
		t.Fatal(err)
	}

	val, ok := e.Get([]byte("k"))
	if !ok || string(val) != "new-value" {
		t.Fatalf("expected active table's newer value, got (%q, %v)", val, ok)
	}
}
