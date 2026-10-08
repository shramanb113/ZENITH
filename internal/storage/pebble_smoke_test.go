package storage

import (
	"testing"

	"github.com/cockroachdb/pebble"
)

func TestPebbleSmoke_OpenSetGetClose(t *testing.T) {
	dir := t.TempDir()
	db, err := pebble.Open(dir, &pebble.Options{})
	if err != nil {
		t.Fatalf("pebble.Open: %v", err)
	}
	defer db.Close()

	if err := db.Set([]byte("k"), []byte("v"), pebble.Sync); err != nil {
		t.Fatalf("Set: %v", err)
	}
	val, closer, err := db.Get([]byte("k"))
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	defer closer.Close()
	if string(val) != "v" {
		t.Fatalf("Get returned %q, want %q", val, "v")
	}
}
