package sstable

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage/memtable"
)

func TestReaderGet_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.sst")

	w, err := NewWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	entries := []memtable.Entry{
		{Key: []byte("alpha"), Value: []byte("1")},
		{Key: []byte("beta"), Value: []byte("2")},
		{Key: []byte("gone"), Deleted: true},
	}
	if err := w.WriteAll(entries); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	val, ok, err := r.Get([]byte("alpha"))
	if err != nil || !ok || string(val) != "1" {
		t.Fatalf("Get(alpha): val=%q ok=%v err=%v", val, ok, err)
	}
	if _, ok, err := r.Get([]byte("gone")); err != nil || ok {
		t.Fatalf("Get(gone): expected (false, nil), got ok=%v err=%v", ok, err)
	}
	if _, ok, err := r.Get([]byte("missing")); err != nil || ok {
		t.Fatalf("Get(missing): expected (false, nil), got ok=%v err=%v", ok, err)
	}
}

// Regression for H5: a corrupted block must surface as an error, never as a
// plain "not found" — the latter lets stale or already-deleted data at an
// older layer silently resurface.
func TestReaderGet_CorruptBlockReturnsErrorNotNotFound(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.sst")

	w, err := NewWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	entries := []memtable.Entry{
		{Key: []byte("alpha"), Value: []byte("1")},
		{Key: []byte("beta"), Value: []byte("2")},
	}
	if err := w.WriteAll(entries); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Flip a byte inside the first data block's payload (offset 10 is past
	// the 6-byte block header [CRC(4)+count(2)], well before bloom/index/footer).
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteAt([]byte{0xFF, 0xFF}, 10); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenReader(path)
	if err != nil {
		t.Fatalf("OpenReader should still succeed (only the data block is corrupt): %v", err)
	}
	defer r.Close()

	_, ok, err := r.Get([]byte("alpha"))
	if err == nil {
		t.Fatalf("expected a corruption error from Get, got ok=%v err=nil", ok)
	}
}

func TestReaderIterator_StreamsAllEntriesInOrder(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.sst")

	w, err := NewWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	want := []memtable.Entry{
		{Key: []byte("a"), Value: []byte("1")},
		{Key: []byte("b"), Deleted: true},
		{Key: []byte("c"), Value: []byte("3")},
	}
	if err := w.WriteAll(want); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	r, err := OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	it := r.NewIterator()
	var got []memtable.Entry
	for {
		e, ok, err := it.Next()
		if err != nil {
			t.Fatalf("iterator error: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, e)
	}

	if len(got) != len(want) {
		t.Fatalf("got %d entries, want %d", len(got), len(want))
	}
	for i := range want {
		if string(got[i].Key) != string(want[i].Key) || got[i].Deleted != want[i].Deleted {
			t.Errorf("entry %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}
