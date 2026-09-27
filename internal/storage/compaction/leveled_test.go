package compaction

import (
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage/memtable"
	"github.com/shramanb113/ZENITH/internal/storage/sstable"
)

func writeTestSSTable(t *testing.T, dir, name string, entries []memtable.Entry) string {
	t.Helper()
	path := filepath.Join(dir, name)
	w, err := sstable.NewWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.WriteAll(entries); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// Regression for C1: L0 compaction used to keep the OLDEST version of a
// duplicate key (because it sorted only by level, and used slice-append
// order — oldest-first — as the tiebreak among same-level files). Two L0
// files can legitimately share a key when it was written, flushed, then
// updated and flushed again before compaction ran.
func TestMergeSSTableS_NewestSeqWinsOnDuplicateKey(t *testing.T) {
	dir := t.TempDir()
	older := writeTestSSTable(t, dir, "older.sst", []memtable.Entry{{Key: []byte("dup"), Value: []byte("old-value")}})
	newer := writeTestSSTable(t, dir, "newer.sst", []memtable.Entry{{Key: []byte("dup"), Value: []byte("new-value")}})

	c := NewCompactor(CompactorConfig{Dir: dir, MaxLevels: 7})

	// Inputs deliberately given in OLDEST-first slice order (as AddSSTable
	// would naturally append them) so a slice-order-based tiebreak would
	// pick the wrong (old) value; Seq is the only thing that should matter.
	inputs := []*SSTableMeta{
		{Path: older, MinKey: []byte("dup"), MaxKey: []byte("dup"), Level: 0, Seq: 1},
		{Path: newer, MinKey: []byte("dup"), MaxKey: []byte("dup"), Level: 0, Seq: 2},
	}

	outputs, err := c.mergeSSTableS(inputs, 1)
	if err != nil {
		t.Fatalf("mergeSSTableS: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("expected 1 output SSTable, got %d", len(outputs))
	}
	if outputs[0].Seq != 2 {
		t.Errorf("expected output Seq to carry forward the max input Seq (2), got %d", outputs[0].Seq)
	}

	r, err := sstable.OpenReader(outputs[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	val, ok, err := r.Get([]byte("dup"))
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected key 'dup' to be present in the merged output")
	}
	if string(val) != "new-value" {
		t.Fatalf("expected the newest (highest-Seq) value to win, got %q", val)
	}
}

func TestMergeSSTableS_DropsTombstonesAtLastLevel(t *testing.T) {
	dir := t.TempDir()
	path := writeTestSSTable(t, dir, "in.sst", []memtable.Entry{
		{Key: []byte("gone"), Deleted: true},
		{Key: []byte("kept"), Value: []byte("v")},
	})

	c := NewCompactor(CompactorConfig{Dir: dir, MaxLevels: 2})
	inputs := []*SSTableMeta{{Path: path, MinKey: []byte("gone"), MaxKey: []byte("kept"), Level: 0, Seq: 1}}

	// MaxLevels=2 means level 1 is the last level (sink).
	outputs, err := c.mergeSSTableS(inputs, 1)
	if err != nil {
		t.Fatalf("mergeSSTableS: %v", err)
	}
	if len(outputs) != 1 {
		t.Fatalf("expected 1 output, got %d", len(outputs))
	}

	r, err := sstable.OpenReader(outputs[0].Path)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if _, ok, _ := r.Get([]byte("gone")); ok {
		t.Error("expected tombstone to be dropped at the last level")
	}
	if val, ok, err := r.Get([]byte("kept")); err != nil || !ok || string(val) != "v" {
		t.Errorf("expected live key to survive, got val=%q ok=%v err=%v", val, ok, err)
	}
}

// Manifest round-trip: Snapshot/LoadSnapshot must reproduce the same level
// layout (used by the storage Engine to rediscover SSTables on restart — C6).
func TestCompactor_SnapshotLoadSnapshotRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := writeTestSSTable(t, dir, "a.sst", []memtable.Entry{{Key: []byte("k"), Value: []byte("v")}})

	c1 := NewCompactor(CompactorConfig{Dir: dir, MaxLevels: 7})
	c1.AddSSTable(&SSTableMeta{Path: path, MinKey: []byte("k"), MaxKey: []byte("k"), Level: 0, Seq: 5})

	snap := c1.Snapshot()
	if len(snap) != 1 || snap[0].Path != path || snap[0].Level != 0 || snap[0].Seq != 5 {
		t.Fatalf("unexpected snapshot: %+v", snap)
	}

	c2 := NewCompactor(CompactorConfig{Dir: dir, MaxLevels: 7})
	c2.LoadSnapshot(snap, func(p string) ([]byte, []byte, int64, error) {
		r, err := sstable.OpenReader(p)
		if err != nil {
			return nil, nil, 0, err
		}
		defer r.Close()
		minKey, err := r.MinKey()
		if err != nil {
			return nil, nil, 0, err
		}
		return minKey, r.MaxKey(), 0, nil
	})

	snap2 := c2.Snapshot()
	if len(snap2) != 1 || snap2[0].Path != path || snap2[0].Seq != 5 {
		t.Fatalf("LoadSnapshot did not reproduce the level state: %+v", snap2)
	}
}
