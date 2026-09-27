package storage_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	storage "github.com/shramanb113/ZENITH/internal/storage"
	"github.com/shramanb113/ZENITH/internal/storage/sstable"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)


func TestStorageEngine_RecordsAfterOpen(t *testing.T) {
	dir := t.TempDir()
	cfg := storage.DefaultEngineConfig()
	cfg.WALPath = filepath.Join(dir, "zenith.wal")
	cfg.WALConfig.Dir = dir
	cfg.SSTDir = filepath.Join(dir, "sst")
	cfg.FSTPath = ""

	ctx := context.Background()

	// Phase 1: write two documents via Put.
	eng1, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open phase1: %v", err)
	}
	if err := eng1.Put(ctx, []byte("doc1"), []byte("hello world")); err != nil {
		t.Fatalf("Put doc1: %v", err)
	}
	if err := eng1.Put(ctx, []byte("doc2"), []byte("foo bar")); err != nil {
		t.Fatalf("Put doc2: %v", err)
	}
	// Close without checkpoint — WAL retains the records.
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close phase1: %v", err)
	}

	// Phase 2: reopen — Records() must return the two documents.
	eng2, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open phase2: %v", err)
	}
	defer eng2.Close()

	records := eng2.Records()
	if len(records) != 2 {
		t.Fatalf("expected 2 records after reopen, got %d", len(records))
	}
	ids := map[string]string{}
	for _, r := range records {
		if r.Op == wal.OpTypePut {
			ids[string(r.Key)] = string(r.Value)
		}
	}
	if ids["doc1"] != "hello world" {
		t.Errorf("doc1 value mismatch: %q", ids["doc1"])
	}
	if ids["doc2"] != "foo bar" {
		t.Errorf("doc2 value mismatch: %q", ids["doc2"])
	}
}

func TestStorageEngine_CheckpointClearsJournal(t *testing.T) {
	dir := t.TempDir()
	cfg := storage.DefaultEngineConfig()
	cfg.WALPath = filepath.Join(dir, "zenith.wal")
	cfg.WALConfig.Dir = dir
	cfg.SSTDir = filepath.Join(dir, "sst")
	cfg.FSTPath = ""

	ctx := context.Background()

	eng1, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := eng1.Put(ctx, []byte("doc1"), []byte("content")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	// Checkpoint — WAL must be reset.
	if err := eng1.Checkpoint(); err != nil {
		t.Fatalf("Checkpoint: %v", err)
	}
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen — Records() must be empty (WAL was reset).
	eng2, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("reopen after checkpoint: %v", err)
	}
	defer eng2.Close()

	if got := eng2.Records(); len(got) != 0 {
		t.Fatalf("expected 0 records after Checkpoint, got %d", len(got))
	}
}

// Regression for C3: WAL replay on Open used to apply every record into a
// single MemTable with no rotation, so replaying more data than
// MemTableMaxSize left the active table permanently frozen — every write
// after Open failed forever.
func TestStorageEngine_ReplayOverflowDoesNotPermanentlyFreeze(t *testing.T) {
	dir := t.TempDir()
	cfg := storage.DefaultEngineConfig()
	cfg.WALPath = filepath.Join(dir, "zenith.wal")
	cfg.WALConfig.Dir = dir
	cfg.SSTDir = filepath.Join(dir, "sst")
	cfg.FSTPath = ""
	cfg.MemTableMaxSize = 32 // tiny — a handful of records overflow it

	ctx := context.Background()

	eng1, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open phase1: %v", err)
	}
	for i := 0; i < 20; i++ {
		key := []byte(fmt.Sprintf("key-%02d", i))
		if err := eng1.Put(ctx, key, []byte("value")); err != nil {
			t.Fatalf("Put #%d (phase1): %v", i, err)
		}
	}
	// Close WITHOUT checkpointing so the WAL retains all 20 records for replay.
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close phase1: %v", err)
	}

	eng2, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open phase2 (replay): %v", err)
	}
	defer eng2.Close()

	if err := eng2.Put(ctx, []byte("after-replay"), []byte("must-succeed")); err != nil {
		t.Fatalf("Put after replay overflow: %v", err)
	}
	val, ok := eng2.Get([]byte("after-replay"))
	if !ok || string(val) != "must-succeed" {
		t.Fatalf("expected to read back the post-replay write, got (%q, %v)", val, ok)
	}
}

// Regression for C6: without SSTable rediscovery on Open, sstCounter and the
// compactor's level state both start empty on every restart, so the next
// flush reuses (and truncates) the previous run's first SSTable file name.
func TestStorageEngine_RestartDoesNotOverwriteExistingSSTable(t *testing.T) {
	dir := t.TempDir()
	cfg := storage.DefaultEngineConfig()
	cfg.WALPath = filepath.Join(dir, "zenith.wal")
	cfg.WALConfig.Dir = dir
	cfg.SSTDir = filepath.Join(dir, "sst")
	cfg.FSTPath = ""
	cfg.MemTableMaxSize = 1 // freeze (and flush) on the very first write

	ctx := context.Background()

	eng1, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open 1: %v", err)
	}
	if err := eng1.Put(ctx, []byte("a"), []byte("first-run")); err != nil {
		t.Fatalf("Put a: %v", err)
	}
	// Checkpoint so the second run doesn't replay "a" again (which, at
	// MemTableMaxSize=1, would itself trigger an extra flush and confuse the
	// file count this test is asserting on).
	if err := eng1.Checkpoint(); err != nil {
		t.Fatalf("Checkpoint 1: %v", err)
	}
	if err := eng1.Close(); err != nil {
		t.Fatalf("Close 1: %v", err)
	}

	files1, _ := filepath.Glob(filepath.Join(cfg.SSTDir, "*.sst"))
	if len(files1) != 1 {
		t.Fatalf("expected 1 sstable after first run, got %d: %v", len(files1), files1)
	}

	eng2, err := storage.Open(cfg)
	if err != nil {
		t.Fatalf("Open 2: %v", err)
	}
	if err := eng2.Put(ctx, []byte("b"), []byte("second-run")); err != nil {
		t.Fatalf("Put b: %v", err)
	}
	if err := eng2.Close(); err != nil {
		t.Fatalf("Close 2: %v", err)
	}

	files2, _ := filepath.Glob(filepath.Join(cfg.SSTDir, "*.sst"))
	if len(files2) != 2 {
		t.Fatalf("expected 2 distinct sstables after second run (no overwrite), got %d: %v", len(files2), files2)
	}

	r, err := sstable.OpenReader(files1[0])
	if err != nil {
		t.Fatalf("open first-run sstable after restart: %v", err)
	}
	defer r.Close()
	val, ok, err := r.Get([]byte("a"))
	if err != nil {
		t.Fatalf("read first-run sstable after restart: %v", err)
	}
	if !ok || string(val) != "first-run" {
		t.Fatalf("first-run sstable was overwritten by the second run: got (%q, %v)", val, ok)
	}
}
