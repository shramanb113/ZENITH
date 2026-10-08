package main

import (
	"context"
	"path/filepath"
	"testing"
)

func TestStorageInspectCmd_FindsJournaledDoc(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	eng, storageEng, _, _, err := buildEngine(false)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	if err := eng.Add(context.Background(), "doc1", "some text"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	// Close only the storage engine (releasing Pebble's directory lock so
	// inspect's own storage.Open can succeed) without running the full
	// teardown's Save+Prune, which would remove the journal entry this
	// test is about to look for.
	if err := storageEng.Close(); err != nil {
		t.Fatalf("storageEng.Close: %v", err)
	}
	_ = eng.Close()

	found, err := runStorageInspect(nil, []string{"doc1"})
	if err != nil {
		t.Fatalf("runStorageInspect: %v", err)
	}
	if !found {
		t.Error("expected doc1 to be found in the journal")
	}
}

func TestStorageInspectCmd_MissingDocReturnsFalse(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	_, storageEng, _, _, err := buildEngine(false)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	if err := storageEng.Close(); err != nil {
		t.Fatalf("storageEng.Close: %v", err)
	}

	found, err := runStorageInspect(nil, []string{"never-written"})
	if err != nil {
		t.Fatalf("runStorageInspect: %v", err)
	}
	if found {
		t.Error("expected a miss for an id that was never journaled")
	}
}

func TestStoragePruneCmd_RunsWithoutError(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	eng, _, _, teardown, err := buildEngine(false)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	if err := eng.Add(context.Background(), "doc1", "some text"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	teardown()

	if err := runStoragePrune(nil, nil); err != nil {
		t.Fatalf("runStoragePrune: %v", err)
	}
}
