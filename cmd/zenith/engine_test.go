package main

import (
	"context"
	"path/filepath"
	"testing"
)

func TestBuildEngine_PersistentCacheSurvivesAcrossCycles(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1 // keep config default

	eng1, _, teardown1, err := buildEngine(false)
	if err != nil {
		t.Fatalf("buildEngine (first): %v", err)
	}
	if err := eng1.Add(context.Background(), "doc1", "the quick brown fox"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	teardown1()

	eng2, _, teardown2, err := buildEngine(true)
	if err != nil {
		t.Fatalf("buildEngine (second, load): %v", err)
	}
	defer teardown2()
	if _, ok := eng2.GetText("doc1"); !ok {
		t.Error("expected doc1 to survive the reload via journal replay")
	}
}
