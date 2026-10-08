package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// withTestCLIFlags points cliFlags at a fresh temp directory for one test
// and restores the previous global values on cleanup. cliFlags is a
// package-level global shared by every command; a test that mutates it
// without restoring leaks state into whichever test runs next (flagged by
// review) — every test in this package that touches cliFlags should use
// this instead of assigning fields directly.
func withTestCLIFlags(t *testing.T) (dir string) {
	t.Helper()
	dir = t.TempDir()
	prev := cliFlags
	t.Cleanup(func() { cliFlags = prev })

	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.ollamaURL = ""
	cliFlags.ollamaModel = ""
	cliFlags.queryCacheSize = -1
	return dir
}

// TestBuildEngine_IndexSurvivesSaveLoadCycle proves buildEngine's storage
// wiring doesn't break the ordinary Save/Load round trip. It uses the
// deterministic embedder, which always fails to embed (returns
// ErrEmbeddingUnavailable with a nil vector) — so this test exercises
// neither journal-vector-carry-on-replay nor the persistent embedding
// cache; see TestBuildEngine_VectorCarryAndPersistentCacheEndToEnd below
// for those. (An earlier version of this test was misleadingly named
// "PersistentCacheSurvivesAcrossCycles" while testing neither the cache nor
// replay — caught by review — teardown1 here calls Save+Prune, so doc1
// comes back from the saved segment, not from journal replay.)
func TestBuildEngine_IndexSurvivesSaveLoadCycle(t *testing.T) {
	withTestCLIFlags(t)

	eng1, _, _, teardown1, err := buildEngine(false, true)
	if err != nil {
		t.Fatalf("buildEngine (first): %v", err)
	}
	if err := eng1.Add(context.Background(), "doc1", "the quick brown fox"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	teardown1()

	eng2, _, _, teardown2, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine (second, load): %v", err)
	}
	defer teardown2()
	if _, ok := eng2.GetText("doc1"); !ok {
		t.Error("expected doc1 to survive the reload")
	}
}

// fakeOllamaServer returns an httptest.Server mimicking Ollama's
// /api/embeddings endpoint closely enough for internal/embedding.OllamaEmbedder:
// it decodes {"model","prompt"} and replies {"embedding":[...]} with a
// deterministic, text-dependent 384-float vector (384 to match the
// "all-minilm" case OllamaEmbedder.Dimensions() recognizes) and always
// succeeds — the fakeVectorEmbedder pattern already used by
// internal/embedding/cache_test.go, reached over real HTTP so this test can
// drive the real --embedder ollama path through buildEngine and get a real
// (non-nil, deterministic) vector without any network dependency.
func fakeOllamaServer(calls *atomic.Int64) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model  string `json:"model"`
			Prompt string `json:"prompt"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		calls.Add(1)
		var seed uint32 = 2166136261
		for _, b := range []byte(req.Prompt) {
			seed ^= uint32(b)
			seed *= 16777619
		}
		vec := make([]float32, 384)
		for i := range vec {
			seed ^= seed << 13
			seed ^= seed >> 7
			seed ^= seed << 17
			vec[i] = float32(seed) / float32(1<<31)
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"embedding": vec})
	}))
}

// TestBuildEngine_VectorCarryAndPersistentCacheEndToEnd drives the real
// --embedder ollama path (pointed at a local fake server, so it produces
// real non-nil vectors instead of the deterministic embedder's permanent
// failure) through buildEngine twice, simulating an unclean exit between
// the two cycles, and proves both of this branch's core claims for real:
//  1. replaying a document whose vector was already journalled (format v2)
//     does not call the embedder again;
//  2. embedding the same text again under a different document ID, in a
//     second process-equivalent cycle with an empty in-memory LRU, hits the
//     persistent cache instead of calling the embedder again.
//
// This is the test review flagged as missing: the CLI-level tests before
// this fix all used the deterministic embedder, which can never produce a
// vector under CGO_ENABLED=0, so none of them could have caught a
// regression in either of these two behaviors.
func TestBuildEngine_VectorCarryAndPersistentCacheEndToEnd(t *testing.T) {
	var embedCalls atomic.Int64
	srv := fakeOllamaServer(&embedCalls)
	defer srv.Close()

	withTestCLIFlags(t)
	cliFlags.embedder = "ollama"
	cliFlags.ollamaURL = srv.URL
	cliFlags.ollamaModel = "all-minilm" // OllamaEmbedder.Dimensions() -> 384

	ctx := context.Background()

	// Cycle 1: index a document, then simulate an unclean exit — close the
	// storage and index handles directly, skipping buildEngine's normal
	// teardown (which would Save+Prune), so the journalled entry (with its
	// vector) is still sitting unsaved in the journal for cycle 2 to replay.
	eng1, storageEng1, _, _, err := buildEngine(false, true)
	if err != nil {
		t.Fatalf("buildEngine (cycle 1): %v", err)
	}
	if err := eng1.Add(ctx, "doc1", "hello world"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	callsAfterAdd := embedCalls.Load()
	if callsAfterAdd == 0 {
		t.Fatal("expected Add to call the fake embedder at least once")
	}
	if err := storageEng1.Close(); err != nil {
		t.Fatalf("storageEng1.Close: %v", err)
	}
	if err := eng1.Close(); err != nil {
		t.Fatalf("eng1.Close: %v", err)
	}

	// Cycle 2: a fresh buildEngine (fresh in-memory LRU, fresh index.Engine)
	// replays the journal left behind by cycle 1's unclean exit.
	eng2, _, _, teardown2, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine (cycle 2): %v", err)
	}
	defer teardown2()

	if _, ok := eng2.GetText("doc1"); !ok {
		t.Fatal("expected doc1 to survive replay after the simulated unclean exit")
	}
	callsAfterReplay := embedCalls.Load()
	if callsAfterReplay != callsAfterAdd {
		t.Errorf("replay called the embedder %d more time(s); want 0 — a journalled vector must never be re-embedded",
			callsAfterReplay-callsAfterAdd)
	}

	// Same text, a different document ID, still within cycle 2: this must
	// hit the persistent cache (written by cycle 1's Add, read back from
	// the same on-disk --storage-dir), not the in-memory LRU (fresh in this
	// cycle) and not the embedder.
	if err := eng2.Add(ctx, "doc2", "hello world"); err != nil {
		t.Fatalf("Add (doc2): %v", err)
	}
	callsAfterSecondAdd := embedCalls.Load()
	if callsAfterSecondAdd != callsAfterReplay {
		t.Errorf("re-embedding repeated text called the embedder %d more time(s); want 0 — the persistent cache must be hit",
			callsAfterSecondAdd-callsAfterReplay)
	}
}

// TestBuildEngine_ReadOnlyCallerDoesNotContendForStorageLock is a regression
// test for a real review finding: before this fix, buildEngine always
// opened storage.Engine, which holds Pebble's exclusive directory lock for
// as long as it stays open. That made `zenith search` fail whenever a
// long-running `zenith watch`/`zenith serve` already held the lock on the
// same --storage-dir — a workflow that worked before storage was wired into
// cmd/zenith at all. withStorage=false (what search.go now passes) must let
// a caller proceed even while another buildEngine call's storage.Engine is
// still open on the same directory.
func TestBuildEngine_ReadOnlyCallerDoesNotContendForStorageLock(t *testing.T) {
	withTestCLIFlags(t)

	// Simulates a long-running `zenith watch`/`zenith serve` that has
	// opened storage and not yet exited.
	_, watcherStorageEng, _, watcherTeardown, err := buildEngine(false, true)
	if err != nil {
		t.Fatalf("buildEngine (simulated watcher): %v", err)
	}
	defer watcherTeardown()
	if watcherStorageEng == nil {
		t.Fatal("expected a non-nil storage engine when withStorage=true")
	}

	// A concurrent read-only caller (search) must succeed without ever
	// trying to open the same Pebble directory.
	_, searchStorageEng, _, searchTeardown, err := buildEngine(true, false)
	if err != nil {
		t.Fatalf("buildEngine (read-only, withStorage=false) failed while storage was held open: %v", err)
	}
	defer searchTeardown()
	if searchStorageEng != nil {
		t.Error("expected a nil storage engine when withStorage=false")
	}
}
