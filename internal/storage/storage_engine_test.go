package storage

import (
	"context"
	"errors"
	"testing"
)

func openTestEngine(t *testing.T) *Engine {
	t.Helper()
	cfg := EngineConfig{Dir: t.TempDir()}
	e, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = e.Close() })
	return e
}

func TestEngine_PutThenGet(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	val, ok := e.Get([]byte("doc1"))
	if !ok {
		t.Fatal("Get: want found, got not found")
	}
	if string(val) != "hello" {
		t.Fatalf("Get = %q, want %q", val, "hello")
	}
}

func TestEngine_DeleteThenGetNotFound(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := e.Delete(ctx, []byte("doc1")); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, ok := e.Get([]byte("doc1")); ok {
		t.Fatal("Get after Delete: want not found")
	}
}

func TestEngine_GetMissingKeyNotFound(t *testing.T) {
	e := openTestEngine(t)
	if _, ok := e.Get([]byte("nope")); ok {
		t.Fatal("Get on missing key: want not found")
	}
}

func TestEngine_PutEmptyKeyRejected(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte{}, []byte("v")); err == nil {
		t.Fatal("Put with empty key: want error, got nil")
	}
}

func TestEngine_DeleteEmptyKeyRejected(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Delete(ctx, []byte{}); err == nil {
		t.Fatal("Delete with empty key: want error, got nil")
	}
}

func TestEngine_PutAfterCloseRejected(t *testing.T) {
	e := openTestEngine(t)
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := e.Put(context.Background(), []byte("k"), []byte("v")); err == nil {
		t.Fatal("Put after Close: want error, got nil")
	}
}

func TestEngine_DeleteAfterCloseRejected(t *testing.T) {
	e := openTestEngine(t)
	if err := e.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := e.Delete(context.Background(), []byte("k")); err == nil {
		t.Fatal("Delete after Close: want error, got nil")
	}
}

func TestEngine_ReplaySeesLivePutsInKeyOrder(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("b"), []byte("vb")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := e.Put(ctx, []byte("a"), []byte("va")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	var gotKeys []string
	var gotVals []string
	var gotDeletes []bool
	err := e.Replay(func(key, value []byte, isDelete bool) error {
		gotKeys = append(gotKeys, string(key))
		gotVals = append(gotVals, string(value))
		gotDeletes = append(gotDeletes, isDelete)
		return nil
	})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	wantKeys := []string{"a", "b"}
	wantVals := []string{"va", "vb"}
	if len(gotKeys) != 2 || gotKeys[0] != wantKeys[0] || gotKeys[1] != wantKeys[1] {
		t.Fatalf("Replay keys = %v, want %v (Pebble iteration order)", gotKeys, wantKeys)
	}
	if gotVals[0] != wantVals[0] || gotVals[1] != wantVals[1] {
		t.Fatalf("Replay values = %v, want %v", gotVals, wantVals)
	}
	if gotDeletes[0] || gotDeletes[1] {
		t.Fatal("Replay reported a live Put as a delete")
	}
}

// This is the Review Focus case: a document deleted after the last segment
// save, with a crash before the next save, must still be reported as deleted
// by replay — not silently absent, which a caller would misread as "never
// existed, nothing to do" instead of "existed, now gone, tell the index."
func TestEngine_ReplaySeesDeletesNotJustAbsence(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := e.Delete(ctx, []byte("doc1")); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	var sawDelete bool
	err := e.Replay(func(key, value []byte, isDelete bool) error {
		if string(key) == "doc1" {
			sawDelete = isDelete
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if !sawDelete {
		t.Fatal("Replay did not report doc1 as deleted — a plain iteration-skips-tombstones implementation would fail this")
	}
}

func TestEngine_ReplayPropagatesCallbackError(t *testing.T) {
	e := openTestEngine(t)
	if err := e.Put(context.Background(), []byte("a"), []byte("va")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	wantErr := errors.New("callback failed")
	err := e.Replay(func(key, value []byte, isDelete bool) error {
		return wantErr
	})
	if !errors.Is(err, wantErr) {
		t.Fatalf("Replay error = %v, want %v", err, wantErr)
	}
}

func TestEngine_PruneRemovesOnlySnapshottedKeys(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("a"), []byte("va")); err != nil {
		t.Fatalf("Put a: %v", err)
	}
	if err := e.Put(ctx, []byte("b"), []byte("vb")); err != nil {
		t.Fatalf("Put b: %v", err)
	}

	snap := e.Snapshot()

	// A write arriving after the snapshot was taken must survive Prune —
	// this is the race Prune exists to avoid (pruning "everything now"
	// would wrongly delete this too).
	if err := e.Put(ctx, []byte("c"), []byte("vc")); err != nil {
		t.Fatalf("Put c: %v", err)
	}

	if err := e.Prune(snap); err != nil {
		t.Fatalf("Prune: %v", err)
	}
	if err := snap.Close(); err != nil {
		t.Fatalf("Snapshot.Close: %v", err)
	}

	if _, ok := e.Get([]byte("a")); ok {
		t.Fatal("Prune left snapshotted key 'a' behind")
	}
	if _, ok := e.Get([]byte("b")); ok {
		t.Fatal("Prune left snapshotted key 'b' behind")
	}
	val, ok := e.Get([]byte("c"))
	if !ok || string(val) != "vc" {
		t.Fatalf("Prune incorrectly removed key 'c' written after the snapshot: Get = (%q, %v)", val, ok)
	}
}

func TestEngine_PruneOfEmptySnapshotIsNoop(t *testing.T) {
	e := openTestEngine(t)
	snap := e.Snapshot()
	defer snap.Close()
	if err := e.Prune(snap); err != nil {
		t.Fatalf("Prune of empty snapshot: %v", err)
	}
}

func TestEngine_DurableAcrossReopen(t *testing.T) {
	dir := t.TempDir()
	e1, err := Open(EngineConfig{Dir: dir})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := e1.Put(context.Background(), []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := e1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	e2, err := Open(EngineConfig{Dir: dir})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer e2.Close()
	val, ok := e2.Get([]byte("doc1"))
	if !ok || string(val) != "hello" {
		t.Fatalf("after reopen, Get = (%q, %v), want (%q, true)", val, ok, "hello")
	}
}

func TestEmbeddingCache_PutThenGet(t *testing.T) {
	e := openTestEngine(t)

	key := []byte("model-a\x00Dabc123")
	vec := []float32{0.5, -0.25, 1.0}
	if err := e.PutEmbedding(key, vec); err != nil {
		t.Fatalf("PutEmbedding: %v", err)
	}
	got, ok := e.GetEmbedding(key)
	if !ok {
		t.Fatal("expected a hit after Put")
	}
	if len(got) != len(vec) {
		t.Fatalf("got len %d, want %d", len(got), len(vec))
	}
	for i := range vec {
		if got[i] != vec[i] {
			t.Errorf("got[%d] = %v, want %v", i, got[i], vec[i])
		}
	}
}

func TestEmbeddingCache_MissForUnknownKey(t *testing.T) {
	e := openTestEngine(t)

	if _, ok := e.GetEmbedding([]byte("never-written")); ok {
		t.Error("expected a miss for a key never written")
	}
}

func TestEmbeddingCache_Overwrite(t *testing.T) {
	e := openTestEngine(t)

	key := []byte("k")
	_ = e.PutEmbedding(key, []float32{1, 2, 3})
	_ = e.PutEmbedding(key, []float32{9, 9})
	got, ok := e.GetEmbedding(key)
	if !ok || len(got) != 2 || got[0] != 9 {
		t.Errorf("got %v, ok=%v, want [9 9], true", got, ok)
	}
}

func TestEmbeddingCache_SeparateFromDocumentJournal(t *testing.T) {
	// The embedding cache and the document journal must not share a
	// keyspace: writing a cache entry must never make Replay see it as a
	// journalled document.
	e := openTestEngine(t)

	if err := e.PutEmbedding([]byte("same-bytes-as-a-docid"), []float32{1, 2}); err != nil {
		t.Fatalf("PutEmbedding: %v", err)
	}
	seen := false
	if err := e.Replay(func(key, value []byte, isDelete bool) error {
		seen = true
		return nil
	}); err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if seen {
		t.Error("Replay must never see embedding-cache entries")
	}
}

func TestEmbeddingCache_SurvivesReopen(t *testing.T) {
	dir := t.TempDir()
	e1, err := Open(EngineConfig{Dir: dir})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	key := []byte("model-a\x00Dabc123")
	if err := e1.PutEmbedding(key, []float32{1, 2, 3}); err != nil {
		t.Fatalf("PutEmbedding: %v", err)
	}
	if err := e1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	e2, err := Open(EngineConfig{Dir: dir})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer e2.Close()
	got, ok := e2.GetEmbedding(key)
	if !ok || len(got) != 3 || got[0] != 1 {
		t.Fatalf("after reopen, GetEmbedding = (%v, %v), want ([1 2 3], true)", got, ok)
	}
}
