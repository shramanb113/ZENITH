package storage

import (
	"context"
	"testing"
)

func TestTxn_CommitAppliesAllKeysAtomically(t *testing.T) {
	e := openTestEngine(t)
	txn := e.NewTxn()
	if err := txn.Put([]byte("a"), []byte("va")); err != nil {
		t.Fatalf("txn.Put a: %v", err)
	}
	if err := txn.Put([]byte("b"), []byte("vb")); err != nil {
		t.Fatalf("txn.Put b: %v", err)
	}
	if err := txn.Commit(context.Background()); err != nil {
		t.Fatalf("Commit: %v", err)
	}

	va, ok := e.Get([]byte("a"))
	if !ok || string(va) != "va" {
		t.Fatalf("Get a after commit = (%q, %v), want (va, true)", va, ok)
	}
	vb, ok := e.Get([]byte("b"))
	if !ok || string(vb) != "vb" {
		t.Fatalf("Get b after commit = (%q, %v), want (vb, true)", vb, ok)
	}
}

func TestTxn_DiscardAppliesNothing(t *testing.T) {
	e := openTestEngine(t)
	txn := e.NewTxn()
	if err := txn.Put([]byte("a"), []byte("va")); err != nil {
		t.Fatalf("txn.Put: %v", err)
	}
	if err := txn.Discard(); err != nil {
		t.Fatalf("Discard: %v", err)
	}
	if _, ok := e.Get([]byte("a")); ok {
		t.Fatal("key visible after Discard — a discarded txn must apply nothing")
	}
}

func TestTxn_CommitWithCancelledContextAppliesNothing(t *testing.T) {
	e := openTestEngine(t)
	txn := e.NewTxn()
	if err := txn.Put([]byte("a"), []byte("va")); err != nil {
		t.Fatalf("txn.Put: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := txn.Commit(ctx); err == nil {
		t.Fatal("Commit with a cancelled context: want error, got nil")
	}
	if _, ok := e.Get([]byte("a")); ok {
		t.Fatal("key visible after a cancelled Commit — must apply nothing")
	}
}

func TestTxn_DeleteWithinTxnIsVisibleToReplayAsDelete(t *testing.T) {
	e := openTestEngine(t)
	ctx := context.Background()
	if err := e.Put(ctx, []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}

	txn := e.NewTxn()
	if err := txn.Delete([]byte("doc1")); err != nil {
		t.Fatalf("txn.Delete: %v", err)
	}
	if err := txn.Commit(ctx); err != nil {
		t.Fatalf("Commit: %v", err)
	}

	var sawDelete bool
	if err := e.Replay(func(key, value []byte, isDelete bool) error {
		if string(key) == "doc1" {
			sawDelete = isDelete
		}
		return nil
	}); err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if !sawDelete {
		t.Fatal("a delete committed via Txn must be visible to Replay as a delete")
	}
}
