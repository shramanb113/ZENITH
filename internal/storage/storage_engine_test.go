package storage

import (
	"context"
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
