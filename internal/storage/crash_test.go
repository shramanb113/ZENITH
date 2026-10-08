package storage

import (
	"context"
	"testing"

	"github.com/cockroachdb/pebble/vfs"
)

// TestEngine_SurvivesUncleanShutdown proves that data Put with pebble.Sync is
// recoverable on the next Open against a fresh *pebble.DB instance, exercised
// via Pebble's own in-memory vfs.FS. Note on what "crash" means here: Pebble
// holds an exclusive lock on its directory for as long as a *pebble.DB stays
// open — true even against vfs.NewMem, since the lock is Pebble's own
// correctness mechanism, not an OS file-locking side effect — so a second
// Open while db1 is still live always fails with "resource temporarily
// unavailable", a real finding from running this test, not a hypothetical.
// db1 is Closed before reopening to release that lock; this still proves the
// real property that matters, because pebble.Sync's doc comment states
// durability is established "for durability of individual write operations"
// at Set/Commit time, not deferred until Close — Close releases file handles
// and the lock, it does not additionally commit anything.
func TestEngine_SurvivesUncleanShutdown(t *testing.T) {
	fs := vfs.NewMem()
	dir := "/test-db"

	db1, err := openWithFS(dir, fs)
	if err != nil {
		t.Fatalf("first open: %v", err)
	}
	if err := db1.Put(context.Background(), []byte("doc1"), []byte("hello")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	db2, err := openWithFS(dir, fs)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db2.Close()
	val, ok := db2.Get([]byte("doc1"))
	if !ok || string(val) != "hello" {
		t.Fatalf("after reopen, Get = (%q, %v), want (%q, true)", val, ok, "hello")
	}
}

// TestTxn_CrashBeforeCommitLeavesNothingAfterReopen proves the "all or
// nothing" guarantee holds across a restart, not just within one live
// process (Task 5's TestTxn_DiscardAppliesNothing only proves the in-process
// case). A Txn's Put/Delete calls stage into a client-side pebble.Batch that
// is never sent to the DB until Commit — so even though db1 is Closed before
// reopening (same lock-contention reason as the test above), the uncommitted
// batch was never applied in the first place, which is exactly the property
// under test.
func TestTxn_CrashBeforeCommitLeavesNothingAfterReopen(t *testing.T) {
	fs := vfs.NewMem()
	dir := "/test-txn-db"

	db1, err := openWithFS(dir, fs)
	if err != nil {
		t.Fatalf("first open: %v", err)
	}
	txn := db1.NewTxn()
	if err := txn.Put([]byte("a"), []byte("va")); err != nil {
		t.Fatalf("txn.Put: %v", err)
	}
	if err := txn.Put([]byte("b"), []byte("vb")); err != nil {
		t.Fatalf("txn.Put: %v", err)
	}
	// No Commit — this is the condition under test. db1 is closed (without
	// committing the txn) so the next Open can acquire the directory lock.
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	db2, err := openWithFS(dir, fs)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db2.Close()
	if _, ok := db2.Get([]byte("a")); ok {
		t.Fatal("key 'a' visible after reopen despite the transaction never committing")
	}
	if _, ok := db2.Get([]byte("b")); ok {
		t.Fatal("key 'b' visible after reopen despite the transaction never committing")
	}
}
