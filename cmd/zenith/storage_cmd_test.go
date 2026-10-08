package main

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage"
)

func TestStorageInspectCmd_FindsJournaledDoc(t *testing.T) {
	withTestCLIFlags(t)

	eng, storageEng, _, _, err := buildEngine(false, true)
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
	if err := eng.Close(); err != nil {
		t.Fatalf("eng.Close: %v", err)
	}

	found, err := runStorageInspect(nil, []string{"doc1"})
	if err != nil {
		t.Fatalf("runStorageInspect: %v", err)
	}
	if !found {
		t.Error("expected doc1 to be found in the journal")
	}
}

func TestStorageInspectCmd_MissingDocReturnsFalse(t *testing.T) {
	withTestCLIFlags(t)

	eng, storageEng, _, _, err := buildEngine(false, true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	if err := storageEng.Close(); err != nil {
		t.Fatalf("storageEng.Close: %v", err)
	}
	if err := eng.Close(); err != nil {
		t.Fatalf("eng.Close: %v", err)
	}

	found, err := runStorageInspect(nil, []string{"never-written"})
	if err != nil {
		t.Fatalf("runStorageInspect: %v", err)
	}
	if found {
		t.Error("expected a miss for an id that was never journaled")
	}
}

// TestStoragePruneCmd_ActuallyPrunesTheJournal does NOT call buildEngine's
// own teardown before invoking runStoragePrune — review correctly flagged
// that doing so (an earlier version of this test did) leaves nothing left
// to prune, so a completely no-op Prune would still pass. Instead it closes
// only the storage handle (leaving the journal entry un-pruned, same
// technique as the inspect tests) and then checks the journal directly
// before and after.
func TestStoragePruneCmd_ActuallyPrunesTheJournal(t *testing.T) {
	withTestCLIFlags(t)

	eng, storageEng, _, _, err := buildEngine(false, true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	if err := eng.Add(context.Background(), "doc1", "some text"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, ok := storageEng.Get([]byte("doc1")); !ok {
		t.Fatal("expected doc1 to be journalled before prune runs")
	}
	if err := storageEng.Close(); err != nil {
		t.Fatalf("storageEng.Close: %v", err)
	}
	if err := eng.Close(); err != nil {
		t.Fatalf("eng.Close: %v", err)
	}

	if err := runStoragePrune(nil, nil); err != nil {
		t.Fatalf("runStoragePrune: %v", err)
	}

	storageEng2, err := storage.Open(storage.EngineConfig{Dir: cliFlags.storageDir})
	if err != nil {
		t.Fatalf("reopen storage engine: %v", err)
	}
	_, stillJournalled := storageEng2.Get([]byte("doc1"))
	if err := storageEng2.Close(); err != nil {
		t.Fatalf("storageEng2.Close: %v", err)
	}
	if stillJournalled {
		t.Error("expected doc1's journal entry to be pruned after runStoragePrune")
	}

	eng2, _, _, teardown2, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine (verify saved): %v", err)
	}
	defer teardown2()
	if _, ok := eng2.GetText("doc1"); !ok {
		t.Error("expected doc1 to have been saved to the segment by prune's Save step")
	}
}
