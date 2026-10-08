package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTxnAddCmd_AllDocsAppliedAtomically(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	jsonlPath := filepath.Join(dir, "docs.jsonl")
	content := `{"id":"a","text":"first document"}
{"id":"b","text":"second document","attrs":{"lang":"en"}}
`
	if err := os.WriteFile(jsonlPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	txnAddFlags.file = jsonlPath

	if err := runTxnAdd(nil, nil); err != nil {
		t.Fatalf("runTxnAdd: %v", err)
	}

	eng, _, _, teardown, err := buildEngine(true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	defer teardown()
	if _, ok := eng.GetText("a"); !ok {
		t.Error("expected doc a to be indexed")
	}
	if _, ok := eng.GetText("b"); !ok {
		t.Error("expected doc b to be indexed")
	}
}

func TestTxnAddCmd_MalformedLineFailsWithLineNumber(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	jsonlPath := filepath.Join(dir, "bad.jsonl")
	content := `{"id":"a","text":"ok"}
not json at all
`
	if err := os.WriteFile(jsonlPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	txnAddFlags.file = jsonlPath

	err := runTxnAdd(nil, nil)
	if err == nil {
		t.Fatal("expected an error for a malformed JSONL line")
	}
	if !strings.Contains(err.Error(), "line 2") {
		t.Errorf("error = %q, want it to mention line 2", err.Error())
	}
}

func TestTxnRemoveCmd_AllRemovedAtomically(t *testing.T) {
	dir := t.TempDir()
	cliFlags.dbPath = filepath.Join(dir, "zenith.db")
	cliFlags.fstPath = filepath.Join(dir, "index.fst")
	cliFlags.storageDir = filepath.Join(dir, "pebble")
	cliFlags.embedder = "deterministic"
	cliFlags.model = ""
	cliFlags.queryCacheSize = -1

	jsonlPath := filepath.Join(dir, "docs.jsonl")
	content := `{"id":"a","text":"first document"}
{"id":"b","text":"second document"}
`
	if err := os.WriteFile(jsonlPath, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	txnAddFlags.file = jsonlPath
	if err := runTxnAdd(nil, nil); err != nil {
		t.Fatalf("runTxnAdd: %v", err)
	}

	txnRemoveFlags.ids = "a, b"
	if err := runTxnRemove(nil, nil); err != nil {
		t.Fatalf("runTxnRemove: %v", err)
	}

	eng, _, _, teardown, err := buildEngine(true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	defer teardown()
	if _, ok := eng.GetText("a"); ok {
		t.Error("expected doc a to be removed")
	}
	if _, ok := eng.GetText("b"); ok {
		t.Error("expected doc b to be removed")
	}
}
