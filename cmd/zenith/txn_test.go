package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The tests in this file check that `zenith txn add/remove` correctly wire
// the CLI's JSONL/--ids input to index.Engine.AddTransaction/RemoveBatch —
// not the all-or-nothing atomicity guarantee itself, which is already
// proven directly against AddTransaction/RemoveBatch at the index.Engine
// unit level (internal/index/transaction_test.go:
// TestAddTransaction_CommitFailureAppliesNothing,
// TestRemoveBatch_AllRemovedAfterCommit) and isn't realistically
// re-triggerable from the CLI surface (it requires a Pebble commit
// failure, not a CLI input error — a malformed JSONL line, covered below,
// fails before AddTransaction is ever called at all, so it doesn't
// exercise atomicity either; it just proves nothing partial gets applied
// on a parse error).
func TestTxnAddCmd_AllDocsAppliedAtomically(t *testing.T) {
	dir := withTestCLIFlags(t)

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

	eng, _, _, teardown, err := buildEngine(true, true)
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
	dir := withTestCLIFlags(t)

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

	// Doc "a" on line 1 parsed fine, but the whole file is scanned before
	// AddTransaction is ever called — confirm the earlier, valid line was
	// never applied either.
	eng, _, _, teardown, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	defer teardown()
	if _, ok := eng.GetText("a"); ok {
		t.Error("expected doc a to NOT be indexed — a later parse error must apply nothing from the file")
	}
}

// TestTxnAddCmd_LineOver64KiBSucceeds is a regression test for a real
// review finding: bufio.Scanner's default 64KiB token limit used to abort
// the whole file with "bufio.Scanner: token too long" (and no line number)
// for any realistic document whose JSONL line exceeds that — a limit that
// has nothing to do with this format's own constraints.
func TestTxnAddCmd_LineOver64KiBSucceeds(t *testing.T) {
	dir := withTestCLIFlags(t)

	bigText := strings.Repeat("word ", 20_000) // ~100KB, over the 64KiB default
	jsonlPath := filepath.Join(dir, "big.jsonl")
	doc := jsonlDoc{ID: "big", Text: bigText}
	line, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(jsonlPath, append(line, '\n'), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	txnAddFlags.file = jsonlPath

	if err := runTxnAdd(nil, nil); err != nil {
		t.Fatalf("runTxnAdd: %v", err)
	}

	eng, _, _, teardown, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	defer teardown()
	if _, ok := eng.GetText("big"); !ok {
		t.Error("expected the over-64KiB document to be indexed")
	}
}

// TestTxnRemoveCmd_TrailingCommaIgnoresEmptyID is a regression test for a
// real review finding: "--ids a," used to split into ["a", ""] and pass the
// empty string to RemoveBatch as if it were a real document ID, inflating
// the printed "Removed N documents" count with an ID that never existed.
func TestTxnRemoveCmd_TrailingCommaIgnoresEmptyID(t *testing.T) {
	dir := withTestCLIFlags(t)

	jsonlPath := filepath.Join(dir, "docs.jsonl")
	if err := os.WriteFile(jsonlPath, []byte(`{"id":"a","text":"first document"}`+"\n"), 0644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	txnAddFlags.file = jsonlPath
	if err := runTxnAdd(nil, nil); err != nil {
		t.Fatalf("runTxnAdd: %v", err)
	}

	txnRemoveFlags.ids = "a,"
	if err := runTxnRemove(nil, nil); err != nil {
		t.Fatalf("runTxnRemove: %v", err)
	}

	eng, _, _, teardown, err := buildEngine(true, true)
	if err != nil {
		t.Fatalf("buildEngine: %v", err)
	}
	defer teardown()
	if _, ok := eng.GetText("a"); ok {
		t.Error("expected doc a to be removed")
	}
}

func TestTxnRemoveCmd_EmptyIDsOnlyIsAnError(t *testing.T) {
	withTestCLIFlags(t)

	txnRemoveFlags.ids = ", ,"
	if err := runTxnRemove(nil, nil); err == nil {
		t.Fatal("expected an error when --ids contains only empty entries")
	}
}

func TestTxnRemoveCmd_AllRemovedAtomically(t *testing.T) {
	dir := withTestCLIFlags(t)

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

	eng, _, _, teardown, err := buildEngine(true, true)
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

// TestToAttrs_ScalarArraySupported is a regression test for a real review
// finding: toAttrs used to silently drop array-valued attrs even though
// index.AttrValue's Kind already has AttrArray specifically to hold them.
// Pure unit test, no cliFlags/global state touched.
func TestToAttrs_ScalarArraySupported(t *testing.T) {
	attrs, err := toAttrs(map[string]any{"tags": []any{"a", "b", float64(3)}})
	if err != nil {
		t.Fatalf("toAttrs: %v", err)
	}
	v, ok := attrs["tags"]
	if !ok {
		t.Fatal("expected \"tags\" to be present")
	}
	if len(v.Arr) != 3 {
		t.Fatalf("Arr has %d elements, want 3", len(v.Arr))
	}
	if v.Arr[0].S != "a" || v.Arr[1].S != "b" || v.Arr[2].N != 3 {
		t.Errorf("Arr = %+v, want [a b 3]", v.Arr)
	}
}

// TestToAttrs_NullIsAnError and TestToAttrs_NestedObjectIsAnError are
// regression tests for the same finding: these were previously silently
// dropped rather than surfaced as an error, which can quietly lose part of
// a document's metadata with nothing telling the caller it happened.
func TestToAttrs_NullIsAnError(t *testing.T) {
	if _, err := toAttrs(map[string]any{"k": nil}); err == nil {
		t.Error("expected an error for a null attribute value")
	}
}

func TestToAttrs_NestedObjectIsAnError(t *testing.T) {
	if _, err := toAttrs(map[string]any{"k": map[string]any{"nested": "object"}}); err == nil {
		t.Error("expected an error for a nested-object attribute value")
	}
}

func TestToAttrs_NestedArrayIsAnError(t *testing.T) {
	if _, err := toAttrs(map[string]any{"k": []any{[]any{"nested"}}}); err == nil {
		t.Error("expected an error for a nested array (an element of Arr must never itself be AttrArray)")
	}
}
