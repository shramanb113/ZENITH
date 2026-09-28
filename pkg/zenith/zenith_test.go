package zenith_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage/wal"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// bgCtx returns a plain background context — used by all tests that don't
// need a specific deadline or cancellation.
func bgCtx() context.Context { return context.Background() }

// openMem opens an in-memory DB with BM25Only so tests run without needing
// the ONNX model downloaded. Registers cleanup automatically.
func openMem(t *testing.T, opts ...zenith.Option) *zenith.DB {
	t.Helper()
	all := append([]zenith.Option{zenith.WithBM25Only()}, opts...)
	db, err := zenith.Open(":memory:", all...)
	if err != nil {
		t.Fatalf("Open(:memory:): %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

// mustAdd calls db.Add and fails the test on error.
func mustAdd(t *testing.T, db *zenith.DB, id, text string) {
	t.Helper()
	if err := db.Add(bgCtx(), id, text); err != nil {
		t.Fatalf("Add(%q): %v", id, err)
	}
}

// ─── Open ────────────────────────────────────────────────────────────────────

func TestOpen_Memory(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open(:memory:): %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestOpen_Persistent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.db")
	db, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open(path): %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestOpen_EmptyPath(t *testing.T) {
	_, err := zenith.Open("")
	if err == nil {
		t.Fatal("want error for empty path")
	}
}

func TestOpen_MemoryDatabasesAreIndependent(t *testing.T) {
	db1, _ := zenith.Open(":memory:", zenith.WithBM25Only())
	db2, _ := zenith.Open(":memory:", zenith.WithBM25Only())
	defer db1.Close()
	defer db2.Close()

	mustAdd(t, db1, "doc1", "unique document in db1")

	results, err := db2.Search(bgCtx(), "unique document")
	if err != nil {
		t.Fatalf("Search db2: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf(":memory: databases share state — they must be independent")
	}
}

func TestOpen_SameFilePath_ErrLocked(t *testing.T) {
	path := filepath.Join(t.TempDir(), "locked.db")

	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("first Open: %v", err)
	}
	defer db1.Close()

	_, err = zenith.Open(path, zenith.WithBM25Only())
	if err == nil {
		t.Fatal("second Open on same path should return ErrLocked")
	}
}

func TestOpen_AfterClose_FileLockReleased(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reopen.db")

	db1, _ := zenith.Open(path, zenith.WithBM25Only())
	db1.Close()

	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open after close should succeed: %v", err)
	}
	db2.Close()
}

func TestOpen_CorruptFile_ReturnsErrorNotPanic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.db")
	// Write garbage that looks like a valid file but isn't a ZENITH index
	os.WriteFile(path, []byte("not a zenith index file"), 0600)

	_, err := zenith.Open(path, zenith.WithBM25Only())
	if err == nil {
		t.Fatal("opening a corrupt file should return an error")
	}
}

// ─── Add ─────────────────────────────────────────────────────────────────────

func TestAdd_Basic(t *testing.T) {
	db := openMem(t)
	if err := db.Add(bgCtx(), "doc1", "hello world"); err != nil {
		t.Fatalf("Add: %v", err)
	}
}

func TestAdd_Idempotent(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "original content")
	// Re-index with different content — should replace, not duplicate.
	mustAdd(t, db, "doc1", "updated content")

	results, _ := db.Search(bgCtx(), "original")
	for _, r := range results {
		if r.ID == "doc1" {
			t.Fatal("old content should be gone after re-index")
		}
	}
}

func TestAdd_EmptyID(t *testing.T) {
	db := openMem(t)
	err := db.Add(bgCtx(), "", "some text")
	if err == nil {
		t.Fatal("want error for empty ID")
	}
}

func TestAdd_IDTooLong(t *testing.T) {
	db := openMem(t)
	longID := make([]byte, 513)
	for i := range longID {
		longID[i] = 'a'
	}
	err := db.Add(bgCtx(), string(longID), "text")
	if err == nil {
		t.Fatal("want error for ID > 512 bytes")
	}
}

func TestAdd_ControlCharInID(t *testing.T) {
	db := openMem(t)
	cases := []string{"\x00id", "id\x01", "id\x1f"}
	for _, id := range cases {
		if err := db.Add(bgCtx(), id, "text"); err == nil {
			t.Errorf("Add(%q): want error for control char", id)
		}
	}
}

func TestAdd_PipeSeparatorInID(t *testing.T) {
	db := openMem(t)
	err := db.Add(bgCtx(), "my||doc", "text")
	if err == nil {
		t.Fatal("want error for || in ID")
	}
}

func TestAdd_EmptyText(t *testing.T) {
	db := openMem(t)
	err := db.Add(bgCtx(), "id", "")
	if err == nil {
		t.Fatal("want error for empty text")
	}
}

func TestAdd_WhitespaceText(t *testing.T) {
	db := openMem(t)
	for _, text := range []string{" ", "\t", "\n", "   \t\n  "} {
		if err := db.Add(bgCtx(), "id", text); err == nil {
			t.Errorf("Add with whitespace-only text %q should error", text)
		}
	}
}

func TestAdd_InvalidUTF8_Sanitised(t *testing.T) {
	db := openMem(t)
	// Invalid UTF-8 in text should be sanitised silently, not rejected.
	err := db.Add(bgCtx(), "doc1", "hello\xff\xfeworld")
	if err != nil {
		t.Fatalf("invalid UTF-8 in text should be sanitised, not rejected: %v", err)
	}
}

func TestAdd_AfterClose(t *testing.T) {
	db := openMem(t)
	db.Close()
	err := db.Add(bgCtx(), "id", "text")
	if err == nil {
		t.Fatal("Add after Close should return an error")
	}
}

func TestAdd_NilDB(t *testing.T) {
	var db *zenith.DB
	err := db.Add(bgCtx(), "id", "text")
	if err == nil {
		t.Fatal("Add on nil DB should return an error, not panic")
	}
}

// ─── AddBatch ────────────────────────────────────────────────────────────────

func TestAddBatch_Nil(t *testing.T) {
	db := openMem(t)
	if err := db.AddBatch(bgCtx(), nil); err != nil {
		t.Fatalf("AddBatch(nil) should be a no-op, got %v", err)
	}
}

func TestAddBatch_Empty(t *testing.T) {
	db := openMem(t)
	if err := db.AddBatch(bgCtx(), map[string]string{}); err != nil {
		t.Fatalf("AddBatch(empty) should be a no-op, got %v", err)
	}
}

func TestAddBatch_Basic(t *testing.T) {
	db := openMem(t)
	err := db.AddBatch(bgCtx(), map[string]string{
		"doc1": "hello world",
		"doc2": "foo bar baz",
	})
	if err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	results, _ := db.Search(bgCtx(), "hello")
	if len(results) == 0 {
		t.Fatal("expected results after AddBatch, got none")
	}
}

func TestAddBatch_Deterministic(t *testing.T) {
	docs := map[string]string{
		"a": "alpha content",
		"b": "beta content",
		"c": "gamma content",
	}

	db1 := openMem(t)
	db2 := openMem(t)
	db1.AddBatch(bgCtx(), docs)
	db2.AddBatch(bgCtx(), docs)

	r1, _ := db1.Search(bgCtx(), "alpha")
	r2, _ := db2.Search(bgCtx(), "alpha")

	if len(r1) != len(r2) {
		t.Fatalf("AddBatch determinism: result counts differ (%d vs %d)", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d] ID mismatch: %q vs %q", i, r1[i].ID, r2[i].ID)
		}
	}
}

func TestAddBatch_InvalidID_Rejected(t *testing.T) {
	db := openMem(t)
	err := db.AddBatch(bgCtx(), map[string]string{
		"valid":   "content",
		"bad||id": "content",
	})
	if err == nil {
		t.Fatal("AddBatch with invalid ID should return an error")
	}
}

func TestAddBatch_AfterClose(t *testing.T) {
	db := openMem(t)
	db.Close()
	err := db.AddBatch(bgCtx(), map[string]string{"id": "text"})
	if err == nil {
		t.Fatal("AddBatch after Close should return an error")
	}
}

func TestAddBatch_NilDB(t *testing.T) {
	var db *zenith.DB
	if err := db.AddBatch(bgCtx(), map[string]string{"id": "text"}); err == nil {
		t.Fatal("AddBatch on nil DB should return an error, not panic")
	}
}

// ─── Search ──────────────────────────────────────────────────────────────────

func TestSearch_EmptyIndex_ReturnsEmptySlice(t *testing.T) {
	db := openMem(t)
	results, err := db.Search(bgCtx(), "anything")
	if err != nil {
		t.Fatalf("Search on empty index: %v", err)
	}
	if results == nil {
		t.Fatal("Search must return []Result{}, not nil")
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestSearch_NoMatch_ReturnsEmptySlice(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "hello world")

	results, err := db.Search(bgCtx(), "xyzzy")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if results == nil {
		t.Fatal("Search must return []Result{}, not nil")
	}
}

func TestSearch_Basic(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "the quick brown fox")

	results, err := db.Search(bgCtx(), "quick brown")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for matching query")
	}

	found := false
	for _, r := range results {
		if r.ID == "doc1" {
			found = true
		}
	}
	if !found {
		t.Fatal("doc1 should be in results")
	}
}

func TestSearch_ScoresInRange(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "a", "machine learning algorithms")
	mustAdd(t, db, "b", "deep learning neural networks")
	mustAdd(t, db, "c", "natural language processing")

	results, err := db.Search(bgCtx(), "learning")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range results {
		if r.Score < 0 || r.Score > 1 {
			t.Errorf("score %f for %q is outside [0,1]", r.Score, r.ID)
		}
	}
}

func TestSearch_ScoresNeverNaN(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "some content here")

	results, _ := db.Search(bgCtx(), "content")
	for _, r := range results {
		if r.Score != r.Score { // NaN check
			t.Errorf("result %q has NaN score", r.ID)
		}
	}
}

func TestSearch_SortedByScoreDescending(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "exact", "golang search engine")
	mustAdd(t, db, "partial", "search engine")
	mustAdd(t, db, "none", "completely different topic")

	results, _ := db.Search(bgCtx(), "golang search engine")
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: results[%d].Score=%f > results[%d].Score=%f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

func TestSearch_DeterministicOrderForEqualScores(t *testing.T) {
	db := openMem(t)
	// Same text, different IDs — should produce equal scores.
	mustAdd(t, db, "bbb", "identical content")
	mustAdd(t, db, "aaa", "identical content")
	mustAdd(t, db, "ccc", "identical content")

	r1, _ := db.Search(bgCtx(), "identical content")
	r2, _ := db.Search(bgCtx(), "identical content")

	if len(r1) != len(r2) {
		t.Fatalf("result count differs: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ID != r2[i].ID {
			t.Errorf("result[%d]: got %q then %q — ordering must be deterministic", i, r1[i].ID, r2[i].ID)
		}
	}
}

func TestSearch_EqualScores_SortedByIDAlpha(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "zzz", "same words here")
	mustAdd(t, db, "aaa", "same words here")
	mustAdd(t, db, "mmm", "same words here")

	results, _ := db.Search(bgCtx(), "same words here")

	// Find the equal-score group and verify they're sorted by ID.
	ids := make([]string, len(results))
	for i, r := range results {
		ids[i] = r.ID
	}
	sorted := append([]string(nil), ids...)
	sort.Strings(sorted)
	for i := range ids {
		if ids[i] != sorted[i] {
			t.Errorf("equal-score results should be sorted by ID: got %v, want %v", ids, sorted)
			break
		}
	}
}

func TestSearch_NoDuplicateIDs(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "hello world search")
	mustAdd(t, db, "doc2", "hello search query")

	results, _ := db.Search(bgCtx(), "hello search")

	seen := map[string]bool{}
	for _, r := range results {
		if seen[r.ID] {
			t.Fatalf("duplicate ID %q in results", r.ID)
		}
		seen[r.ID] = true
	}
}

func TestSearch_WithLimitOption(t *testing.T) {
	db, _ := zenith.Open(":memory:", zenith.WithBM25Only(), zenith.WithLimit(2))
	defer db.Close()

	mustAdd(t, db, "a", "common words here")
	mustAdd(t, db, "b", "common words there")
	mustAdd(t, db, "c", "common words everywhere")

	results, _ := db.Search(bgCtx(), "common words")
	if len(results) > 2 {
		t.Fatalf("WithLimit(2) should cap at 2, got %d", len(results))
	}
}

func TestSearch_LimitSearchOption_OverridesDefault(t *testing.T) {
	db := openMem(t)
	for i := 0; i < 5; i++ {
		mustAdd(t, db, string(rune('a'+i)), "matching content here")
	}

	results, _ := db.Search(bgCtx(), "matching content", zenith.Limit(2))
	if len(results) > 2 {
		t.Fatalf("Limit(2) search option should cap at 2, got %d", len(results))
	}
}

// Regression test: the RRF ranker's internal candidate cap used to be
// hardcoded to 10 regardless of the caller's requested limit, so Limit(n)
// for n > 10 silently still returned at most 10 results. See
// internal/config.Config.MaxResults, which now sets a generous internal
// candidate cap decoupled from the user-facing limit.
func TestSearch_LimitAboveTen(t *testing.T) {
	db := openMem(t)
	for i := 0; i < 40; i++ {
		mustAdd(t, db, fmt.Sprintf("doc%d", i), "matching content about widgets")
	}

	results, err := db.Search(bgCtx(), "widgets", zenith.Limit(30))
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 30 {
		t.Fatalf("Limit(30) with 40 matching docs should return 30, got %d", len(results))
	}
}

func TestSearch_CancelledContext_NoPanic(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "hello world")

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // pre-cancel

	// Must not panic regardless of whether context cancellation is observed.
	_, _ = db.Search(ctx, "hello")
}

func TestSearch_AfterClose(t *testing.T) {
	db := openMem(t)
	db.Close()

	_, err := db.Search(bgCtx(), "query")
	if err == nil {
		t.Fatal("Search after Close should return an error")
	}
}

func TestSearch_NilDB(t *testing.T) {
	var db *zenith.DB
	_, err := db.Search(bgCtx(), "query")
	if err == nil {
		t.Fatal("Search on nil DB should return an error, not panic")
	}
}

// ─── Delete ──────────────────────────────────────────────────────────────────

func TestDelete_Existing(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "hello world")

	if err := db.Delete(bgCtx(), "doc1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	results, _ := db.Search(bgCtx(), "hello world")
	for _, r := range results {
		if r.ID == "doc1" {
			t.Fatal("deleted document should not appear in results")
		}
	}
}

func TestDelete_NonExistent_Idempotent(t *testing.T) {
	db := openMem(t)
	// Deleting a non-existent ID should return nil, not error.
	if err := db.Delete(bgCtx(), "doesnotexist"); err != nil {
		t.Fatalf("Delete(non-existent): want nil, got %v", err)
	}
}

func TestDelete_Twice_Idempotent(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "content")

	db.Delete(bgCtx(), "doc1")
	if err := db.Delete(bgCtx(), "doc1"); err != nil {
		t.Fatalf("second Delete should be idempotent, got %v", err)
	}
}

func TestDelete_EmptyID(t *testing.T) {
	db := openMem(t)
	if err := db.Delete(bgCtx(), ""); err == nil {
		t.Fatal("Delete with empty ID should return an error")
	}
}

func TestDelete_AfterClose(t *testing.T) {
	db := openMem(t)
	db.Close()
	if err := db.Delete(bgCtx(), "id"); err == nil {
		t.Fatal("Delete after Close should return an error")
	}
}

func TestDelete_NilDB(t *testing.T) {
	var db *zenith.DB
	if err := db.Delete(bgCtx(), "id"); err == nil {
		t.Fatal("Delete on nil DB should return an error, not panic")
	}
}

// ─── Close ───────────────────────────────────────────────────────────────────

func TestClose_Idempotent(t *testing.T) {
	db := openMem(t)
	if err := db.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := db.Close(); err != nil {
		t.Fatalf("second Close should return nil, got %v", err)
	}
}

func TestClose_NilDB(t *testing.T) {
	var db *zenith.DB
	if err := db.Close(); err != nil {
		t.Fatalf("Close on nil DB should return nil, got %v", err)
	}
}

func TestClose_DeferSafe(t *testing.T) {
	// Verifies that calling Close twice via defer is safe.
	func() {
		db := openMem(t)
		defer db.Close()
		defer db.Close()
	}()
}

// ─── Persistent mode ─────────────────────────────────────────────────────────

func TestPersistent_DataSurvivesClose(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "persist.db")

	// Phase 1: write
	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open (write): %v", err)
	}
	mustAdd(t, db1, "doc1", "persistent content survives")
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Phase 2: read
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open (read): %v", err)
	}
	defer db2.Close()

	results, err := db2.Search(bgCtx(), "persistent content")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc1" {
			found = true
		}
	}
	if !found {
		t.Fatal("doc1 should survive Close and reopen")
	}
}

func TestPersistent_IndexFileCreated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "created.db")

	db, _ := zenith.Open(path, zenith.WithBM25Only())
	mustAdd(t, db, "doc1", "content")
	db.Close()

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("index file should exist after Close")
	}
}

func TestPersistent_NoLockFileAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "unlock.db")

	db, _ := zenith.Open(path, zenith.WithBM25Only())
	db.Close()

	if _, err := os.Stat(path + ".lock"); !os.IsNotExist(err) {
		t.Fatal("lock file should be removed after Close")
	}
}

// ─── Concurrency ─────────────────────────────────────────────────────────────

func TestConcurrent_AddAndSearch(t *testing.T) {
	db := openMem(t)

	const writers = 5
	const readers = 10

	// Pre-seed a document so searches have something to find.
	mustAdd(t, db, "seed", "concurrent test document")

	var wg sync.WaitGroup

	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			id := string(rune('a' + n))
			db.Add(bgCtx(), id, "concurrent write document")
		}(i)
	}

	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			db.Search(bgCtx(), "concurrent")
		}()
	}

	wg.Wait()
}

func TestConcurrent_MultipleSearches(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "parallel search test content")
	mustAdd(t, db, "doc2", "parallel search test data")

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results, err := db.Search(bgCtx(), "parallel search")
			if err != nil {
				t.Errorf("concurrent Search: %v", err)
			}
			_ = results
		}()
	}
	wg.Wait()
}

func TestConcurrent_CloseWhileSearching(t *testing.T) {
	db, _ := zenith.Open(":memory:", zenith.WithBM25Only())
	mustAdd(t, db, "doc1", "content for concurrent close test")

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Either returns results or ErrClosed — both are valid.
			db.Search(bgCtx(), "content")
		}()
	}

	db.Close()
	wg.Wait()
}

// ─── Update pattern (re-index) ───────────────────────────────────────────────

func TestUpdate_ReplacesContent(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "original search terms here")
	mustAdd(t, db, "doc1", "completely different content now")

	// Old terms should no longer be searchable for doc1.
	results, _ := db.Search(bgCtx(), "original search terms")
	for _, r := range results {
		if r.ID == "doc1" {
			t.Error("doc1 should have replaced original content, not kept it")
		}
	}
}

// ─── Edge cases common in client packages ───────────────────────────────────

func TestSearch_LargeQuery(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "short content")

	// Queries from user input can be very long.
	longQuery := make([]byte, 10_000)
	for i := range longQuery {
		longQuery[i] = 'a'
	}
	// Must not panic or hang.
	_, err := db.Search(bgCtx(), string(longQuery))
	if err != nil {
		t.Fatalf("Search with very long query should not error: %v", err)
	}
}

func TestAdd_SpecialCharactersInID(t *testing.T) {
	db := openMem(t)
	// These IDs are common in real applications.
	validIDs := []string{
		"uuid-1234-5678",
		"/path/to/file.txt",
		"user@example.com",
		"product:sku:12345",
		"key=value",
	}
	for _, id := range validIDs {
		if err := db.Add(bgCtx(), id, "some content"); err != nil {
			t.Errorf("Add(%q) should succeed: %v", id, err)
		}
	}
}

func TestAdd_UnicodeContent(t *testing.T) {
	db := openMem(t)
	// Real applications index multilingual content.
	if err := db.Add(bgCtx(), "unicode", "日本語テスト محتوا فارسی Ελληνικά"); err != nil {
		t.Fatalf("Add with unicode content should not error: %v", err)
	}
}

func TestSearch_EmptyQuery(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "doc1", "some content")
	// Empty queries come from unvalidated user input.
	results, err := db.Search(bgCtx(), "")
	// Should not panic — error or empty results are both acceptable.
	_ = results
	_ = err
}

func TestAddBatch_LargeBatch(t *testing.T) {
	db := openMem(t)
	docs := make(map[string]string, 200)
	for i := 0; i < 200; i++ {
		id := string(rune('a'+i%26)) + string(rune('0'+i/26))
		docs[id] = "content for document " + id
	}
	if err := db.AddBatch(bgCtx(), docs); err != nil {
		t.Fatalf("AddBatch with 200 docs: %v", err)
	}
}

func TestSearch_AfterDelete_ResultsUpdated(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "keep", "important search content")
	mustAdd(t, db, "remove", "important search content")

	db.Delete(bgCtx(), "remove")

	results, _ := db.Search(bgCtx(), "important search content")
	for _, r := range results {
		if r.ID == "remove" {
			t.Fatal("deleted document still appearing in results")
		}
	}
	found := false
	for _, r := range results {
		if r.ID == "keep" {
			found = true
		}
	}
	if !found {
		t.Fatal("non-deleted document should still appear in results")
	}
}

// ─── AddBatch WAL single-fsync ───────────────────────────────────────────────

func TestAddBatch_WALRecoveredAfterCrash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "batch_crash.db")

	// Phase 1: clean baseline.
	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Phase 2: inject a batch of WAL records simulating a crash mid-AddBatch.
	injectWALRecords(t, path, map[string]string{
		"batch1": "first document in batch",
		"batch2": "second document in batch",
		"batch3": "third document in batch",
	})

	// Phase 3: reopen — all three must be recovered.
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open after crash: %v", err)
	}
	defer db2.Close()

	for _, id := range []string{"batch1", "batch2", "batch3"} {
		results, err := db2.Search(bgCtx(), id)
		if err != nil {
			t.Fatalf("Search %s: %v", id, err)
		}
		found := false
		for _, r := range results {
			if r.ID == id {
				found = true
			}
		}
		if !found {
			t.Errorf("%s not recovered from WAL after simulated batch crash", id)
		}
	}
}

// ─── WAL crash-recovery (persistent mode) ────────────────────────────────────

// injectWALRecords writes WAL records directly to path+".wal", simulating
// the state where documents were indexed (WAL written) but the process
// crashed before the gob checkpoint.
func injectWALRecords(t *testing.T, dbPath string, docs map[string]string) {
	t.Helper()
	walPath := dbPath + ".wal"
	w, _, err := wal.OpenWAL(walPath, wal.WALConfig{SyncMode: wal.SyncAlways})
	if err != nil {
		t.Fatalf("injectWALRecords OpenWAL: %v", err)
	}
	for id, text := range docs {
		if _, err := w.Append(context.Background(), &wal.Record{
			Op: wal.OpTypePut, Key: []byte(id), Value: []byte(text),
		}); err != nil {
			t.Fatalf("injectWALRecords Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("injectWALRecords Close: %v", err)
	}
}

func injectWALDelete(t *testing.T, dbPath, id string) {
	t.Helper()
	walPath := dbPath + ".wal"
	w, _, err := wal.OpenWAL(walPath, wal.WALConfig{SyncMode: wal.SyncAlways})
	if err != nil {
		t.Fatalf("injectWALDelete OpenWAL: %v", err)
	}
	if _, err := w.Append(context.Background(), &wal.Record{
		Op: wal.OpTypeDelete, Key: []byte(id),
	}); err != nil {
		t.Fatalf("injectWALDelete Append: %v", err)
	}
	_ = w.Close()
}

func TestPersistent_WALCrashRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "crash.db")

	// Phase 1: clean open+close — establishes an empty gob baseline.
	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open phase1: %v", err)
	}
	mustAdd(t, db1, "doc1", "the quick brown fox")
	if err := db1.Close(); err != nil {
		t.Fatalf("Close phase1: %v", err)
	}
	// After close: gob has doc1, WAL is empty.

	// Phase 2: simulate crash — inject WAL records that were never checkpointed.
	injectWALRecords(t, path, map[string]string{
		"doc2": "machine learning algorithms",
		"doc3": "database crash recovery test",
	})

	// Phase 3: reopen — WAL replay must recover doc2 and doc3 on top of gob's doc1.
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open after crash: %v", err)
	}
	defer db2.Close()

	results, err := db2.Search(bgCtx(), "machine learning")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc2" {
			found = true
		}
	}
	if !found {
		t.Fatal("doc2 must be recovered from WAL replay after simulated crash")
	}
}

func TestPersistent_WALEmptyAfterCleanClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "clean.db")

	db, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	mustAdd(t, db, "doc1", "some content here")
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// WAL file must exist but be empty (0 bytes) after a clean close.
	walPath := path + ".wal"
	info, err := os.Stat(walPath)
	if os.IsNotExist(err) {
		t.Fatal("WAL file must exist alongside the gob file")
	}
	if err != nil {
		t.Fatalf("stat WAL: %v", err)
	}
	if info.Size() != 0 {
		t.Errorf("WAL must be empty after clean close, got %d bytes", info.Size())
	}
}

func TestPersistent_DeleteSurvivesCrash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "delete_crash.db")

	// Phase 1: index a document and close cleanly.
	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	mustAdd(t, db1, "remove-me", "content to be removed later")
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	// gob has "remove-me", WAL is empty.

	// Phase 2: inject a Delete WAL record (simulating: delete was written, then crash).
	injectWALDelete(t, path, "remove-me")

	// Phase 3: reopen — WAL replay must remove the doc.
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open after delete crash: %v", err)
	}
	defer db2.Close()

	results, _ := db2.Search(bgCtx(), "content to be removed")
	for _, r := range results {
		if r.ID == "remove-me" {
			t.Fatal("deleted document must not appear after WAL replay")
		}
	}
}

func TestPersistent_CorruptGobFallsBackToWAL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "corrupt.db")

	// Phase 1: clean open+close — gob is saved, WAL is empty.
	db, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	mustAdd(t, db, "doc1", "base document in gob")
	if err := db.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Phase 2: inject a WAL record (crash before gob save).
	injectWALRecords(t, path, map[string]string{
		"doc2": "added before corruption, WAL only",
	})

	// Phase 3: corrupt the gob file.
	if err := os.WriteFile(path, []byte("NOT A ZENITH GOB"), 0600); err != nil {
		t.Fatalf("corrupt gob: %v", err)
	}

	// Phase 4: reopen — must fall back to WAL-only rebuild.
	// doc1 is lost (gob corrupted), doc2 must be found (WAL intact).
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open after corruption: %v", err)
	}
	defer db2.Close()

	results, err := db2.Search(bgCtx(), "added before corruption")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc2" {
			found = true
		}
	}
	if !found {
		t.Fatal("doc2 must be recovered from WAL even when gob is corrupt")
	}
}

// ─── Embedder identity header (P0-5) ──────────────────────────────────────────

// namedEmbedder implements the optional embedding.Named interface (Name()
// string) structurally, without importing any internal package — the same
// way customEmbedder proves zenith.Embedder is fully public.
type namedEmbedder struct{ name string }

func (e *namedEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0, 0, 0}, nil
}
func (e *namedEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{0, 0, 0}
	}
	return out, nil
}
func (e *namedEmbedder) Dimensions() int { return 3 }
func (e *namedEmbedder) Name() string    { return e.name }

// TestOpen_EmbedderMismatchRefusesToLoad verifies that reopening an index
// file with a differently-identified embedder is refused outright (P0-5)
// instead of silently loading and mixing two incompatible vector spaces.
func TestOpen_EmbedderMismatchRefusesToLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "mismatch.db")

	db1, err := zenith.Open(path, zenith.WithEmbedder(&namedEmbedder{name: "model-a"}))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	mustAdd(t, db1, "doc1", "hello world")
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	_, err = zenith.Open(path, zenith.WithEmbedder(&namedEmbedder{name: "model-b"}))
	if !errors.Is(err, zenith.ErrEmbedderMismatch) {
		t.Fatalf("Open with a different embedder identity = %v, want ErrEmbedderMismatch", err)
	}
}

// TestOpen_SameEmbedderIdentityReopensFine is the control: identical
// embedder identity across Open calls must not be treated as a mismatch.
func TestOpen_SameEmbedderIdentityReopensFine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "match.db")

	db1, err := zenith.Open(path, zenith.WithEmbedder(&namedEmbedder{name: "model-a"}))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	mustAdd(t, db1, "doc1", "hello world")
	if err := db1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	db2, err := zenith.Open(path, zenith.WithEmbedder(&namedEmbedder{name: "model-a"}))
	if err != nil {
		t.Fatalf("reopen with the same embedder identity should succeed: %v", err)
	}
	defer db2.Close()
}

// ─── Public Embedder interface ────────────────────────────────────────────────

// customEmbedder is implemented without importing any internal ZENITH package.
// It proves that zenith.Embedder is a fully public, self-contained interface.
type customEmbedder struct{}

func (c *customEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return []float32{0.1, 0.2, 0.3}, nil
}
func (c *customEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{0.1, 0.2, 0.3}
	}
	return out, nil
}
func (c *customEmbedder) Dimensions() int { return 3 }

// TestWithEmbedder_PublicInterface verifies that a caller can satisfy
// zenith.Embedder and pass it to WithEmbedder without importing any
// internal ZENITH packages.
func TestWithEmbedder_PublicInterface(t *testing.T) {
	// Compile-time assertion: *customEmbedder must satisfy the public interface.
	var _ zenith.Embedder = (*customEmbedder)(nil)

	db, err := zenith.Open(":memory:", zenith.WithEmbedder(&customEmbedder{}))
	if err != nil {
		t.Fatalf("Open with custom Embedder: %v", err)
	}
	defer db.Close()

	if err := db.Add(bgCtx(), "doc1", "hello world"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	results, err := db.Search(bgCtx(), "hello")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	_ = results
}
