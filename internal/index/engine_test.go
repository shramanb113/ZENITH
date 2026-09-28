package index

import (
	"context"
	"hash/fnv"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// newTestEngine returns a minimal Engine backed by a deterministic embedder.
// No FST path is set — keeps tests portable (no disk I/O for the FST).
func newTestEngine() *Engine {
	cfg := config.DefaultConfig()
	emb := embedding.NewDeterministicEmbedder(384)
	scorer := ranking.NewRRFRanker(0, 0)
	ana := analysis.NewStandardAnalyzer()
	return NewEngine(cfg, emb, scorer, ana)
}

// ─── generateEdgeNgrams ───────────────────────────────────────────────────────

func TestGenerateEdgeNgrams_ShortToken(t *testing.T) {
	// Token shorter than MinGram (3): should return just the token itself.
	got := generateEdgeNgrams("ab")
	if len(got) != 1 || got[0] != "ab" {
		t.Errorf("short token: got %v, want [ab]", got)
	}
}

func TestGenerateEdgeNgrams_ExactMinGram(t *testing.T) {
	// Exactly 3 chars: full token + no prefixes (MinGram..MinGram range is empty).
	got := generateEdgeNgrams("abc")
	if len(got) != 1 || got[0] != "abc" {
		t.Errorf("3-char token: got %v, want [abc]", got)
	}
}

func TestGenerateEdgeNgrams_LongToken(t *testing.T) {
	token := "kubernetes"
	got := generateEdgeNgrams(token)

	// Full token must always be present.
	if !slices.Contains(got, token) {
		t.Errorf("full token %q missing from ngrams %v", token, got)
	}

	// "kub" (len 3) must be present.
	if !slices.Contains(got, "kub") {
		t.Errorf("prefix 'kub' missing from ngrams %v", got)
	}

	// No duplicates.
	seen := make(map[string]bool)
	for _, ng := range got {
		if seen[ng] {
			t.Errorf("duplicate ngram %q in %v", ng, got)
		}
		seen[ng] = true
	}
}

func TestGenerateEdgeNgrams_NoDuplicateFullToken(t *testing.T) {
	// Original bug: the full token was appended twice when n > MaxGram.
	// "superlongword" has 13 chars > MaxGram(10).
	got := generateEdgeNgrams("superlongword")
	count := 0
	for _, ng := range got {
		if ng == "superlongword" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("full token appears %d times in ngrams (want exactly 1): %v", count, got)
	}
}

// ─── Add / Search ─────────────────────────────────────────────────────────────

func TestEngine_AddAndSearch_BasicMatch(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "kubernetes cluster deployment pod service"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := e.Add(ctx, "doc2", "docker container image registry"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	results, err := e.Search(ctx, "kubernetes deployment")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected non-empty search results")
	}
	if results[0].ID != "doc1" {
		t.Errorf("expected doc1 first, got %q (score=%.4f)", results[0].ID, results[0].Score)
	}
}

func TestEngine_AddAndSearch_FuzzyMatch(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "kubernetes cluster deployment"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// "kubrnetes" is 1 edit away from "kubernetes"
	results, err := e.Search(ctx, "kubrnetes")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc1" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("fuzzy search for 'kubrnetes' should match doc1; got %v", results)
	}
}

func TestEngine_AddAndSearch_PhoneticMatch(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "Robert Smith engineer"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// "Rupert" has the same Soundex code as "Robert"
	results, err := e.Search(ctx, "Rupert")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "doc1" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("phonetic search for 'Rupert' should match doc with 'Robert'; got %v", results)
	}
}

func TestEngine_AddIdempotent(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	// Index the same doc twice with different content — second should win.
	if err := e.Add(ctx, "doc1", "kubernetes cluster"); err != nil {
		t.Fatalf("first Add: %v", err)
	}
	if err := e.Add(ctx, "doc1", "docker container image"); err != nil {
		t.Fatalf("second Add: %v", err)
	}

	// Should match new content.
	results, err := e.Search(ctx, "docker container")
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
		t.Error("re-indexed doc1 should match new content 'docker container'")
	}
}

func TestEngine_Search_NoResults(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "kubernetes cluster"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Query for something completely unrelated and not fuzzy-close to anything indexed.
	results, err := e.Search(ctx, "zzzzzzzzz")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	// May return no results or very low scores — either is acceptable.
	for _, r := range results {
		if r.Score > 0.5 {
			t.Errorf("unexpectedly high score %.4f for unrelated query (doc=%q)", r.Score, r.ID)
		}
	}
}

func TestEngine_Search_MultipleDocumentRanking(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	// doc_relevant has 3 occurrences of search terms; doc_partial has 1.
	docs := map[string]string{
		"doc_relevant":  "search engine search ranking search algorithm",
		"doc_partial":   "database storage index",
		"doc_unrelated": "cooking recipe pasta tomato sauce",
	}
	for id, content := range docs {
		if err := e.Add(ctx, id, content); err != nil {
			t.Fatalf("Add(%q): %v", id, err)
		}
	}

	results, err := e.Search(ctx, "search ranking")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	if results[0].ID != "doc_relevant" {
		t.Errorf("expected doc_relevant first, got %q", results[0].ID)
	}
}

func TestEngine_FSTContains_AfterAdd(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "kubernetes deployment cluster"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	// Stemmed terms should be in the FST.
	// "kubernetes" is stemmed to "kubernet" by Porter2 (approximately).
	// We check that the FST was rebuilt and contains some indexed term.
	// Use FSTPrefixSearch for robustness.
	terms, err := e.FSTPrefixSearch("kub", 10)
	if err != nil {
		t.Fatalf("FSTPrefixSearch: %v", err)
	}
	if len(terms) == 0 {
		t.Error("expected FST to return terms with prefix 'kub' after indexing 'kubernetes'")
	}
}

// ─── Save / Load round-trip ───────────────────────────────────────────────────

func TestEngine_SaveLoad_RoundTrip(t *testing.T) {
	e := newTestEngine()
	defer e.Close()
	ctx := context.Background()

	docs := []struct{ id, text string }{
		{"alpha", "kubernetes cluster pod service"},
		{"beta", "docker container image registry"},
		{"gamma", "search index ranking score"},
	}
	for _, d := range docs {
		if err := e.Add(ctx, d.id, d.text); err != nil {
			t.Fatalf("Add(%q): %v", d.id, err)
		}
	}

	path := filepath.Join(t.TempDir(), "zenith.db")
	if err := e.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	// Load into fresh engine.
	e2 := newTestEngine()
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Search must work on the loaded engine.
	results, err := e2.Search(ctx, "kubernetes cluster")
	if err != nil {
		t.Fatalf("Search after Load: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results from loaded engine")
	}
	if results[0].ID != "alpha" {
		t.Errorf("expected 'alpha' first after load, got %q", results[0].ID)
	}
}

func TestEngine_Load_MissingFile(t *testing.T) {
	e := newTestEngine()
	err := e.Load(filepath.Join(t.TempDir(), "no_such_file.db"))
	if err == nil {
		t.Error("expected error when loading non-existent file")
	}
}

func TestEngine_Save_CreatesFile(t *testing.T) {
	e := newTestEngine()
	defer e.Close()
	ctx := context.Background()
	_ = e.Add(ctx, "doc1", "hello world")

	path := filepath.Join(t.TempDir(), "out.db")
	if err := e.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("Save did not create file: %v", err)
	}
}

// ─── AddWithVector ────────────────────────────────────────────────────────────

func TestEngine_AddWithVector_UsesProvidedVector(t *testing.T) {
	cfg := config.DefaultConfig()
	tkz := analysis.NewStandardAnalyzer()
	emb := embedding.NewDeterministicEmbedder(384)
	scorer := ranking.NewRRFRanker(0, 0)
	eng := NewEngine(cfg, emb, scorer, tkz)

	preVec := make([]float32, 384)
	for i := range preVec {
		preVec[i] = 0.42
	}

	err := eng.AddWithVector(context.Background(), "doc1", "hello world", preVec)
	if err != nil {
		t.Fatalf("AddWithVector returned error: %v", err)
	}

	results, err := eng.Search(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected at least one result after AddWithVector")
	}
	if results[0].ID != "doc1" {
		t.Errorf("expected result id=doc1, got %s", results[0].ID)
	}
}

// ─── Remove ───────────────────────────────────────────────────────────────────

func TestEngine_Remove_DocNotSearchable(t *testing.T) {
	cfg := config.DefaultConfig()
	eng := NewEngine(cfg, embedding.NewDeterministicEmbedder(384), ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())

	if err := eng.Add(context.Background(), "doc1", "hello world zenith"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	results, _ := eng.Search(context.Background(), "hello")
	if len(results) == 0 {
		t.Fatal("expected results before Remove")
	}

	if err := eng.Remove(context.Background(), "doc1"); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	results, _ = eng.Search(context.Background(), "hello")
	for _, r := range results {
		if r.ID == "doc1" {
			t.Error("doc1 still appears in results after Remove")
		}
	}
}

func TestEngine_Remove_Idempotent(t *testing.T) {
	cfg := config.DefaultConfig()
	eng := NewEngine(cfg, embedding.NewDeterministicEmbedder(384), ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())

	if err := eng.Remove(context.Background(), "never-existed"); err != nil {
		t.Errorf("Remove of non-existent doc returned error: %v", err)
	}
}

func TestEngine_Remove_ReAdd(t *testing.T) {
	cfg := config.DefaultConfig()
	eng := NewEngine(cfg, embedding.NewDeterministicEmbedder(384), ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())

	_ = eng.Add(context.Background(), "doc1", "hello world")
	_ = eng.Remove(context.Background(), "doc1")
	_ = eng.Add(context.Background(), "doc1", "hello world again")

	results, _ := eng.Search(context.Background(), "hello")
	found := false
	for _, r := range results {
		if r.ID == "doc1" {
			found = true
		}
	}
	if !found {
		t.Error("doc1 not found after Remove + re-Add")
	}
}

// ─── AddBatch ────────────────────────────────────────────────────────────────

func TestEngine_AddBatch_IndexesAllDocs(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	docs := []BatchDoc{
		{ID: "b1", Text: "distributed tracing observability", Vector: make([]float32, 384)},
		{ID: "b2", Text: "container orchestration kubernetes", Vector: make([]float32, 384)},
		{ID: "b3", Text: "machine learning inference pipeline", Vector: make([]float32, 384)},
	}
	if err := e.AddBatch(ctx, docs); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	for _, d := range docs {
		results, err := e.Search(ctx, d.Text[:10])
		if err != nil {
			t.Fatalf("Search(%q): %v", d.Text[:10], err)
		}
		found := false
		for _, r := range results {
			if r.ID == d.ID {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("doc %q not found after AddBatch", d.ID)
		}
	}
}

func TestEngine_AddBatch_FSTRebuildOnce(t *testing.T) {
	// Verify FST is usable after AddBatch (prefix search works → FST was rebuilt).
	e := newTestEngine()
	ctx := context.Background()

	docs := []BatchDoc{
		{ID: "d1", Text: "kubernetes cluster service", Vector: make([]float32, 384)},
		{ID: "d2", Text: "kubernetes deployment pod", Vector: make([]float32, 384)},
	}
	if err := e.AddBatch(ctx, docs); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	terms, err := e.FSTPrefixSearch("kub", 10)
	if err != nil {
		t.Fatalf("FSTPrefixSearch: %v", err)
	}
	if len(terms) == 0 {
		t.Error("expected FST to contain terms with prefix 'kub' after AddBatch")
	}
}

func TestEngine_AddBatch_Empty(t *testing.T) {
	e := newTestEngine()
	if err := e.AddBatch(context.Background(), nil); err != nil {
		t.Errorf("AddBatch(nil) should not error, got: %v", err)
	}
}

func TestEngine_AddBatch_SearchableAfterSave(t *testing.T) {
	e := newTestEngine()
	defer e.Close()
	ctx := context.Background()

	docs := []BatchDoc{
		{ID: "x1", Text: "golang concurrency channels goroutines", Vector: make([]float32, 384)},
	}
	if err := e.AddBatch(ctx, docs); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	path := filepath.Join(t.TempDir(), "batch.db")
	if err := e.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	e2 := newTestEngine()
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	results, err := e2.Search(ctx, "golang goroutines")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	found := false
	for _, r := range results {
		if r.ID == "x1" {
			found = true
		}
	}
	if !found {
		t.Error("AddBatch doc not found after Save/Load round-trip")
	}
}

// ─── removeID ─────────────────────────────────────────────────────────────────

func TestRemoveID(t *testing.T) {
	cases := []struct {
		ids    []uint64
		target uint64
		want   []uint64
	}{
		{[]uint64{1, 2, 3}, 2, []uint64{1, 3}},
		{[]uint64{1, 2, 3}, 1, []uint64{2, 3}},
		{[]uint64{1, 2, 3}, 3, []uint64{1, 2}},
		{[]uint64{1, 2, 3}, 9, []uint64{1, 2, 3}}, // not found
		{[]uint64{}, 1, []uint64{}},
		{nil, 1, nil},
	}
	for _, c := range cases {
		got := removeID(c.ids, c.target)
		if len(got) != len(c.want) {
			t.Errorf("removeID(%v, %d) = %v, want %v", c.ids, c.target, got, c.want)
			continue
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Errorf("removeID(%v, %d)[%d] = %d, want %d", c.ids, c.target, i, got[i], c.want[i])
			}
		}
	}
}

// ─── DocumentJournal ─────────────────────────────────────────────────────────

// mockJournal records Put/Delete calls for assertion in tests.
type mockJournal struct {
	puts    []string // doc IDs passed to Put
	deletes []string // doc IDs passed to Delete
}

func (m *mockJournal) Put(_ context.Context, key, _ []byte) error {
	m.puts = append(m.puts, string(key))
	return nil
}
func (m *mockJournal) Delete(_ context.Context, key []byte) error {
	m.deletes = append(m.deletes, string(key))
	return nil
}

func TestIndexEngine_JournalReceivesPut(t *testing.T) {
	e := newTestEngine()
	j := &mockJournal{}
	e.SetDocumentJournal(j)

	ctx := context.Background()
	if err := e.Add(ctx, "doc1", "hello world"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	if len(j.puts) != 1 || j.puts[0] != "doc1" {
		t.Errorf("journal puts: got %v, want [doc1]", j.puts)
	}
	if len(j.deletes) != 0 {
		t.Errorf("unexpected journal deletes: %v", j.deletes)
	}
}

func TestIndexEngine_JournalReceivesDelete(t *testing.T) {
	e := newTestEngine()
	j := &mockJournal{}
	e.SetDocumentJournal(j)

	ctx := context.Background()
	_ = e.Add(ctx, "doc1", "hello world")

	// Reset puts count, then delete.
	j.puts = nil
	if err := e.Remove(ctx, "doc1"); err != nil {
		t.Fatalf("Remove: %v", err)
	}

	if len(j.deletes) != 1 || j.deletes[0] != "doc1" {
		t.Errorf("journal deletes: got %v, want [doc1]", j.deletes)
	}
}

func TestIndexEngine_JournalNilSafe(t *testing.T) {
	e := newTestEngine() // no journal set
	ctx := context.Background()

	// Must not panic.
	if err := e.Add(ctx, "doc1", "text"); err != nil {
		t.Fatalf("Add without journal: %v", err)
	}
	if err := e.Remove(ctx, "doc1"); err != nil {
		t.Fatalf("Remove without journal: %v", err)
	}
}

// ─── Audit fix regressions ─────────────────────────────────────────────────────

// The BK-tree (fuzzy search) was never persisted or rebuilt on Load, so
// fuzzy search silently died after every restart.
func TestEngine_Load_RebuildsBKTree(t *testing.T) {
	e := newTestEngine()
	defer e.Close()
	ctx := context.Background()

	if err := e.Add(ctx, "doc1", "kubernetes cluster deployment"); err != nil {
		t.Fatalf("Add: %v", err)
	}

	path := filepath.Join(t.TempDir(), "bktree.db")
	if err := e.Save(path); err != nil {
		t.Fatalf("Save: %v", err)
	}

	e2 := newTestEngine()
	defer e2.Close()
	if err := e2.Load(path); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// "kuberntes" is a 1-edit typo of "kubernetes" — must be found via the
	// BK-tree's fuzzy pass after Load, not just via exact/n-gram matching.
	results, err := e2.Search(ctx, "kuberntes")
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
		t.Error("fuzzy match for 'kuberntes' failed after Load — BK-tree was not rebuilt")
	}
}

// A 64-bit doc-ID hash collision must not silently overwrite an unrelated
// document's postings, vector and BM25 state.
func TestEngine_Add_RejectsIDHashCollision(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	h := fnv.New64a()
	h.Write([]byte("victim"))
	internalID := h.Sum64()

	// Simulate a genuine 64-bit hash collision: some other original ID
	// already claims this internal ID.
	e.idMapping[internalID] = "attacker-doc"

	if err := e.Add(ctx, "victim", "hello world"); err == nil {
		t.Fatal("expected an error on id hash collision, got nil")
	}
	if e.idMapping[internalID] != "attacker-doc" {
		t.Error("colliding Add must not overwrite the existing id mapping")
	}
}

// AddBatch previously used a non-stable sort for duplicate IDs within a
// batch, so which duplicate "won" was nondeterministic. It must now be
// deterministic: the last occurrence in the input wins.
func TestEngine_AddBatch_DuplicateID_LastWins(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()

	docs := []BatchDoc{
		{ID: "dup", Text: "first version mentions alpha"},
		{ID: "dup", Text: "second version mentions beta"},
	}
	if err := e.AddBatch(ctx, docs); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	betaResults, _ := e.Search(ctx, "beta")
	foundBeta := false
	for _, r := range betaResults {
		if r.ID == "dup" {
			foundBeta = true
		}
	}
	if !foundBeta {
		t.Error("expected the last batch entry (beta) to win for duplicate ID")
	}

	alphaResults, _ := e.Search(ctx, "alpha")
	for _, r := range alphaResults {
		if r.ID == "dup" {
			t.Error("first batch entry (alpha) should have been fully overwritten")
		}
	}
}

// A blank query must return no results instead of embedding "" and
// returning whatever the fallback/embedder considers "closest to nothing".
func TestEngine_Search_EmptyQuery_ReturnsNoResults(t *testing.T) {
	e := newTestEngine()
	ctx := context.Background()
	_ = e.Add(ctx, "doc1", "hello world")

	results, err := e.Search(ctx, "")
	if err != nil {
		t.Fatalf("Search(\"\"): %v", err)
	}
	if len(results) != 0 {
		t.Errorf("expected no results for an empty query, got %d", len(results))
	}
}

// Re-indexing a document while the embedder is down must drop the doc's
// stale vector rather than leave a vector for content it no longer has.
func TestEngine_ReIndex_DropsStaleVectorOnEmbedFailure(t *testing.T) {
	cfg := config.DefaultConfig()
	ana := analysis.NewStandardAnalyzer()
	eng := NewEngine(cfg, embedding.NewDeterministicEmbedder(4), ranking.NewRRFRanker(0, 0), ana)
	ctx := context.Background()

	firstVec := []float32{1, 0, 0, 0}
	if err := eng.AddWithVector(ctx, "doc1", "original content", firstVec); err != nil {
		t.Fatalf("AddWithVector: %v", err)
	}
	if _, ok := eng.vectors.GetVectors()[docInternalID("doc1")]; !ok {
		t.Fatal("expected doc1 to have a stored vector after AddWithVector")
	}

	// Re-index without a vector; DeterministicEmbedder always errors, so this
	// exercises the embed-failure path.
	if err := eng.Add(ctx, "doc1", "updated content, embedder down"); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if _, ok := eng.vectors.GetVectors()[docInternalID("doc1")]; ok {
		t.Error("stale vector from the previous version was not dropped")
	}
}

func docInternalID(originalID string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(originalID))
	return h.Sum64()
}
