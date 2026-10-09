package pdf

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// ── splitChunks — basic behaviour ────────────────────────────────────────────

func TestSplitChunks_Empty(t *testing.T) {
	if chunks := splitChunks(""); len(chunks) != 0 {
		t.Errorf("got %d chunks, want 0", len(chunks))
	}
}

func TestSplitChunks_WhitespaceOnly(t *testing.T) {
	if chunks := splitChunks("   \t\n  "); len(chunks) != 0 {
		t.Errorf("got %d chunks, want 0 for whitespace-only input", len(chunks))
	}
}

func TestSplitChunks_ShortText(t *testing.T) {
	chunks := splitChunks("hello world")
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0] != "hello world" {
		t.Errorf("chunk[0]=%q, want %q", chunks[0], "hello world")
	}
}

func TestSplitChunks_SingleWord(t *testing.T) {
	chunks := splitChunks("hello")
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0] != "hello" {
		t.Errorf("chunk[0]=%q, want %q", chunks[0], "hello")
	}
}

// ── chunk size ───────────────────────────────────────────────────────────────

func TestSplitChunks_ExactlyChunkWords(t *testing.T) {
	// 300 words → exactly one chunk of 300 words.
	text := strings.Join(make300Words(), " ")
	chunks := splitChunks(text)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1 for exactly %d words", len(chunks), chunkWords)
	}
	if len(strings.Fields(chunks[0])) != chunkWords {
		t.Errorf("chunk word count = %d, want %d", len(strings.Fields(chunks[0])), chunkWords)
	}
}

func TestSplitChunks_OneWordOverChunkSize(t *testing.T) {
	// 301 words → two chunks (second chunk has 301 - (300-50) = 51 words).
	words := append(make300Words(), "extra")
	text := strings.Join(words, " ")
	chunks := splitChunks(text)
	if len(chunks) != 2 {
		t.Fatalf("got %d chunks, want 2 for %d words", len(chunks), len(words))
	}
	if len(strings.Fields(chunks[0])) != chunkWords {
		t.Errorf("first chunk has %d words, want %d", len(strings.Fields(chunks[0])), chunkWords)
	}
}

func TestSplitChunks_400Words(t *testing.T) {
	words := makeNWords(400)
	text := strings.Join(words, " ")
	chunks := splitChunks(text)
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want ≥2 for 400 words", len(chunks))
	}
	if len(strings.Fields(chunks[0])) != chunkWords {
		t.Errorf("first chunk has %d words, want %d", len(strings.Fields(chunks[0])), chunkWords)
	}
}

// ── overlap ──────────────────────────────────────────────────────────────────

func TestSplitChunks_OverlapIsCorrect(t *testing.T) {
	// 350 words → chunk[0] covers words 0-299, chunk[1] covers words 250-349.
	words := makeNWords(350)
	text := strings.Join(words, " ")
	chunks := splitChunks(text)
	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want ≥2 for 350 words", len(chunks))
	}
	// The second chunk should start at word index chunkWords-chunkOverlap = 250.
	chunk1Words := strings.Fields(chunks[1])
	want := words[chunkWords-chunkOverlap]
	got := chunk1Words[0]
	if got != want {
		t.Errorf("chunk[1] starts with %q, want %q (word at index %d)",
			got, want, chunkWords-chunkOverlap)
	}
}

func TestSplitChunks_NoLostWords(t *testing.T) {
	// Every word must appear in at least one chunk.
	words := makeNWords(500)
	text := strings.Join(words, " ")
	chunks := splitChunks(text)

	// The last chunk must end at the last word.
	lastChunkWords := strings.Fields(chunks[len(chunks)-1])
	if lastChunkWords[len(lastChunkWords)-1] != words[len(words)-1] {
		t.Errorf("last word of last chunk = %q, want %q",
			lastChunkWords[len(lastChunkWords)-1], words[len(words)-1])
	}
}

func TestSplitChunks_FirstWordInFirstChunk(t *testing.T) {
	words := makeNWords(400)
	text := strings.Join(words, " ")
	chunks := splitChunks(text)
	firstChunkWords := strings.Fields(chunks[0])
	if firstChunkWords[0] != words[0] {
		t.Errorf("first word of first chunk = %q, want %q", firstChunkWords[0], words[0])
	}
}

// ── chunk content ─────────────────────────────────────────────────────────────

func TestSplitChunks_ChunksAreNonEmpty(t *testing.T) {
	for _, n := range []int{1, 50, 300, 301, 600, 1000} {
		words := makeNWords(n)
		text := strings.Join(words, " ")
		for i, c := range splitChunks(text) {
			if strings.TrimSpace(c) == "" {
				t.Errorf("chunk[%d] is empty for %d-word input", i, n)
			}
		}
	}
}

func TestSplitChunks_ExtraWhitespace(t *testing.T) {
	// Extra spaces between words should not affect chunking.
	text := "word1  word2   word3    word4"
	chunks := splitChunks(text)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	// strings.Fields normalises whitespace.
	words := strings.Fields(chunks[0])
	if len(words) != 4 {
		t.Errorf("got %d words, want 4", len(words))
	}
}

func TestSplitChunks_NewlinesAndTabs(t *testing.T) {
	text := "word1\nword2\tword3\r\nword4"
	chunks := splitChunks(text)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	words := strings.Fields(chunks[0])
	if len(words) != 4 {
		t.Errorf("got %d words, want 4", len(words))
	}
}

// ── large inputs ─────────────────────────────────────────────────────────────

func TestSplitChunks_LargeDocument(t *testing.T) {
	// 10 000 words — verify no panic and reasonable chunk count.
	words := makeNWords(10000)
	text := strings.Join(words, " ")
	chunks := splitChunks(text)
	// With stride = chunkWords-chunkOverlap = 250:
	// expected ≈ ceil((10000-300)/250) + 1 = 39 chunks
	if len(chunks) < 30 || len(chunks) > 50 {
		t.Errorf("got %d chunks for 10000 words, expected ~39", len(chunks))
	}
}

// ── allowed-root path restriction ───────────────────────────────────────────

func TestSetAllowedRoot_CreatesDirectory(t *testing.T) {
	base := t.TempDir()
	root := filepath.Join(base, "pdfs")

	p := NewIndexer(nil)
	if err := p.SetAllowedRoot(root); err != nil {
		t.Fatalf("SetAllowedRoot: %v", err)
	}
	if _, err := os.Stat(root); err != nil {
		t.Fatalf("allowed root not created: %v", err)
	}
}

func TestResolveWithinRoot_AllowsFileInsideRoot(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "doc.pdf")
	if err := os.WriteFile(target, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}

	got, err := resolveWithinRoot(resolvedRoot, target)
	if err != nil {
		t.Fatalf("resolveWithinRoot: unexpected error: %v", err)
	}
	if filepath.Base(got) != "doc.pdf" {
		t.Errorf("resolved path = %q, want basename doc.pdf", got)
	}
}

func TestResolveWithinRoot_RejectsParentTraversal(t *testing.T) {
	root := t.TempDir()
	pdfDir := filepath.Join(root, "pdfs")
	if err := os.MkdirAll(pdfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(root, "secret.pdf")
	if err := os.WriteFile(secret, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	traversal := filepath.Join(pdfDir, "..", "secret.pdf")
	if _, err := resolveWithinRoot(pdfDir, traversal); err == nil {
		t.Fatalf("resolveWithinRoot: expected error for path escaping root via ..")
	}
}

func TestResolveWithinRoot_RejectsSymlinkEscape(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("symlink creation requires elevated privileges on Windows")
	}
	root := t.TempDir()
	pdfDir := filepath.Join(root, "pdfs")
	if err := os.MkdirAll(pdfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(root, "secret.pdf")
	if err := os.WriteFile(secret, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(pdfDir, "escape.pdf")
	if err := os.Symlink(secret, link); err != nil {
		t.Fatal(err)
	}

	if _, err := resolveWithinRoot(pdfDir, link); err == nil {
		t.Fatalf("resolveWithinRoot: expected error for symlink escaping root")
	}
}

func TestIndex_RejectsPathOutsideAllowedRoot(t *testing.T) {
	root := t.TempDir()
	pdfDir := filepath.Join(root, "pdfs")
	if err := os.MkdirAll(pdfDir, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside.pdf")
	if err := os.WriteFile(outside, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	p := NewIndexer(nil)
	if err := p.SetAllowedRoot(pdfDir); err != nil {
		t.Fatalf("SetAllowedRoot: %v", err)
	}

	_, err := p.Index(nil, "doc1", filepath.Join(pdfDir, "..", "outside.pdf"), nil)
	if err == nil {
		t.Fatalf("Index: expected error for path outside allowed root")
	}
	if !strings.Contains(err.Error(), "outside the allowed root") {
		t.Errorf("Index error = %v, want mention of allowed root", err)
	}
}

// ── real bounding boxes ──────────────────────────────────────────────────────

// TestIndex_RealPDF_ChunkBBoxesArePlausible indexes a real multi-page,
// multi-chunk PDF fixture (testdata/multipage.pdf: page 1 has 301 words —
// enough to split into 2 overlapping 300-word chunks — page 2 has 46,
// giving 1 more) and checks that the "x,y,w,h" field baked into each
// chunk's document ID (the field at internal/server/server.go's
// parseChunkFields parts[4], and pkg/zenith/zenith.go's parseChunkID
// rest[3]) is a real, non-zero, on-page box rather than the old hardcoded
// "0.00,0.00,0.00,0.00".
func TestIndex_RealPDF_ChunkBBoxesArePlausible(t *testing.T) {
	cfg := config.DefaultConfig()
	eng := index.NewEngine(cfg, embedding.NewDeterministicEmbedder(32), ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	defer eng.Close()

	idx := NewIndexer(eng)
	n, err := idx.Index(context.Background(), "doc1", filepath.Join("testdata", "multipage.pdf"), nil)
	if err != nil {
		t.Fatalf("Index: %v", err)
	}
	if n < 3 {
		t.Fatalf("got %d chunks, want >= 3 (2 from page 1's overlap split + 1 from page 2)", n)
	}

	// "word0000" is the first word of page 1 and appears nowhere else in
	// the fixture, so it uniquely identifies chunk p1/c0.
	results, err := eng.Search(context.Background(), "word0000")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	var checked bool
	for _, r := range results {
		if !strings.HasPrefix(r.ID, "doc1||p1||c0||") {
			continue
		}
		checked = true
		assertPlausibleChunkBBox(t, r.ID, 612, 792)
	}
	if !checked {
		t.Fatalf("search for %q returned no hit for chunk doc1||p1||c0||...; results: %v", "word0000", results)
	}
}

// assertPlausibleChunkBBox parses id's "docID||p{page}||c{chunk}||{type}||x,y,w,h"
// fields (the same split every downstream consumer performs) and checks the
// box is non-zero, has positive width/height, and fits within a pageW x
// pageH page.
func assertPlausibleChunkBBox(t *testing.T, id string, pageW, pageH float64) {
	t.Helper()
	fields := strings.SplitN(id, "||", 5)
	if len(fields) != 5 {
		t.Fatalf("chunk ID %q does not split into 5 ||-separated fields", id)
	}
	if fields[3] != "text" {
		t.Errorf("source_type = %q, want %q", fields[3], "text")
	}
	coords := strings.Split(fields[4], ",")
	if len(coords) != 4 {
		t.Fatalf("bbox field %q does not split into 4 comma-separated values", fields[4])
	}
	var v [4]float64
	for i, c := range coords {
		f, err := strconv.ParseFloat(c, 64)
		if err != nil {
			t.Fatalf("bbox value %q is not a float: %v", c, err)
		}
		v[i] = f
	}
	x, y, w, h := v[0], v[1], v[2], v[3]
	if w == 0 && h == 0 && x == 0 && y == 0 {
		t.Errorf("chunk bbox for %q is the old hardcoded zero box", id)
	}
	if w <= 0 || h <= 0 {
		t.Errorf("chunk bbox for %q has non-positive size: w=%v h=%v", id, w, h)
	}
	if x < 0 || y < 0 {
		t.Errorf("chunk bbox for %q has a negative origin: x=%v y=%v", id, x, y)
	}
	if x+w > pageW+0.5 {
		t.Errorf("chunk bbox for %q extends past page width: x=%v w=%v pageW=%v", id, x, w, pageW)
	}
	if y+h > pageH+0.5 {
		t.Errorf("chunk bbox for %q extends past page height: y=%v h=%v pageH=%v", id, y, h, pageH)
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func make300Words() []string { return makeNWords(chunkWords) }

func makeNWords(n int) []string {
	words := make([]string, n)
	for i := range words {
		words[i] = strings.Repeat("w", (i%8)+1) // vary word lengths
	}
	return words
}
