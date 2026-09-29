package image

import (
	"context"
	"strings"
	"testing"
)

// ── OCR availability / stub contract ─────────────────────────────────────────

// This package is built without -tags ocr in normal `go test`, so the stub
// (ocr_stub.go) is what's under test here: OCR must report unavailable and
// extractOCRText must be a true no-op (no text, no error) so callers always
// fall back to filename-only indexing.
func TestOCRAvailable_FalseWithoutOCRTag(t *testing.T) {
	if OCRAvailable() {
		t.Fatal("OCRAvailable() = true in a build without -tags ocr")
	}
}

func TestExtractOCRText_StubReturnsNoTextNoError(t *testing.T) {
	text, err := extractOCRText("/does/not/exist.png")
	if err != nil {
		t.Fatalf("extractOCRText (stub) returned error %v, want nil", err)
	}
	if text != "" {
		t.Fatalf("extractOCRText (stub) = %q, want empty", text)
	}
}

// ── Index() fallback behaviour ───────────────────────────────────────────────

func TestIndex_EmptyPath_ReturnsZeroWithoutTouchingEngine(t *testing.T) {
	idx := NewIndexer(nil)
	n, err := idx.Index(context.Background(), "doc1", "")
	if err != nil {
		t.Fatalf("Index(\"\") error = %v, want nil", err)
	}
	if n != 0 {
		t.Fatalf("Index(\"\") = %d, want 0", n)
	}
}

// ── splitOCRChunks ────────────────────────────────────────────────────────────

func TestSplitOCRChunks_Empty(t *testing.T) {
	if got := splitOCRChunks(""); got != nil {
		t.Errorf("splitOCRChunks(\"\") = %v, want nil", got)
	}
}

func TestSplitOCRChunks_WhitespaceOnly(t *testing.T) {
	if got := splitOCRChunks("   \n\t  "); got != nil {
		t.Errorf("splitOCRChunks(whitespace) = %v, want nil", got)
	}
}

func TestSplitOCRChunks_ShortText(t *testing.T) {
	got := splitOCRChunks("a scanned receipt from the corner store")
	if len(got) != 1 {
		t.Fatalf("splitOCRChunks(short) = %d chunks, want 1", len(got))
	}
}

func TestSplitOCRChunks_NoLostWords(t *testing.T) {
	words := makeNOCRWords(700)
	text := strings.Join(words, " ")
	chunks := splitOCRChunks(text)

	seen := make(map[string]bool)
	for _, c := range chunks {
		for _, w := range strings.Fields(c) {
			seen[w] = true
		}
	}
	for _, w := range words {
		if !seen[w] {
			t.Fatalf("word %q lost from chunks", w)
		}
	}
}

func TestSplitOCRChunks_OverlapIsCorrect(t *testing.T) {
	words := makeNOCRWords(400)
	chunks := splitOCRChunks(strings.Join(words, " "))
	if len(chunks) < 2 {
		t.Fatalf("expected at least 2 chunks for 400 words, got %d", len(chunks))
	}
	firstWords := strings.Fields(chunks[0])
	secondWords := strings.Fields(chunks[1])
	// The overlap words at the tail of chunk 0 should reappear at the head of chunk 1.
	overlapStart := len(firstWords) - ocrChunkOverlap
	for i := 0; i < ocrChunkOverlap; i++ {
		if firstWords[overlapStart+i] != secondWords[i] {
			t.Fatalf("overlap mismatch at %d: %q vs %q", i, firstWords[overlapStart+i], secondWords[i])
		}
	}
}

func TestSplitOCRChunks_ChunksAreNonEmpty(t *testing.T) {
	words := makeNOCRWords(900)
	for _, c := range splitOCRChunks(strings.Join(words, " ")) {
		if strings.TrimSpace(c) == "" {
			t.Error("splitOCRChunks produced an empty chunk")
		}
	}
}

func makeNOCRWords(n int) []string {
	words := make([]string, n)
	for i := range words {
		words[i] = "word" + string(rune('a'+i%26))
	}
	return words
}
