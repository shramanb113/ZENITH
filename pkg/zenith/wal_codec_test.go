package zenith

import (
	"context"
	"path/filepath"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

func TestWALValueCodec_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		text string
		vec  []float32
	}{
		{"with vector", "hello world", []float32{0.1, -0.2, 0.3}},
		{"nil vector", "no embedding available", nil},
		{"empty vector", "zero-length vector", []float32{}},
		{"empty text", "", []float32{1, 2, 3}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded := encodeWALValue(tc.text, tc.vec, nil)
			gotText, gotVec, _ := decodeWALValue(encoded)
			if gotText != tc.text {
				t.Errorf("text = %q, want %q", gotText, tc.text)
			}
			if len(gotVec) != len(tc.vec) {
				t.Fatalf("vector length = %d, want %d", len(gotVec), len(tc.vec))
			}
			for i := range tc.vec {
				if gotVec[i] != tc.vec[i] {
					t.Errorf("vector[%d] = %v, want %v", i, gotVec[i], tc.vec[i])
				}
			}
		})
	}
}

// A pre-P0-4 WAL wrote the raw text as Value with no framing at all. decode
// must treat that as legacy plain text instead of misparsing it as the new
// [tag|textLen|text|vecLen|vec] layout or crashing on out-of-range slices.
func TestWALValueCodec_LegacyPlainTextFallback(t *testing.T) {
	legacy := []byte("plain text written by an older binary")
	text, vec, _ := decodeWALValue(legacy)
	if text != string(legacy) {
		t.Errorf("text = %q, want %q", text, string(legacy))
	}
	if vec != nil {
		t.Errorf("vec = %v, want nil for a legacy value", vec)
	}
}

// A truncated/corrupt value (tag byte set but not enough bytes to hold the
// length-prefixed fields it claims) must fall back safely rather than
// slicing out of range.
func TestWALValueCodec_TruncatedValueFallsBackSafely(t *testing.T) {
	full := encodeWALValue("some text", []float32{1, 2, 3}, nil)
	for n := 0; n < len(full); n++ {
		truncated := full[:n]
		text, vec, _ := decodeWALValue(truncated)
		_ = text
		_ = vec // must not panic; that's the assertion
	}
}

// countingEmbedder counts Embed/EmbedBatch invocations. It deliberately does
// not implement embedding.Named, so index.Engine records its identity as
// "unknown" and never mismatch-checks it — this test is about P0-4 (replay
// cost), not P0-5 (identity checking), and the two must stay independent.
type countingEmbedder struct{ calls atomic.Int64 }

func (c *countingEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	c.calls.Add(1)
	return []float32{0.1, 0.2, 0.3}, nil
}

func (c *countingEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	c.calls.Add(int64(len(texts)))
	out := make([][]float32, len(texts))
	for i := range texts {
		out[i] = []float32{0.1, 0.2, 0.3}
	}
	return out, nil
}

func (c *countingEmbedder) Dimensions() int { return 3 }

// TestWALReplay_UsesStoredVectorInsteadOfReEmbedding is the direct test of
// the P0-4 claim: a WAL Put record written with encodeWALValue carries its
// document embedding, so replaying it after a crash must not call the
// embedder to reproduce it. Before this change, replay always called
// eng.Add, which re-embeds every document — for a WAL with many records,
// crash-recovery cost scaled with ONNX/Ollama inference time instead of just
// I/O.
//
// WithoutWordVectors() isolates that claim from per-token word-vector
// warming (addInternal's separate EmbedBatch call for neural expansion):
// that mechanism is deduplicated by vocabulary already (HasWordVector), so
// its replay cost is bounded by unique-term count, not document count, and
// is out of scope here — this test is about the O(N) document-embedding
// cost, not the already-sub-linear word-vector cost.
func TestWALReplay_UsesStoredVectorInsteadOfReEmbedding(t *testing.T) {
	path := filepath.Join(t.TempDir(), "replay_no_reembed.db")

	// Phase 1: clean baseline — establishes an empty gob and an empty WAL.
	baseline, err := Open(path, WithEmbedder(&countingEmbedder{}), WithoutWordVectors())
	if err != nil {
		t.Fatalf("Open baseline: %v", err)
	}
	if err := baseline.Close(); err != nil {
		t.Fatalf("Close baseline: %v", err)
	}

	// Phase 2: simulate a crash by writing a Put record directly to the WAL,
	// exactly as db.Add would have (text + a pre-computed vector), but
	// without ever calling Close (so the WAL is never reset).
	walPath := path + ".wal"
	w, _, err := wal.OpenWAL(walPath, wal.WALConfig{SyncMode: wal.SyncAlways})
	if err != nil {
		t.Fatalf("OpenWAL: %v", err)
	}
	vec := []float32{0.4, 0.5, 0.6}
	if _, err := w.Append(context.Background(), &wal.Record{
		Op: wal.OpTypePut, Key: []byte("doc1"), Value: encodeWALValue("hello world", vec, nil),
	}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close injected WAL: %v", err)
	}

	// Phase 3: reopen with a fresh call-counter and replay the WAL.
	replayEmb := &countingEmbedder{}
	db, err := Open(path, WithEmbedder(replayEmb), WithoutWordVectors())
	if err != nil {
		t.Fatalf("Open after crash: %v", err)
	}
	defer db.Close()

	if calls := replayEmb.calls.Load(); calls != 0 {
		t.Errorf("embedder called %d times during WAL replay, want 0 — the stored vector should have been reused", calls)
	}

	text, found, err := db.Get("doc1")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !found || text != "hello world" {
		t.Errorf("Get(doc1) = (%q, %v), want (\"hello world\", true) — replay must still index the document correctly", text, found)
	}
}

func TestWALValueCodec_AttrsRoundTrip(t *testing.T) {
	attrs := index.Attrs{
		"lang": {Kind: index.AttrString, S: "en"},
		"year": {Kind: index.AttrNumber, N: 2024},
		"live": {Kind: index.AttrBool, N: 1},
	}
	text, vec, got := decodeWALValue(encodeWALValue("doc body", []float32{1, 2}, attrs))
	if text != "doc body" || len(vec) != 2 {
		t.Fatalf("text/vec = %q/%v", text, vec)
	}
	if len(got) != len(attrs) {
		t.Fatalf("attrs len = %d, want %d", len(got), len(attrs))
	}
	for k, v := range attrs {
		if !reflect.DeepEqual(got[k], v) {
			t.Errorf("attrs[%q] = %+v, want %+v", k, got[k], v)
		}
	}
}

func TestWALValueCodec_TruncatedWithAttrsDoesNotPanic(t *testing.T) {
	full := encodeWALValue("some text", []float32{1, 2, 3}, index.Attrs{"k": {Kind: index.AttrString, S: "v"}})
	for n := 0; n < len(full); n++ {
		decodeWALValue(full[:n])
	}
}
