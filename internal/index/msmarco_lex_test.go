package index

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestMSMARCOLexicalLatency measures the lexical (BM25-mode, no embedder)
// query path on the real MS MARCO corpus: the first 100k passages plus the
// ground-truth passages of the qrel dev queries, so all ~6,980 queries are
// evaluable. It reports Recall@10 and latency percentiles, and is the
// regression gate for lexical-path changes: a change that alters Recall@10
// must be justified, and one that slows p50 needs a reason.
//
// Run: ZENITH_MSMARCO_LEX=1 go test ./internal/index -run TestMSMARCOLexicalLatency -v -timeout 3600s
// Optional: ZENITH_MSMARCO_QUERIES=<n> limits the number of queries timed.
// Requires bench/.cache (go run ./cmd/fetch in the bench module).
func TestMSMARCOLexicalLatency(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_LEX") == "" {
		t.Skip("set ZENITH_MSMARCO_LEX=1 to run (needs bench/.cache)")
	}
	const cacheDir = "../../bench/.cache"

	passages, augmented, queries, qrels := loadMSMARCO(t, cacheDir, 100_000)
	if n, _ := strconv.Atoi(os.Getenv("ZENITH_MSMARCO_QUERIES")); n > 0 && n < len(queries) {
		queries = queries[:n]
	}

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	// Experiment knobs: ZENITH_FUZZY_BY_LENGTH=0|1, ZENITH_PREFIX_CAP=<n> (-1 = off).
	if v := os.Getenv("ZENITH_FUZZY_BY_LENGTH"); v != "" {
		cfg.FuzzyByLength = v == "1"
	}
	if v, err := strconv.Atoi(os.Getenv("ZENITH_PREFIX_CAP")); err == nil {
		cfg.PrefixFragmentCap = v
	}
	eng := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())

	var before runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)

	batch := make([]BatchDoc, 0, len(passages)+len(augmented))
	for id, text := range passages {
		batch = append(batch, BatchDoc{ID: id, Text: text})
	}
	for id, text := range augmented {
		batch = append(batch, BatchDoc{ID: id, Text: text})
	}
	start := time.Now()
	// ZENITH_MSMARCO_INDEX=<path> caches the built index in the on-disk segment
	// format: later runs map it instead of re-indexing (seconds, not minutes) and
	// measure the memory-mapped configuration users actually run.
	idxPath := os.Getenv("ZENITH_MSMARCO_INDEX")
	loaded := false
	if idxPath != "" {
		if _, err := os.Stat(idxPath); err == nil {
			if err := eng.Load(idxPath); err != nil {
				t.Fatalf("loading cached index: %v", err)
			}
			loaded = true
		}
	}
	if !loaded {
		if err := eng.AddBatch(context.Background(), batch); err != nil {
			t.Fatal(err)
		}
		if idxPath != "" {
			if err := eng.Save(idxPath); err != nil {
				t.Fatal(err)
			}
		}
	}
	indexTime := time.Since(start)
	defer eng.Close()

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	heapMB := float64(int64(after.HeapAlloc)-int64(before.HeapAlloc)) / 1e6 // negative once Save has moved the corpus out of the heap

	ctx := context.Background()
	lat := make([]time.Duration, 0, len(queries))
	hits := 0
	for _, q := range queries {
		t0 := time.Now()
		res, err := eng.Search(ctx, q.text)
		lat = append(lat, time.Since(t0))
		if err != nil {
			t.Fatal(err)
		}
		gt := qrels[q.id]
		for i, r := range res {
			if i >= 10 {
				break
			}
			if r.ID == gt {
				hits++
				break
			}
		}
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
	pct := func(p float64) time.Duration { return lat[int(float64(len(lat)-1)*p)] }

	t.Logf("corpus=%d docs  queries=%d  index=%s  heap=%.0f MB (%.2f KB/doc)",
		len(batch), len(queries), indexTime.Round(time.Second), heapMB, heapMB*1000/float64(len(batch)))
	t.Logf("Recall@10=%.4f  p50=%s  p95=%s  p99=%s  mean=%s",
		float64(hits)/float64(len(queries)),
		pct(0.50).Round(10*time.Microsecond), pct(0.95).Round(10*time.Microsecond), pct(0.99).Round(10*time.Microsecond),
		meanDur(lat).Round(10*time.Microsecond))
	fmt.Fprintf(os.Stderr, "RESULT recall=%.4f p50=%s p95=%s\n", float64(hits)/float64(len(queries)), pct(0.50), pct(0.95))

	// Typo robustness: the same queries with one edit (delete or substitute a
	// middle letter) in their longest word of 6+ letters, judged against the same
	// ground truth. Fuzzy-matching changes are measured here as well as on clean
	// queries, since MS MARCO's queries have almost no typos of their own.
	tlat := make([]time.Duration, 0, len(queries))
	thits, tn := 0, 0
	for _, q := range queries {
		tq, ok := withTypo(q.id, q.text)
		if !ok {
			continue
		}
		t0 := time.Now()
		res, err := eng.Search(ctx, tq)
		tlat = append(tlat, time.Since(t0))
		if err != nil {
			t.Fatal(err)
		}
		tn++
		for i, r := range res {
			if i >= 10 {
				break
			}
			if r.ID == qrels[q.id] {
				thits++
				break
			}
		}
	}
	if tn > 0 {
		sort.Slice(tlat, func(i, j int) bool { return tlat[i] < tlat[j] })
		tp := func(p float64) time.Duration { return tlat[int(float64(len(tlat)-1)*p)] }
		t.Logf("TypoRecall@10=%.4f over %d typo'd queries  p50=%s p95=%s", float64(thits)/float64(tn), tn, tp(0.50).Round(10*time.Microsecond), tp(0.95).Round(10*time.Microsecond))
		fmt.Fprintf(os.Stderr, "RESULT typo_recall=%.4f typo_p50=%s\n", float64(thits)/float64(tn), tp(0.50))
	}
	runtime.KeepAlive(eng)
}

func meanDur(d []time.Duration) time.Duration {
	var s time.Duration
	for _, x := range d {
		s += x
	}
	return s / time.Duration(len(d))
}

// withTypo applies one edit to the longest word of 6+ letters in text.
func withTypo(id, text string) (string, bool) {
	words := strings.Fields(text)
	best := -1
	for i, w := range words {
		if len(w) < 6 || (best >= 0 && len(w) <= len(words[best])) {
			continue
		}
		letters := true
		for _, r := range w {
			if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
				letters = false
				break
			}
		}
		if letters {
			best = i
		}
	}
	if best < 0 {
		return "", false
	}
	w := []rune(words[best])
	mid := len(w) / 2
	switch {
	case internalIDOf(id)%2 == 0:
		w = append(w[:mid], w[mid+1:]...) // delete a middle letter
	case w[mid] == 'q':
		w[mid] = 'z'
	default:
		w[mid] = 'q' // substitute a middle letter
	}
	words[best] = string(w)
	return strings.Join(words, " "), true
}
