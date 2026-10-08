package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestLexicalLatencyGate is the CI regression gate for the lexical query path.
// It runs on a deterministic synthetic corpus (no download), sized so an
// accidental return to exhaustive candidate expansion is unmistakable, and
// asserts three things:
//
//  1. Recall@10 on clean and on one-typo queries stays above a floor — a
//     speed-up that costs relevance fails.
//  2. Work per query — the number of lexical candidates — stays under a bound.
//     This is deterministic (it depends only on the corpus and the algorithm),
//     so it catches algorithmic regressions without wall-clock noise.
//  3. p50 latency stays under a generous ceiling, catching gross slowdowns
//     that leave candidate counts alone (a quadratic step, a lost index, ...).
//
// The real-corpus numbers (MS MARCO) are measured by TestMSMARCOLexicalLatency;
// this test guards against regressions between those manual runs.
//
// Overrides for unusual hardware: ZENITH_GATE_P50_MS, ZENITH_GATE_SCALE (multiplies
// the latency ceiling). ZENITH_GATE=off skips it.
func TestLexicalLatencyGate(t *testing.T) {
	if testing.Short() || os.Getenv("ZENITH_GATE") == "off" {
		t.Skip("latency gate skipped")
	}
	if raceEnabled {
		// -race slows the engine 5-10x, which would make the wall-clock ceiling
		// meaningless; the gate runs (deterministically) on the non-race legs.
		t.Skip("latency gate is timing-sensitive; skipped under -race")
	}
	const (
		nDocs    = 30_000
		nQueries = 200
	)
	words := synthVocab(6000)
	r := rand.New(rand.NewSource(1))
	zipf := rand.NewZipf(r, 1.15, 4, uint64(len(words)-1))

	docs := make([]BatchDoc, nDocs)
	docWords := make([][]string, nDocs)
	df := make(map[string]int)
	for i := range docs {
		n := 12 + r.Intn(14)
		ws := make([]string, n)
		seen := map[string]bool{}
		text := ""
		for j := range ws {
			w := words[zipf.Uint64()]
			ws[j] = w
			text += w + " "
			if !seen[w] {
				seen[w] = true
				df[w]++
			}
		}
		docs[i] = BatchDoc{ID: fmt.Sprintf("d%06d", i), Text: text}
		docWords[i] = ws
	}

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	// Experiment knobs, to confirm the gate really fails on the old behaviour:
	// ZENITH_FUZZY_BY_LENGTH=0, ZENITH_PREFIX_CAP=-1.
	if v := os.Getenv("ZENITH_FUZZY_BY_LENGTH"); v != "" {
		cfg.FuzzyByLength = v == "1"
	}
	if v, err := strconv.Atoi(os.Getenv("ZENITH_PREFIX_CAP")); err == nil {
		cfg.PrefixFragmentCap = v
	}
	eng := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	t.Cleanup(func() { eng.Close() })
	if err := eng.AddBatch(context.Background(), docs); err != nil {
		t.Fatal(err)
	}

	// Queries: the three rarest distinct words of a random document. Ground
	// truth is that document.
	type q struct {
		text, typo, id string
	}
	qs := make([]q, 0, nQueries)
	for len(qs) < nQueries {
		d := r.Intn(nDocs)
		uniq := map[string]bool{}
		var cand []string
		for _, w := range docWords[d] {
			if !uniq[w] {
				uniq[w] = true
				cand = append(cand, w)
			}
		}
		if len(cand) < 3 {
			continue
		}
		sort.Slice(cand, func(i, j int) bool {
			if df[cand[i]] != df[cand[j]] {
				return df[cand[i]] < df[cand[j]]
			}
			return cand[i] < cand[j]
		})
		// two rare words identify the document; a short common word (which real
		// queries always contain) exercises the expansion-prone path.
		common := ""
		for _, w := range docWords[d] {
			if len(w) <= 4 && df[w] > nDocs/50 {
				common = w
				break
			}
		}
		if common == "" {
			continue
		}
		pick := []string{cand[0], cand[1], common}
		clean := pick[0] + " " + pick[1] + " " + pick[2]
		// typo: drop the middle letter of the longest rare word
		li := 0
		if len(pick[1]) > len(pick[0]) {
			li = 1
		}
		w := []rune(pick[li])
		if len(w) < 6 {
			continue // only words long enough to be a plausible typo target
		}
		w = append(w[:len(w)/2], w[len(w)/2+1:]...)
		typoPick := append([]string(nil), pick...)
		typoPick[li] = string(w)
		qs = append(qs, q{text: clean, typo: typoPick[0] + " " + typoPick[1] + " " + typoPick[2], id: docs[d].ID})
	}

	ctx := context.Background()
	run := func(pick func(q) string) (recall float64, lat []time.Duration) {
		hits := 0
		for _, x := range qs {
			t0 := time.Now()
			res, err := eng.Search(ctx, pick(x))
			lat = append(lat, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			for i, rr := range res {
				if i >= 10 {
					break
				}
				if rr.ID == x.id {
					hits++
					break
				}
			}
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		return float64(hits) / float64(len(qs)), lat
	}
	recall, lat := run(func(x q) string { return x.text })
	typoRecall, _ := run(func(x q) string { return x.typo })

	// Work per query: candidates produced by the lexical pass.
	var cand int
	for _, x := range qs {
		toks := eng.analyzer.(analysis.QueryAnalyzer).AnalyzeQuery(x.text)
		raw := make([]string, 0, len(toks))
		for _, tk := range toks {
			raw = append(raw, tk.Term)
		}
		eng.mu.RLock()
		cand += len(eng.lexicalPass(raw, eng.config.PhoneticWeight))
		eng.mu.RUnlock()
	}
	avgCand := float64(cand) / float64(len(qs))
	p50 := lat[len(lat)/2]

	t.Logf("gate corpus: %d docs, %d queries: Recall@10=%.3f  TypoRecall@10=%.3f  candidates/query=%.0f  p50=%s p95=%s",
		nDocs, len(qs), recall, typoRecall, avgCand, p50.Round(10*time.Microsecond), lat[len(lat)*95/100].Round(10*time.Microsecond))

	if recall < gateRecallFloor {
		t.Errorf("Recall@10 = %.3f, floor %.3f — the lexical path lost relevance", recall, gateRecallFloor)
	}
	if typoRecall < gateTypoRecallFloor {
		t.Errorf("TypoRecall@10 = %.3f, floor %.3f — fuzzy matching lost typo tolerance", typoRecall, gateTypoRecallFloor)
	}
	if avgCand > gateMaxCandidates {
		t.Errorf("%.0f candidates per query, bound %.0f — candidate generation regressed toward exhaustive expansion", avgCand, gateMaxCandidates)
	}
	ceiling := gateP50Ceiling
	if v, err := strconv.ParseFloat(os.Getenv("ZENITH_GATE_P50_MS"), 64); err == nil {
		ceiling = time.Duration(v * float64(time.Millisecond))
	}
	if s, err := strconv.ParseFloat(os.Getenv("ZENITH_GATE_SCALE"), 64); err == nil && s > 0 {
		ceiling = time.Duration(float64(ceiling) * s)
	}
	if p50 > ceiling {
		t.Errorf("p50 = %s, ceiling %s", p50, ceiling)
	}
}

// synthVocab builds n distinct pronounceable pseudo-words in frequency-rank
// order. Like natural language, the most frequent words are the shortest (2–4
// letters, which have huge edit-distance neighbourhoods) and word length grows
// with rank; consonant-vowel syllables give realistic prefix sharing.
func synthVocab(n int) []string {
	cons := []byte("bcdfghjklmnprstvwz")
	vow := []byte("aeiou")
	r := rand.New(rand.NewSource(7))
	seen := map[string]bool{}
	out := make([]string, 0, n)
	for len(out) < n {
		rank := len(out)
		var syl int
		switch {
		case rank < 250:
			syl = 1 + r.Intn(2)
		case rank < 1500:
			syl = 2 + r.Intn(2)
		default:
			syl = 3 + r.Intn(3)
		}
		b := make([]byte, 0, syl*2+1)
		for k := 0; k < syl; k++ {
			b = append(b, cons[r.Intn(len(cons))], vow[r.Intn(len(vow))])
		}
		if r.Intn(2) == 0 {
			b = append(b, cons[r.Intn(len(cons))])
		}
		w := string(b)
		if !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	return out
}

// Gate thresholds, from measurements on this corpus: current defaults give
// Recall@10 1.000, typo Recall@10 0.885, ~9.2k candidates/query, p50 ~13ms; the
// pre-optimisation behaviour (fixed edit distance, exhaustive prefixes) gives
// ~20.8k candidates and p50 ~36ms, which the candidate bound rejects. Recall floors and the candidate bound are set from measured
// results with headroom; the latency ceiling is deliberately loose (several
// times the measured value) so slow CI runners do not flake.
var (
	gateRecallFloor     = 0.95
	gateTypoRecallFloor = 0.80
	gateMaxCandidates   = 13_000.0
	gateP50Ceiling      = 50 * time.Millisecond
)

var _ = math.Abs
