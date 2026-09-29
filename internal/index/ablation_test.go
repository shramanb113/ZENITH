package index

import (
	"context"
	"fmt"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestMSMARCOAblation measures what each lexical signal contributes, on real
// MS MARCO queries — clean and with one typo — by switching one signal off at a
// time over the same memory-mapped index. It needs the index the lexical
// benchmark caches:
//
//	ZENITH_MSMARCO_LEX=1 ZENITH_MSMARCO_INDEX=../../bench/.cache/lex-index.db go test ./internal/index -run TestMSMARCOLexicalLatency
//	ZENITH_MSMARCO_ABLATE=1 ZENITH_MSMARCO_INDEX=../../bench/.cache/lex-index.db go test ./internal/index -run TestMSMARCOAblation -v
//
// Signals: prefix n-gram expansion, phonetic (Soundex) matches, fuzzy
// (edit-distance) matches; BM25 always ranks. "old fuzzy" is the pre-optimisation
// rule (edit distance 2 for every word length).
func TestMSMARCOAblation(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_ABLATE") == "" {
		t.Skip("set ZENITH_MSMARCO_ABLATE=1 and ZENITH_MSMARCO_INDEX=<cached lexical index> to run")
	}
	const cacheDir = "../../bench/.cache"
	idxPath := os.Getenv("ZENITH_MSMARCO_INDEX")
	if _, err := os.Stat(idxPath); idxPath == "" || err != nil {
		t.Skip("cached lexical index missing (see the test's doc comment)")
	}
	_, _, queries, qrels := loadMSMARCO(t, cacheDir, 100_000)

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	if err := eng.Load(idxPath); err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	ctx := context.Background()
	base := *cfg

	type variant struct {
		name string
		set  func(c *config.Config)
	}
	variants := []variant{
		{"full (default)", func(c *config.Config) {}},
		{"no fuzzy", func(c *config.Config) { c.FuzzyMaxDist = 0 }},
		{"no phonetic", func(c *config.Config) { c.PhoneticWeight = 0 }},
		{"no prefix expansion", func(c *config.Config) { c.PrefixFragmentCap = 1 }},
		{"exact terms + BM25 only", func(c *config.Config) { c.FuzzyMaxDist = 0; c.PhoneticWeight = 0; c.PrefixFragmentCap = 1 }},
		{"old fuzzy (distance 2 for every word)", func(c *config.Config) { c.FuzzyByLength = false }},
	}

	type result struct {
		name                      string
		clean, typo               float64
		cleanP50, typoP50         time.Duration
		nClean, nTypo, hitsClean int
	}
	var out []result
	for _, v := range variants {
		*cfg = base
		v.set(cfg)
		var r result
		r.name = v.name
		var cl, tl []time.Duration
		hitsT := 0
		for _, q := range queries {
			t0 := time.Now()
			res, err := eng.Search(ctx, q.text)
			cl = append(cl, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			r.nClean++
			for i, x := range res {
				if i >= 10 {
					break
				}
				if x.ID == qrels[q.id] {
					r.hitsClean++
					break
				}
			}
			tq, ok := withTypo(q.id, q.text)
			if !ok {
				continue
			}
			t0 = time.Now()
			res, err = eng.Search(ctx, tq)
			tl = append(tl, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			r.nTypo++
			for i, x := range res {
				if i >= 10 {
					break
				}
				if x.ID == qrels[q.id] {
					hitsT++
					break
				}
			}
		}
		sort.Slice(cl, func(i, j int) bool { return cl[i] < cl[j] })
		sort.Slice(tl, func(i, j int) bool { return tl[i] < tl[j] })
		r.clean = float64(r.hitsClean) / float64(r.nClean)
		r.typo = float64(hitsT) / float64(r.nTypo)
		r.cleanP50, r.typoP50 = cl[len(cl)/2], tl[len(tl)/2]
		out = append(out, r)
		t.Logf("%-40s clean Recall@10 %.4f (p50 %s)   typo Recall@10 %.4f (p50 %s)   [%d clean / %d typo queries]",
			r.name, r.clean, r.cleanP50.Round(10*time.Microsecond), r.typo, r.typoP50.Round(10*time.Microsecond), r.nClean, r.nTypo)
		fmt.Fprintf(os.Stderr, "ABLATION %s | clean=%.4f p50=%s | typo=%.4f p50=%s\n", r.name, r.clean, r.cleanP50, r.typo, r.typoP50)
	}
	full := out[0]
	for _, r := range out[1:] {
		t.Logf("%-40s vs full: clean %+.4f, typo %+.4f", r.name, r.clean-full.clean, r.typo-full.typo)
	}
}
