package index

import (
	"os"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// Diagnostic: how many candidates does the lexical pass produce, and how many
// of them have a real BM25 hit? Skipped unless a cached MS MARCO index exists.
func TestCandidateStats(t *testing.T) {
	idx := os.Getenv("ZENITH_MSMARCO_INDEX")
	if os.Getenv("ZENITH_MSMARCO_LEX") == "" || idx == "" {
		t.Skip("needs ZENITH_MSMARCO_LEX=1 and ZENITH_MSMARCO_INDEX")
	}
	_, _, queries, _ := loadMSMARCO(t, "../../bench/.cache", 100_000)
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	if err := e.Load(idx); err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	var cand, withBM25, n, nf, np, nz int
	for _, q := range queries[:200] {
		toks := e.analyzer.(analysis.QueryAnalyzer).AnalyzeQuery(q.text)
		raw := make([]string, 0, len(toks))
		for _, tk := range toks {
			raw = append(raw, tk.Term)
		}
		e.mu.RLock()
		kw := e.lexicalPass(raw, e.config.PhoneticWeight)
		bm := e.bm25.Query(raw)
		e.mu.RUnlock()
		cand += len(kw)
		// Per-source candidate counts.
		e.mu.RLock()
		frag, phon, fuzz := map[uint64]bool{}, map[uint64]bool{}, map[uint64]bool{}
		for _, token := range raw {
			Q := len(token)
			frags := []string{token}
			if Q >= 3 {
				frags = generateEdgeNgrams(token)
			}
			for _, f := range frags {
				if len(f) < Q && e.prefixCap() > 0 && e.fragCount(f) > e.prefixCap() {
					continue
				}
				e.eachFragDoc(f, func(id uint64) { frag[id] = true })
			}
			if c := analysis.Soundex(token); c != "" {
				e.eachPhonDoc(c, func(id uint64) { phon[id] = true })
			}
			if Q >= 2 {
				for _, m := range e.fuzzyMatches(token) {
					if m.Distance > 0 {
						e.eachFragDoc(m.Word, func(id uint64) { fuzz[id] = true })
					}
				}
			}
		}
		e.mu.RUnlock()
		nf, np, nz = nf+len(frag), np+len(phon), nz+len(fuzz)
		withBM25 += len(bm)
		n++
	}
	t.Logf("by source (union per query): prefix-fragments=%d phonetic=%d fuzzy=%d", nf/n, np/n, nz/n)
	t.Logf("avg per query: %d lexical candidates, of which %d have a BM25 hit (corpus %d docs)", cand/n, withBM25/n, e.Count())
}
