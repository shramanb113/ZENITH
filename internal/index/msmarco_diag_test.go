//go:build cgo

package index

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestMSMARCORecallDiag instruments hybrid recall on MS MARCO 100k: for every
// qrel-filtered query it records the ground-truth document's rank in the
// dense (vector) list and the BM25-ranked lexical list independently, then
// simulates RRF fusion variants offline from the recorded ranks.
//
// Run with:
//
//	ZENITH_MSMARCO_DIAG=1 go test ./internal/index -run TestMSMARCORecallDiag -v -timeout 3600s
//
// The first run indexes 100k passages (~15 min) and saves the engine state to
// bench/.cache/diag-engine.gob; later runs load it in seconds.
func TestMSMARCORecallDiag(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_DIAG") == "" {
		t.Skip("set ZENITH_MSMARCO_DIAG=1 to run (long-running diagnostic)")
	}
	const cacheDir = `..\..\bench\.cache`
	const enginePath = cacheDir + `\diag-engine.gob`
	const augPath = cacheDir + `\diag-engine-aug.gob`
	const nDocs = 100_000

	// passages = first 100k (the official benchmark corpus). augmented = the
	// ground-truth passages for the remaining qrel queries, pulled from deeper
	// in the collection. Indexing 100k+gt makes all ~6,980 queries evaluable
	// instead of 32 — the official corpus leaves a 32-query recall denominator
	// (steps of 0.031), far too coarse to tune fusion parameters against.
	passages, augmented, queries, qrels := loadMSMARCO(t, cacheDir, nDocs)
	t.Logf("loaded %d passages + %d gt-augmented, %d evaluable queries",
		len(passages), len(augmented), len(queries))

	localEmb, err := localembedder.New()
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}
	emb, err := embedding.NewCachingEmbedder(localEmb, 10_000)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	tkz := analysis.NewStandardAnalyzer()
	// Mirror pkg/zenith.Open wiring so fused numbers match production.
	eng := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), tkz)

	if err := eng.Load(augPath); err == nil {
		t.Log("loaded augmented engine state")
	} else {
		if err := eng.Load(enginePath); err != nil {
			t.Logf("no saved engine (%v) — indexing %d docs", err, len(passages))
			batch := make([]BatchDoc, 0, len(passages))
			for id, text := range passages {
				batch = append(batch, BatchDoc{ID: id, Text: text})
			}
			if err := eng.AddBatch(context.Background(), batch); err != nil {
				t.Fatalf("AddBatch: %v", err)
			}
			if err := eng.Save(enginePath); err != nil {
				t.Logf("save failed (diag continues without persistence): %v", err)
			}
		} else {
			t.Log("loaded 100k engine state")
		}
		augBatch := make([]BatchDoc, 0, len(augmented))
		for id, text := range augmented {
			augBatch = append(augBatch, BatchDoc{ID: id, Text: text})
		}
		if err := eng.AddBatch(context.Background(), augBatch); err != nil {
			t.Fatalf("AddBatch augmented: %v", err)
		}
		if err := eng.Save(augPath); err != nil {
			t.Logf("save failed (diag continues without persistence): %v", err)
		}
	}

	// BM25 variants evaluated against the same built index in one pass:
	// query tokens with vs without synonym expansion × k1/b parameter sets
	// (1.2/0.75 = shipping default, 0.9/0.4 = Anserini default,
	// 0.82/0.68 = standard MS MARCO-tuned values).
	type bm25Variant struct {
		name  string
		base  bool // true = exclude SYNONYM-type tokens from BM25 scoring
		k1, b float64
	}
	variants := []bm25Variant{
		{"expanded k1=1.20 b=0.75 (ship)", false, 1.2, 0.75},
		{"base     k1=1.20 b=0.75", true, 1.2, 0.75},
		{"expanded k1=0.90 b=0.40", false, 0.9, 0.4},
		{"base     k1=0.90 b=0.40", true, 0.9, 0.4},
		{"expanded k1=0.82 b=0.68", false, 0.82, 0.68},
		{"base     k1=0.82 b=0.68", true, 0.82, 0.68},
	}

	type qResult struct {
		vecRank  int // 1-based rank of gt in dense list; 0 = absent
		lexRank  int // 1-based rank of gt in BM25-ranked lexical list; 0 = absent
		fusedHit bool
		vecTop   []uint64 // top-500 dense
		lexTop   []uint64 // top-500 lexical
		varLex   []bool   // per-variant: gt in lexical top-10
		varFused []bool   // per-variant: gt in weighted-RRF(k=20,wVec=2) top-10
	}
	results := make([]qResult, len(queries))

	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU()-2)
	ctx := context.Background()
	for qi := range queries {
		wg.Add(1)
		sem <- struct{}{}
		go func(qi int) {
			defer wg.Done()
			defer func() { <-sem }()
			q := queries[qi]
			gtID := internalIDOf(qrels[q.id])

			eng.mu.RLock()
			defer eng.mu.RUnlock()

			var tokens []analysis.Token
			if qa, ok := eng.analyzer.(analysis.QueryAnalyzer); ok {
				tokens = qa.AnalyzeQuery(q.text)
			} else {
				tokens = eng.analyzer.Analyze(q.text)
			}
			raw := make([]string, 0, len(tokens))
			for _, tk := range tokens {
				raw = append(raw, tk.Term)
			}

			eng.inverted.RLock()
			eng.phonetics.RLock()
			kwScores := eng.lexicalPass(raw, eng.config.PhoneticWeight)
			eng.phonetics.RUnlock()
			eng.inverted.RUnlock()

			queryVec, _ := eng.embedder.Embed(ctx, q.text)
			eng.vectors.RLock()
			vScores := eng.vectorPass(queryVec, nil)
			eng.vectors.RUnlock()

			// Lexical ranking exactly as the hybrid branch of rankAndFuse.
			bm25Results := eng.bm25.Query(raw)
			bm25ByID := make(map[uint64]float64, len(bm25Results))
			for _, r := range bm25Results {
				bm25ByID[r.DocID] = r.Score
			}
			kwRank := make(map[uint64]float64, len(kwScores))
			for id, cov := range kwScores {
				if cov <= 0 {
					continue
				}
				if s, ok := bm25ByID[id]; ok {
					kwRank[id] = 1.0 + s
				} else {
					kwRank[id] = cov * 1e-9
				}
			}

			res := qResult{
				vecRank:  rankOf(vScores, gtID),
				lexRank:  rankOf(kwRank, gtID),
				vecTop:   topIDs(vScores, 500),
				lexTop:   topIDs(kwRank, 500),
				varLex:   make([]bool, len(variants)),
				varFused: make([]bool, len(variants)),
			}
			for _, r := range eng.rankAndFuse(kwScores, bm25Results, vScores, eng.scorer) {
				if r.ID == qrels[q.id] {
					res.fusedHit = true
					break
				}
			}

			// Variant evaluation: same lexical candidate set, different BM25
			// scoring tokens and parameters.
			baseToks := make([]string, 0, len(tokens))
			for _, tk := range tokens {
				if tk.Type != analysis.SYNONYM {
					baseToks = append(baseToks, tk.Term)
				}
			}
			candIDs := make([]uint64, 0, len(kwScores))
			for id := range kwScores {
				candIDs = append(candIDs, id)
			}
			for vi, v := range variants {
				terms := raw
				if v.base {
					terms = baseToks
				}
				scores := eng.bm25.ScoreDocs(candIDs, terms, v.k1, v.b)
				vRank := make(map[uint64]float64, len(kwScores))
				for id, cov := range kwScores {
					if cov <= 0 {
						continue
					}
					if s, ok := scores[id]; ok {
						vRank[id] = 1.0 + s
					} else {
						vRank[id] = cov * 1e-9
					}
				}
				if rk := rankOf(vRank, gtID); rk > 0 && rk <= 10 {
					res.varLex[vi] = true
				}
				res.varFused[vi] = simulateRRFHit(topIDs(vRank, 500), res.vecTop, gtID, 20.0, 1.0, 2.0)
			}
			results[qi] = res
		}(qi)
	}
	wg.Wait()

	// ── Aggregates ───────────────────────────────────────────────────────────
	n := float64(len(results))
	recallAt := func(rank func(qResult) int, k int) float64 {
		var hits int
		for _, r := range results {
			if rk := rank(r); rk > 0 && rk <= k {
				hits++
			}
		}
		return float64(hits) / n
	}
	vec := func(r qResult) int { return r.vecRank }
	lex := func(r qResult) int { return r.lexRank }

	var fused, union10, both10, neither50 int
	for _, r := range results {
		if r.fusedHit {
			fused++
		}
		v10 := r.vecRank > 0 && r.vecRank <= 10
		l10 := r.lexRank > 0 && r.lexRank <= 10
		if v10 || l10 {
			union10++
		}
		if v10 && l10 {
			both10++
		}
		v50 := r.vecRank > 0 && r.vecRank <= 50
		l50 := r.lexRank > 0 && r.lexRank <= 50
		if !v50 && !l50 {
			neither50++
		}
	}
	t.Logf("fused recall@10 (engine)     = %.4f", float64(fused)/n)
	t.Logf("dense  recall@10/20/50/100  = %.4f / %.4f / %.4f / %.4f",
		recallAt(vec, 10), recallAt(vec, 20), recallAt(vec, 50), recallAt(vec, 100))
	t.Logf("lexical recall@10/20/50/100 = %.4f / %.4f / %.4f / %.4f",
		recallAt(lex, 10), recallAt(lex, 20), recallAt(lex, 50), recallAt(lex, 100))
	t.Logf("union@10 = %.4f   both@10 = %.4f   gt outside both top-50 = %.4f",
		float64(union10)/n, float64(both10)/n, float64(neither50)/n)

	// ── Offline RRF fusion grid from recorded top-500 lists ──────────────────
	t.Log("fusion grid (recall@10): rows=k, cols=wVec (wLex=1), lists truncated at 500")
	for _, k := range []float64{10, 20, 30, 60, 120} {
		var line strings.Builder
		fmt.Fprintf(&line, "k=%5.0f:", k)
		for _, wv := range []float64{0.5, 1.0, 1.5, 2.0, 3.0} {
			var hits int
			for qi, r := range results {
				gtID := internalIDOf(qrels[queries[qi].id])
				if simulateRRFHit(r.lexTop, r.vecTop, gtID, k, 1.0, wv) {
					hits++
				}
			}
			fmt.Fprintf(&line, "  wv=%.1f→%.4f", wv, float64(hits)/n)
		}
		t.Log(line.String())
	}

	// ── BM25 variant table ────────────────────────────────────────────────────
	t.Log("BM25 variants (same candidates, scored over candidate set):")
	for vi, v := range variants {
		var lexHits, fusedHits int
		for _, r := range results {
			if r.varLex[vi] {
				lexHits++
			}
			if r.varFused[vi] {
				fusedHits++
			}
		}
		t.Logf("  %-32s lexical@10=%.4f  fused@10(k=20,wv=2)=%.4f",
			v.name, float64(lexHits)/n, float64(fusedHits)/n)
	}
}

// simulateRRFHit fuses two ranked ID lists with weighted RRF and reports
// whether gt lands in the fused top-10.
func simulateRRFHit(lexTop, vecTop []uint64, gt uint64, k, wLex, wVec float64) bool {
	scores := make(map[uint64]float64, len(lexTop)+len(vecTop))
	for i, id := range lexTop {
		scores[id] += wLex / (k + float64(i+1))
	}
	for i, id := range vecTop {
		scores[id] += wVec / (k + float64(i+1))
	}
	gtScore, ok := scores[gt]
	if !ok {
		return false
	}
	higher := 0
	for _, s := range scores {
		if s > gtScore {
			higher++
			if higher >= 10 {
				return false
			}
		}
	}
	return true
}

func rankOf(scores map[uint64]float64, gt uint64) int {
	gtScore, ok := scores[gt]
	if !ok {
		return 0
	}
	rank := 1
	for _, s := range scores {
		if s > gtScore {
			rank++
		}
	}
	return rank
}

func topIDs(scores map[uint64]float64, k int) []uint64 {
	ids := make([]uint64, 0, len(scores))
	for id := range scores {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(a, b int) bool {
		if scores[ids[a]] != scores[ids[b]] {
			return scores[ids[a]] > scores[ids[b]]
		}
		return ids[a] < ids[b]
	})
	if len(ids) > k {
		ids = ids[:k]
	}
	return ids
}
