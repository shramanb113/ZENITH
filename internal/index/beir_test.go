//go:build cgo

package index

import (
	"bufio"
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestBEIR evaluates ZENITH on BEIR datasets other than MS MARCO — out-of-domain
// retrieval, graded by nDCG@10 and Recall@10 on each dataset's test qrels — and
// ablates the signals: lexical variants with one signal removed, and the same
// with the dense vector added (hybrid), over one shared index per dataset.
//
//	ZENITH_BEIR=scifact,nfcorpus,fiqa [ZENITH_BEIR_MODEL=gte-small] \
//	  go test ./internal/index -run TestBEIR -v -timeout 6h
//
// Datasets are read from bench/.cache/beir/<name>/{corpus.jsonl,queries.jsonl,qrels/test.tsv}
// (the layout of the official BEIR zips). The first run embeds each corpus.
func TestBEIR(t *testing.T) {
	sets := os.Getenv("ZENITH_BEIR")
	if sets == "" {
		t.Skip("set ZENITH_BEIR=<comma-separated dataset names> to run")
	}
	const cache = "../../bench/.cache"
	modelID := os.Getenv("ZENITH_BEIR_MODEL")
	for _, name := range strings.Split(sets, ",") {
		name = strings.TrimSpace(name)
		t.Run(name, func(t *testing.T) { runBEIR(t, cache, name, modelID) })
	}
}

type beirVariant struct {
	name string
	set  func(c *config.Config)
}

var beirVariants = []beirVariant{
	{"full", func(c *config.Config) {}},
	{"no fuzzy", func(c *config.Config) { c.FuzzyMaxDist = 0 }},
	{"no phonetic", func(c *config.Config) { c.PhoneticWeight = 0 }},
	{"no prefix expansion", func(c *config.Config) { c.PrefixFragmentCap = 1 }},
	{"exact terms + BM25 only", func(c *config.Config) { c.FuzzyMaxDist = 0; c.PhoneticWeight = 0; c.PrefixFragmentCap = 1 }},
}

func runBEIR(t *testing.T, cache, name, modelID string) {
	dir := filepath.Join(cache, "beir", name)
	var docIDs, docs []string
	jsonl(t, filepath.Join(dir, "corpus.jsonl"), func(m map[string]any) {
		title, _ := m["title"].(string)
		text, _ := m["text"].(string)
		docIDs = append(docIDs, m["_id"].(string))
		docs = append(docs, strings.TrimSpace(title+". "+text))
	})
	qtext := map[string]string{}
	jsonl(t, filepath.Join(dir, "queries.jsonl"), func(m map[string]any) { qtext[m["_id"].(string)] = m["text"].(string) })
	rel := map[string]map[string]bool{}
	var qids []string
	f, err := os.Open(filepath.Join(dir, "qrels", "test.tsv"))
	if err != nil {
		t.Skipf("%s qrels missing: %v", name, err)
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		p := strings.Split(sc.Text(), "\t")
		if len(p) < 3 || p[0] == "query-id" || p[2] == "0" {
			continue
		}
		if _, ok := qtext[p[0]]; !ok {
			continue
		}
		if rel[p[0]] == nil {
			rel[p[0]] = map[string]bool{}
			qids = append(qids, p[0])
		}
		rel[p[0]][p[1]] = true
	}
	f.Close()

	batch := make([]BatchDoc, len(docs))
	for i := range docs {
		batch[i] = BatchDoc{ID: docIDs[i], Text: docs[i]}
	}
	t.Logf("%s: %d docs, %d test queries", name, len(docs), len(qids))

	eval := func(eng *Engine, transform func(q string) string) (ndcg, recall float64, p50 time.Duration) {
		var lat []time.Duration
		for _, qid := range qids {
			q := qtext[qid]
			if transform != nil {
				q = transform(q)
			}
			t0 := time.Now()
			res, err := eng.Search(context.Background(), q)
			lat = append(lat, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			dcg, hit := 0.0, 0
			for i, r := range res {
				if i >= 10 {
					break
				}
				if rel[qid][r.ID] {
					dcg += 1 / math.Log2(float64(i+2))
					hit++
				}
			}
			ideal := 0.0
			for i := 0; i < min(10, len(rel[qid])); i++ {
				ideal += 1 / math.Log2(float64(i+2))
			}
			ndcg += dcg / ideal
			recall += float64(hit) / float64(len(rel[qid]))
		}
		n := float64(len(qids))
		sortDurations(lat)
		return ndcg / n, recall / n, lat[len(lat)/2]
	}
	// Queries with one typo in their longest word of 6+ letters (as in the MS MARCO runs).
	typo := func(q string) string {
		if tq, ok := withTypo(q, q); ok {
			return tq
		}
		return q
	}

	build := func(emb embedding.Embedder) (*Engine, *config.Config) {
		cfg := config.DefaultConfig()
		cfg.WordVectors = false
		e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
		e.SetANNThreshold(0)
		start := time.Now()
		if err := e.AddBatch(context.Background(), batch); err != nil {
			t.Fatal(err)
		}
		t.Logf("  indexed in %s", time.Since(start).Round(time.Second))
		return e, cfg
	}
	matrix := func(label string, eng *Engine, cfg *config.Config) {
		base := *cfg
		for _, v := range beirVariants {
			*cfg = base
			v.set(cfg)
			n, r, p50 := eval(eng, nil)
			tn, tr, _ := eval(eng, typo)
			line := fmt.Sprintf("%-8s %-26s nDCG@10 %.4f  Recall@10 %.4f  p50 %-8s | with a typo: nDCG@10 %.4f  Recall@10 %.4f",
				label, v.name, n, r, p50.Round(10*time.Microsecond), tn, tr)
			t.Log(line)
			fmt.Fprintf(os.Stderr, "BEIR %s | %s\n", name, line)
		}
		*cfg = base
	}

	lex, lcfg := build(noVecEmbedder{})
	matrix("lexical", lex, lcfg)
	lex.Close()

	emb, err := localembedder.NewByID(modelID, cache+"/models")
	if err != nil {
		t.Logf("dense signal skipped: %v", err)
		return
	}
	cached, _ := embedding.NewCachingEmbedder(emb, 100_000)
	hyb, hcfg := build(cached)
	matrix("hybrid", hyb, hcfg)
	hyb.Close()
}

func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		for j := i; j > 0 && d[j] < d[j-1]; j-- {
			d[j], d[j-1] = d[j-1], d[j]
		}
	}
}
