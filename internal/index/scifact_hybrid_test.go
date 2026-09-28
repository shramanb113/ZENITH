//go:build cgo

package index

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestSciFactHybrid measures the FULL hybrid pipeline (lexical + dense + RRF) on
// BEIR SciFact — an out-of-domain scientific-claim task — for each embedding
// model, next to a lexical-only baseline. TestModelComparison shows how models
// differ dense-only; this shows whether that difference survives fusion with
// BM25, which is what the default-model decision actually turns on.
//
// Run: ZENITH_SCIFACT_HYBRID=1 go test ./internal/index -run TestSciFactHybrid -v -timeout 3600s
// Needs bench/.cache/beir/scifact and the models under bench/.cache/models.
func TestSciFactHybrid(t *testing.T) {
	if os.Getenv("ZENITH_SCIFACT_HYBRID") == "" {
		t.Skip("set ZENITH_SCIFACT_HYBRID=1 to run")
	}
	const cache = "../../bench/.cache"
	dir := filepath.Join(cache, "beir", "scifact")

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
		t.Skipf("SciFact missing: %v", err)
	}
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		p := strings.Split(sc.Text(), "\t")
		if len(p) < 3 || p[0] == "query-id" || p[2] == "0" {
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

	eval := func(name string, eng *Engine) {
		var ndcg, recall float64
		for _, qid := range qids {
			res, err := eng.Search(context.Background(), qtext[qid])
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
		t.Logf("%-34s nDCG@10=%.4f  Recall@10=%.4f  (%d queries, %d docs)", name, ndcg/n, recall/n, len(qids), len(docs))
	}
	build := func(emb embedding.Embedder) *Engine {
		cfg := config.DefaultConfig()
		cfg.WordVectors = false
		e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
		e.SetANNThreshold(0)
		if err := e.AddBatch(context.Background(), batch); err != nil {
			t.Fatal(err)
		}
		return e
	}

	lex := build(noVecEmbedder{})
	eval("lexical only (BM25/n-gram/fuzzy)", lex)
	lex.Close()

	for _, spec := range localembedder.Models() {
		emb, err := localembedder.NewByID(spec.ID, cache+"/models")
		if err != nil {
			t.Logf("%-34s SKIPPED: %v", "hybrid / "+spec.ID, err)
			continue
		}
		cached, _ := embedding.NewCachingEmbedder(emb, 10_000)
		e := build(cached)
		eval("hybrid / "+spec.ID, e)
		e.Close()
	}
}

func jsonl(t *testing.T, path string, fn func(map[string]any)) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("dataset missing: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<22), 1<<22)
	for sc.Scan() {
		var m map[string]any
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatal(err)
		}
		fn(m)
	}
}
