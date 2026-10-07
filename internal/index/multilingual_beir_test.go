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

// TestMultilingualBEIR evaluates registered multilingual embedding models
// (labse, distiluse-multilingual) with a real retrieval metric — nDCG@10 and
// Recall@10, the same harness TestBEIR uses — instead of the dot-product-only
// sanity check in internal/localembedder/multilingual_test.go.
//
// The corpus is hand-curated and deliberately small (16 factual passages, 6
// questions, one relevant passage per question, per language) under
// testdata/multilingual/<lang>/, in the same {corpus.jsonl,queries.jsonl,
// qrels/test.tsv} layout TestBEIR reads. This is smoke-scale, not BEIR-scale:
// it is real nDCG/Recall instead of a similarity-ordering check, but the
// corpus is far too small (and not independently reviewed by native
// speakers) to claim benchmark rigor or to compare against published
// MIRACL/Mr.TyDi numbers.
//
//	ZENITH_MODEL_EVAL=1 go test ./internal/index -run TestMultilingualBEIR -v
//
// Needs: zenith models pull labse (and/or) zenith models pull distiluse-multilingual
func TestMultilingualBEIR(t *testing.T) {
	if os.Getenv("ZENITH_MODEL_EVAL") == "" {
		t.Skip("set ZENITH_MODEL_EVAL=1 to run (needs a multilingual model pulled)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	modelsDir := filepath.Join(home, ".zenith", "models")

	langs := []string{"fr", "de", "es", "hi", "ja", "zh", "ar"}
	models := []string{"labse", "distiluse-multilingual"}

	for _, modelID := range models {
		emb, err := localembedder.NewByID(modelID, modelsDir)
		if err != nil {
			t.Logf("%s: skipped, not installed (%v) — run: zenith models pull %s", modelID, err, modelID)
			continue
		}
		cached, cerr := embedding.NewCachingEmbedder(emb, 1000)
		if cerr != nil {
			t.Fatalf("%s: caching embedder: %v", modelID, cerr)
		}
		for _, lang := range langs {
			lang := lang
			t.Run(modelID+"/"+lang, func(t *testing.T) {
				runMultilingualBEIR(t, lang, modelID, cached)
			})
		}
	}
}

func runMultilingualBEIR(t *testing.T, lang, modelID string, emb embedding.Embedder) {
	dir := filepath.Join("testdata", "multilingual", lang)

	var docIDs, docs []string
	jsonl(t, filepath.Join(dir, "corpus.jsonl"), func(m map[string]any) {
		docIDs = append(docIDs, m["_id"].(string))
		docs = append(docs, m["text"].(string))
	})
	qtext := map[string]string{}
	jsonl(t, filepath.Join(dir, "queries.jsonl"), func(m map[string]any) { qtext[m["_id"].(string)] = m["text"].(string) })

	rel := map[string]map[string]bool{}
	var qids []string
	f, err := os.Open(filepath.Join(dir, "qrels", "test.tsv"))
	if err != nil {
		t.Fatalf("qrels missing: %v", err)
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

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	eng.SetANNThreshold(0)
	defer eng.Close()
	if err := eng.AddBatch(context.Background(), batch); err != nil {
		t.Fatal(err)
	}

	var ndcg, recall float64
	var lat []time.Duration
	for _, qid := range qids {
		t0 := time.Now()
		res, err := eng.Search(context.Background(), qtext[qid])
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
	line := fmt.Sprintf("%-6s %-24s nDCG@10 %.4f  Recall@10 %.4f  p50 %s  (n=%d queries, %d docs)",
		lang, modelID, ndcg/n, recall/n, lat[len(lat)/2].Round(10*time.Microsecond), len(qids), len(docs))
	t.Log(line)
	fmt.Fprintf(os.Stderr, "MULTILINGUAL %s\n", line)
}
