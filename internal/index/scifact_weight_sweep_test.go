//go:build cgo

package index

import (
	"bufio"
	"context"
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

// TestSciFactWeightSweep runs the same VectorWeight/RRFConstant candidates as
// TestMSMARCOHybridWeightSweep against BEIR SciFact (out-of-domain, unlike
// MS MARCO which the current defaults were originally tuned against), to
// check a candidate isn't just overfit to one benchmark before it's adopted
// as a new default (see config.DefaultConfig's comment).
//
// SciFact's corpus is small (~5.2k docs) so each variant rebuilds the engine
// fresh, unlike the MS MARCO sweep which reuses one loaded ~107k-doc index.
//
// Run: ZENITH_SCIFACT_WEIGHTSWEEP=1 go test ./internal/index -run TestSciFactWeightSweep -v -timeout 1800s
// Needs bench/.cache/beir/scifact and the model under bench/.cache/models.
func TestSciFactWeightSweep(t *testing.T) {
	if os.Getenv("ZENITH_SCIFACT_WEIGHTSWEEP") == "" {
		t.Skip("set ZENITH_SCIFACT_WEIGHTSWEEP=1 to run")
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

	localEmb, err := localembedder.NewByID(os.Getenv("ZENITH_MODEL"), cache+"/models")
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}

	type variant struct {
		name                      string
		vectorWeight, rrfConstant float64
	}
	dv, dr := config.DefaultConfig().VectorWeight, config.DefaultConfig().RRFConstant
	variants := []variant{
		{"default", dv, dr},
		{"vectorWeight=4.0", 4.0, dr},
		{"rrfConstant=10", dv, 10.0},
		{"vectorWeight=4.0,rrfConstant=10", 4.0, 10.0},
	}

	ctx := context.Background()
	for _, v := range variants {
		emb, err := embedding.NewCachingEmbedder(localEmb, 10_000)
		if err != nil {
			t.Fatal(err)
		}
		cfg := config.DefaultConfig()
		cfg.WordVectors = false
		eng := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(v.rrfConstant, 0, 1.0, v.vectorWeight), analysis.NewStandardAnalyzer())
		eng.SetANNThreshold(0)
		if err := eng.AddBatch(ctx, batch); err != nil {
			t.Fatal(err)
		}

		var ndcg, recall float64
		for _, qid := range qids {
			res, err := eng.Search(ctx, qtext[qid])
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
		t.Logf("%-32s nDCG@10=%.4f  Recall@10=%.4f  (%d queries, %d docs)", v.name, ndcg/n, recall/n, len(qids), len(docs))
		eng.Close()
	}
}
