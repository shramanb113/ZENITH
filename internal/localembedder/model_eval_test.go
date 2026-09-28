//go:build cgo

package localembedder

import (
	"bufio"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"
)

// TestModelComparison scores every registered model on the same retrieval
// tasks with dense-only search (no lexical signal, no fusion), so the numbers
// isolate the embedding model:
//
//   - msmarco: 2,000 dev queries; corpus = their ground-truth passages plus
//     18,000 distractors from the first 100k passages (in-domain for most
//     sentence-embedding models, so it flatters them — read it as an upper bound).
//   - scifact: BEIR SciFact test split (300 queries, 5,183 abstracts), an
//     out-of-domain scientific-claim task.
//
// Run: ZENITH_MODEL_EVAL=1 go test ./internal/localembedder -run TestModelComparison -v -timeout 3600s
// Needs bench/.cache (MS MARCO via `go run ./cmd/fetch`, models via `zenith models pull`,
// SciFact unzipped to bench/.cache/beir/scifact).
func TestModelComparison(t *testing.T) {
	if os.Getenv("ZENITH_MODEL_EVAL") == "" {
		t.Skip("set ZENITH_MODEL_EVAL=1 to run (downloads/uses bench/.cache, several minutes)")
	}
	cache := "../../bench/.cache"
	modelsDir := filepath.Join(cache, "models")

	tasks := []evalTask{loadMSMARCOTask(t, cache, 2000, 18000), loadSciFactTask(t, filepath.Join(cache, "beir", "scifact"))}

	for _, spec := range Models() {
		emb, err := NewByID(spec.ID, modelsDir)
		if err != nil {
			t.Logf("%-20s SKIPPED: %v", spec.ID, err)
			continue
		}
		variants := []struct {
			label  string
			prefix string
		}{{"", spec.QueryPrefix}}
		if spec.QueryPrefix != "" {
			variants = append(variants, struct{ label, prefix string }{" (no query prefix)", ""})
		}
		for _, task := range tasks {
			start := time.Now()
			docVecs := embedAll(t, emb, task.docs)
			docTime := time.Since(start)
			for _, v := range variants {
				saved := emb.spec.QueryPrefix
				emb.spec.QueryPrefix = v.prefix
				ndcg, recall, mrr := task.score(t, emb, docVecs)
				emb.spec.QueryPrefix = saved
				t.Logf("%-20s%-22s %-8s nDCG@10=%.4f  Recall@10=%.4f  MRR@10=%.4f  (embed %d docs in %s)",
					spec.ID, v.label, task.name, ndcg, recall, mrr, len(task.docs), docTime.Round(time.Second))
			}
		}
		emb.model.close()
	}
}

type evalTask struct {
	name    string
	docIDs  []string
	docs    []string
	queries []string
	qids    []string
	rel     map[string]map[string]bool // qid -> set of relevant doc IDs
}

func embedAll(t *testing.T, e *Embedder, texts []string) [][]float32 {
	t.Helper()
	// Length-sorted batches of 64: same regime the engine uses for indexing.
	order := make([]int, len(texts))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return len(texts[order[a]]) < len(texts[order[b]]) })
	out := make([][]float32, len(texts))
	const bs = 64
	for i := 0; i < len(order); i += bs {
		end := min(i+bs, len(order))
		batch := make([]string, 0, end-i)
		for _, idx := range order[i:end] {
			batch = append(batch, texts[idx])
		}
		vecs, err := e.EmbedBatch(context.Background(), batch)
		if err != nil {
			t.Fatal(err)
		}
		for j, idx := range order[i:end] {
			out[idx] = vecs[j]
		}
	}
	return out
}

func (task evalTask) score(t *testing.T, e *Embedder, docVecs [][]float32) (ndcg, recall, mrr float64) {
	t.Helper()
	for qi, q := range task.queries {
		qv, err := e.EmbedQuery(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}
		type sd struct {
			s float32
			i int
		}
		scored := make([]sd, len(docVecs))
		for i, dv := range docVecs {
			var dot float32
			for k := range dv {
				dot += qv[k] * dv[k]
			}
			scored[i] = sd{dot, i}
		}
		sort.Slice(scored, func(a, b int) bool { return scored[a].s > scored[b].s })
		rel := task.rel[task.qids[qi]]
		var dcg, firstRR float64
		found := 0
		for r := 0; r < 10 && r < len(scored); r++ {
			if rel[task.docIDs[scored[r].i]] {
				dcg += 1 / math.Log2(float64(r)+2)
				found++
				if firstRR == 0 {
					firstRR = 1 / float64(r+1)
				}
			}
		}
		var idcg float64
		for r := 0; r < min(len(rel), 10); r++ {
			idcg += 1 / math.Log2(float64(r)+2)
		}
		if idcg > 0 {
			ndcg += dcg / idcg
		}
		if found > 0 {
			recall++
		}
		mrr += firstRR
	}
	n := float64(len(task.queries))
	return ndcg / n, recall / n, mrr / n
}

func loadMSMARCOTask(t *testing.T, cache string, nQueries, nDistract int) evalTask {
	t.Helper()
	qrels := map[string]string{}
	var qorder []string
	readTSV(t, filepath.Join(cache, "qrels.dev.small.tsv"), func(p []string) bool {
		if len(p) >= 3 {
			if _, seen := qrels[p[0]]; !seen {
				qorder = append(qorder, p[0])
			}
			qrels[p[0]] = p[2]
		}
		return true
	})
	if len(qorder) > nQueries {
		qorder = qorder[:nQueries]
	}
	gt := map[string]bool{}
	for _, q := range qorder {
		gt[qrels[q]] = true
	}
	texts := map[string]string{}
	distract := 0
	readTSV(t, filepath.Join(cache, "collection.tsv"), func(p []string) bool {
		if len(p) < 2 {
			return true
		}
		if gt[p[0]] {
			texts[p[0]] = p[1]
		} else if distract < nDistract {
			texts[p[0]] = p[1]
			distract++
		}
		return distract < nDistract || len(texts) < len(gt)+nDistract
	})
	qtext := map[string]string{}
	readTSV(t, filepath.Join(cache, "queries.dev.small.tsv"), func(p []string) bool {
		if len(p) >= 2 {
			qtext[p[0]] = p[1]
		}
		return true
	})
	task := evalTask{name: "msmarco", rel: map[string]map[string]bool{}}
	ids := make([]string, 0, len(texts))
	for id := range texts {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		task.docIDs = append(task.docIDs, id)
		task.docs = append(task.docs, texts[id])
	}
	for _, q := range qorder {
		if _, ok := texts[qrels[q]]; !ok {
			continue
		}
		task.qids = append(task.qids, q)
		task.queries = append(task.queries, qtext[q])
		task.rel[q] = map[string]bool{qrels[q]: true}
	}
	return task
}

func loadSciFactTask(t *testing.T, dir string) evalTask {
	t.Helper()
	task := evalTask{name: "scifact", rel: map[string]map[string]bool{}}
	readJSONL(t, filepath.Join(dir, "corpus.jsonl"), func(m map[string]any) {
		title, _ := m["title"].(string)
		text, _ := m["text"].(string)
		task.docIDs = append(task.docIDs, m["_id"].(string))
		task.docs = append(task.docs, strings.TrimSpace(title+". "+text))
	})
	qtext := map[string]string{}
	readJSONL(t, filepath.Join(dir, "queries.jsonl"), func(m map[string]any) {
		qtext[m["_id"].(string)] = m["text"].(string)
	})
	readTSV(t, filepath.Join(dir, "qrels", "test.tsv"), func(p []string) bool {
		if len(p) < 3 || p[0] == "query-id" || p[2] == "0" {
			return true
		}
		if task.rel[p[0]] == nil {
			task.rel[p[0]] = map[string]bool{}
			task.qids = append(task.qids, p[0])
			task.queries = append(task.queries, qtext[p[0]])
		}
		task.rel[p[0]][p[1]] = true
		return true
	})
	return task
}

func readTSV(t *testing.T, path string, fn func([]string) bool) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("dataset missing: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if !fn(strings.Split(sc.Text(), "\t")) {
			return
		}
	}
}

func readJSONL(t *testing.T, path string, fn func(map[string]any)) {
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
