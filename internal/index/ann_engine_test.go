package index

import (
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

const bagDim = 64

// bagEmbedder maps text to a normalised bag-of-hashed-words vector, so
// documents sharing words are genuinely close — enough structure to test
// ANN against exact search without a real model.
type bagEmbedder struct{}

func (bagEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	v := make([]float32, bagDim)
	for _, w := range strings.Fields(strings.ToLower(text)) {
		h := fnv.New32a()
		h.Write([]byte(w))
		s := h.Sum32()
		v[s%bagDim] += 1
		v[(s>>8)%bagDim] += 0.5
	}
	var ss float64
	for _, x := range v {
		ss += float64(x) * float64(x)
	}
	if ss == 0 {
		return nil, nil
	}
	n := float32(math.Sqrt(ss))
	for i := range v {
		v[i] /= n
	}
	return v, nil
}
func (b bagEmbedder) EmbedBatch(ctx context.Context, t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i := range t {
		out[i], _ = b.Embed(ctx, t[i])
	}
	return out, nil
}
func (bagEmbedder) Dimensions() int { return bagDim }

func newBagEngine(annMin int) *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, bagEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetANNThreshold(annMin)
	return e
}

func bagDocs(n int, seed int64) []BatchDoc {
	r := rand.New(rand.NewSource(seed))
	vocab := make([]string, 400)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%03dx", i)
	}
	docs := make([]BatchDoc, n)
	for i := range docs {
		topic := r.Intn(20)
		var sb strings.Builder
		for j := 0; j < 25; j++ {
			if r.Intn(3) > 0 {
				sb.WriteString(vocab[topic*20+r.Intn(20)])
			} else {
				sb.WriteString(vocab[r.Intn(len(vocab))])
			}
			sb.WriteByte(' ')
		}
		docs[i] = BatchDoc{ID: fmt.Sprintf("d%05d", i), Text: sb.String()}
	}
	return docs
}

func topResultIDs(rs []SearchResponse, k int) map[string]bool {
	m := map[string]bool{}
	for i, r := range rs {
		if i >= k {
			break
		}
		m[r.ID] = true
	}
	return m
}

func TestANN_MatchesExactSearchClosely(t *testing.T) {
	docs := bagDocs(1500, 1)
	exact, fast := newBagEngine(0), newBagEngine(100)
	ctx := context.Background()
	if err := exact.AddBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if err := fast.AddBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if !fast.ANNActive() || exact.ANNActive() {
		t.Fatalf("ANN active: fast=%v exact=%v", fast.ANNActive(), exact.ANNActive())
	}

	r := rand.New(rand.NewSource(4))
	var overlap float64
	const queries = 60
	for i := 0; i < queries; i++ {
		q := docs[r.Intn(len(docs))].Text
		a, _ := exact.Search(ctx, q)
		b, _ := fast.Search(ctx, q)
		ea, eb := topResultIDs(a, 10), topResultIDs(b, 10)
		hit := 0
		for id := range ea {
			if eb[id] {
				hit++
			}
		}
		overlap += float64(hit) / float64(len(ea))
	}
	avg := overlap / queries
	t.Logf("top-10 overlap ANN vs exact hybrid search: %.3f", avg)
	if avg < 0.9 {
		t.Fatalf("top-10 overlap %.3f, want >= 0.9", avg)
	}
}

func TestANN_RespectsFilterDeleteAndReplace(t *testing.T) {
	e := newBagEngine(50)
	ctx := context.Background()
	docs := bagDocs(800, 2)
	for i := range docs {
		grp := "even"
		if i%2 == 1 {
			grp = "odd"
		}
		docs[i].Attrs = Attrs{"g": {Kind: AttrString, S: grp}}
	}
	if err := e.AddBatch(ctx, docs); err != nil {
		t.Fatal(err)
	}
	if !e.ANNActive() {
		t.Fatal("ANN should be active")
	}

	odd := func(a Attrs) bool { return a["g"].S == "odd" }
	rs, err := e.SearchWithFilter(ctx, docs[10].Text, odd)
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) == 0 {
		t.Fatal("expected filtered results")
	}
	for _, r := range rs {
		var n int
		fmt.Sscanf(r.ID, "d%d", &n)
		if n%2 != 1 {
			t.Fatalf("filter leaked %s", r.ID)
		}
	}

	// Delete: the exact-duplicate doc must vanish from results.
	if err := e.Remove(ctx, docs[10].ID); err != nil {
		t.Fatal(err)
	}
	rs, _ = e.Search(ctx, docs[10].Text)
	for _, r := range rs {
		if r.ID == docs[10].ID {
			t.Fatal("deleted document still returned via ANN")
		}
	}

	// Replace: re-adding an id with different text must move it.
	if err := e.Add(ctx, docs[20].ID, "zzzuniqueone zzzuniquetwo zzzuniquethree"); err != nil {
		t.Fatal(err)
	}
	rs, _ = e.Search(ctx, "zzzuniqueone zzzuniquetwo")
	if len(rs) == 0 || rs[0].ID != docs[20].ID {
		t.Fatalf("replaced doc not top hit: %+v", rs)
	}
}

func TestANN_RebuiltAfterLoadAndBelowThresholdStaysExact(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "ann.db")
	a := newBagEngine(100)
	defer a.Close()
	if err := a.AddBatch(ctx, bagDocs(400, 3)); err != nil {
		t.Fatal(err)
	}
	if err := a.Save(path); err != nil {
		t.Fatal(err)
	}

	b := newBagEngine(100)
	defer b.Close()
	if b.ANNActive() {
		t.Fatal("fresh engine must not have ANN")
	}
	if err := b.Load(path); err != nil {
		t.Fatal(err)
	}
	if !b.ANNActive() {
		t.Fatal("Load should rebuild the ANN graph for a corpus over the threshold")
	}

	small := newBagEngine(10_000)
	if err := small.AddBatch(ctx, bagDocs(200, 4)); err != nil {
		t.Fatal(err)
	}
	if small.ANNActive() {
		t.Fatal("corpus below threshold must stay on the exact path")
	}
}

func BenchmarkHybridVectorPass(b *testing.B) {
	docs := bagDocs(30000, 5)
	ctx := context.Background()
	for _, tc := range []struct {
		name string
		min  int
	}{{"exact", 0}, {"ann", 20000}} {
		e := newBagEngine(tc.min)
		if err := e.AddBatch(ctx, docs); err != nil {
			b.Fatal(err)
		}
		q, _ := bagEmbedder{}.Embed(ctx, docs[7].Text)
		b.Run(tc.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				e.mu.RLock()
				e.vectors.RLock()
				_ = e.vectorPass(q, nil)
				e.vectors.RUnlock()
				e.mu.RUnlock()
			}
		})
	}
}
