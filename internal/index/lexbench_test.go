package index

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

type noVecEmbedder struct{}

func (noVecEmbedder) Embed(context.Context, string) ([]float32, error) { return nil, nil }
func (noVecEmbedder) EmbedBatch(_ context.Context, t []string) ([][]float32, error) {
	return make([][]float32, len(t)), nil
}
func (noVecEmbedder) Dimensions() int { return 0 }

// synthWords builds a deterministic Zipf-ish vocabulary of pronounceable words.
func synthWords(n int, r *rand.Rand) []string {
	cons := "bcdfghjklmnprstvwz"
	vow := "aeiou"
	seen := map[string]bool{}
	var out []string
	for len(out) < n {
		l := 3 + r.Intn(7)
		b := make([]byte, 0, l*2)
		for i := 0; i < l; i++ {
			b = append(b, cons[r.Intn(len(cons))], vow[r.Intn(len(vow))])
		}
		w := string(b)
		if !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	return out
}

func buildLexEngine(tb testing.TB, docs, vocab int) (*Engine, []string) {
	r := rand.New(rand.NewSource(1))
	words := synthWords(vocab, r)
	zipf := rand.NewZipf(r, 1.2, 1, uint64(vocab-1))
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	batch := make([]BatchDoc, docs)
	for i := range batch {
		txt := ""
		for j := 0; j < 40; j++ {
			txt += words[zipf.Uint64()] + " "
		}
		batch[i] = BatchDoc{ID: fmt.Sprintf("d%d", i), Text: txt}
	}
	if err := eng.AddBatch(context.Background(), batch); err != nil {
		tb.Fatal(err)
	}
	return eng, words
}

func BenchmarkLexicalSearch(b *testing.B) {
	eng, words := buildLexEngine(b, lexBenchDocs(), 30000)
	r := rand.New(rand.NewSource(2))
	queries := make([]string, 200)
	for i := range queries {
		q := words[r.Intn(len(words)/3)] + " " + words[r.Intn(len(words)/3)]
		if i%2 == 1 { // typo: drop a letter from the first word
			w := words[r.Intn(len(words)/3)]
			q = w[:len(w)-1] + " " + words[r.Intn(len(words)/3)]
		}
		queries[i] = q
	}
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := eng.Search(ctx, queries[i%len(queries)]); err != nil {
			b.Fatal(err)
		}
	}
}

// lexBenchDocs returns the corpus size, overridable with ZENITH_BENCH_DOCS.
func lexBenchDocs() int {
	if v, err := strconv.Atoi(os.Getenv("ZENITH_BENCH_DOCS")); err == nil && v > 0 {
		return v
	}
	return 20000
}
