//go:build cgo

package localembedder

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

// multilingualModelsDir returns ~/.zenith/models, the real default
// `zenith models pull` writes to (not the bench/.cache/models convention used
// by the MS MARCO/SciFact eval harness, which expects a hand-populated cache).
func multilingualModelsDir(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skipf("cannot determine home dir: %v", err)
	}
	return filepath.Join(home, ".zenith", "models")
}

// TestMultilingualSmoke is a hand-built sanity check for labse (LaBSE), not a
// benchmark: a handful of non-English queries must embed closer to their
// correct-language/topic passage than to unrelated English distractors. This
// reconstructs the real sentence-transformers pipeline (CLS pool -> trained
// dense+tanh projection -> normalize, see modelspec's DenseURL), so it checks
// more than "basic cross-lingual signal survives" — but it is still not
// retrieval quality at MS MARCO/BEIR rigor or a guarantee of LaBSE's
// published numbers.
//
// Run: ZENITH_MODEL_EVAL=1 go test ./internal/localembedder -run TestMultilingualSmoke -v
// Needs: zenith models pull labse
func TestMultilingualSmoke(t *testing.T) {
	if os.Getenv("ZENITH_MODEL_EVAL") == "" {
		t.Skip("set ZENITH_MODEL_EVAL=1 to run (needs the multilingual model pulled)")
	}
	emb, err := NewByID("labse", multilingualModelsDir(t))
	if err != nil {
		t.Skipf("multilingual model not installed: %v (run: zenith models pull labse)", err)
	}

	distractors := []string{
		"The stock market closed higher today on strong earnings reports.",
		"Scientists discovered a new species of beetle in the rainforest.",
		"The city council approved funding for a new public park.",
	}

	cases := []struct {
		name    string
		query   string
		correct string
	}{
		{"french", "Quelle est la capitale de la France ?", "Paris est la capitale et la plus grande ville de France."},
		{"german", "Wie funktioniert ein Elektroauto?", "Ein Elektroauto wird von einem Elektromotor angetrieben, der Energie aus einer Batterie bezieht."},
		{"spanish", "¿Cómo se prepara una tortilla española?", "La tortilla española se hace con huevos, patatas y cebolla fritos en aceite de oliva."},
		{"hindi", "भारत की राजधानी क्या है?", "नई दिल्ली भारत की राजधानी है।"},
		{"japanese", "富士山の高さはどれくらいですか？", "富士山は日本一高い山で、標高は3776メートルです。"},
	}

	ctx := context.Background()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			qVec, err := emb.EmbedQuery(ctx, c.query)
			if err != nil {
				t.Fatalf("embed query: %v", err)
			}
			correctVec, err := emb.Embed(ctx, c.correct)
			if err != nil {
				t.Fatalf("embed correct passage: %v", err)
			}
			correctSim := dot(qVec, correctVec)

			for _, d := range distractors {
				dVec, err := emb.Embed(ctx, d)
				if err != nil {
					t.Fatalf("embed distractor: %v", err)
				}
				if sim := dot(qVec, dVec); sim >= correctSim {
					t.Errorf("distractor %q scored %.4f >= correct passage %.4f for query %q",
						d, sim, correctSim, c.query)
				}
			}
			t.Logf("%s: correct passage similarity %.4f (highest distractor must be lower)", c.name, correctSim)
		})
	}
}

func dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
