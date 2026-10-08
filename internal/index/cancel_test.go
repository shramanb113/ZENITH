package index

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// A canceled ctx on a large exact-scan corpus (no ANN graph — small enough
// to stay below the threshold, forcing vectorPass's unbounded eachVector
// scan) must return promptly with ctx.Err(), not finish the scan.
func TestSearchFilteredWeighted_CtxCancellationStopsExactScanPromptly(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = 0 // isolate the cancellation path from caching
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	e.SetANNThreshold(0) // force the exact-scan path regardless of corpus size
	defer e.Close()

	ctx := context.Background()
	const docs = 20_000
	for i := 0; i < docs; i++ {
		id := fmt.Sprintf("doc%d", i)
		if err := e.AddWithVectorAttrs(ctx, id, "hello world filler text", []float32{1, 0}, nil); err != nil {
			t.Fatal(err)
		}
	}

	canceled, cancel := context.WithCancel(context.Background())
	cancel() // already canceled before the search starts

	start := time.Now()
	_, err := e.SearchFiltered(canceled, "hello", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("SearchFiltered with an already-canceled ctx returned no error")
	}
	if err != context.Canceled {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("canceled search took %v; want a prompt return", elapsed)
	}
}
