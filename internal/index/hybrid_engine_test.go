//go:build cgo

package index

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestHybridEngineLatency times the hybrid engine alone (ANN active, query
// embeddings cached, so it is the engine's own cost) over the cached MS MARCO
// hybrid index. It is the quick loop for hybrid-path work — TestMSMARCOHybrid
// takes ~17 minutes because it also runs the exact scan — and is the target for
// `go test -cpuprofile`.
//
//	ZENITH_HYBRID_ENGINE=1 ZENITH_MSMARCO_HYBRID_INDEX=../../bench/.cache/hybrid-gte-small.db \
//	  go test ./internal/index -run TestHybridEngineLatency -v [-cpuprofile cpu.out]
//
// Optional: ZENITH_MSMARCO_QUERIES=<n> (default 1000).
func TestHybridEngineLatency(t *testing.T) {
	if os.Getenv("ZENITH_HYBRID_ENGINE") == "" {
		t.Skip("set ZENITH_HYBRID_ENGINE=1 and ZENITH_MSMARCO_HYBRID_INDEX=<cached hybrid index> to run")
	}
	const cacheDir = "../../bench/.cache"
	idxPath := os.Getenv("ZENITH_MSMARCO_HYBRID_INDEX")
	if _, err := os.Stat(idxPath); idxPath == "" || err != nil {
		t.Skip("cached hybrid index missing (build it with TestMSMARCOHybrid)")
	}
	nQueries := 1000
	if n, _ := strconv.Atoi(os.Getenv("ZENITH_MSMARCO_QUERIES")); n > 0 {
		nQueries = n
	}
	_, _, queries, qrels := loadMSMARCO(t, cacheDir, 100_000)
	if nQueries < len(queries) {
		queries = queries[:nQueries]
	}
	localEmb, err := localembedder.NewByID(os.Getenv("ZENITH_MODEL"), cacheDir+"/models")
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}
	cache, err := embedding.NewCachingEmbedder(localEmb, 10_000)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, cache, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	eng.SetANNThreshold(0)
	defer eng.Close()
	if err := eng.Load(idxPath); err != nil {
		t.Fatal(err)
	}
	eng.SetANNThreshold(defaultANNMinDocs)
	if !eng.ANNActive() {
		t.Fatal("ANN did not activate")
	}
	ctx := context.Background()
	for _, q := range queries { // fill the embedding cache
		if _, err := cache.Embed(ctx, q.text); err != nil {
			t.Fatal(err)
		}
	}

	// ZENITH_HYBRID_SWEEP="150:200,400:500,..." measures each annK:annEF once
	// (the default is restored afterwards) instead of the three default passes.
	if sweep := os.Getenv("ZENITH_HYBRID_SWEEP"); sweep != "" {
		defK, defEF := annK, annEF
		defer func() { annK, annEF = defK, defEF }()
		for _, kv := range strings.Split(sweep, ",") {
			var k, ef int
			if _, err := fmt.Sscanf(kv, "%d:%d", &k, &ef); err != nil {
				t.Fatalf("bad ZENITH_HYBRID_SWEEP entry %q", kv)
			}
			annK, annEF = k, ef
			var lat []time.Duration
			hits := 0
			for _, q := range queries {
				t0 := time.Now()
				res, err := eng.Search(ctx, q.text)
				lat = append(lat, time.Since(t0))
				if err != nil {
					t.Fatal(err)
				}
				for i, x := range res {
					if i >= 10 {
						break
					}
					if x.ID == qrels[q.id] {
						hits++
						break
					}
				}
			}
			sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
			p := func(f float64) time.Duration { return lat[int(float64(len(lat)-1)*f)].Round(10 * time.Microsecond) }
			t.Logf("annK=%d annEF=%d: Recall@10=%.4f  p50=%s p95=%s p99=%s", k, ef, float64(hits)/float64(len(queries)), p(0.5), p(0.95), p(0.99))
		}
		return
	}

	for pass := 1; pass <= 3; pass++ {
		var lat []time.Duration
		hits := 0
		for _, q := range queries {
			t0 := time.Now()
			res, err := eng.Search(ctx, q.text)
			lat = append(lat, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			for i, x := range res {
				if i >= 10 {
					break
				}
				if x.ID == qrels[q.id] {
					hits++
					break
				}
			}
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		p := func(f float64) time.Duration { return lat[int(float64(len(lat)-1)*f)].Round(10 * time.Microsecond) }
		t.Logf("pass %d: Recall@10=%.4f  p50=%s p95=%s p99=%s  (%d queries)", pass, float64(hits)/float64(len(queries)), p(0.5), p(0.95), p(0.99), len(queries))
	}
}
