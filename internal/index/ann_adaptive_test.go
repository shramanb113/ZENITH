package index

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

func testConfigWithANN(bandPct float64) *config.Config {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.ANNThresholdBandPct = bandPct
	return cfg
}

func testScorer(cfg *config.Config) ranking.Scorer {
	return ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight)
}

func testAnalyzer() analysis.Analyzer {
	return analysis.NewStandardAnalyzer()
}

func TestEWMA_StartsAtFirstObservation(t *testing.T) {
	var e ewma
	e.observe(10)
	if e.value() != 10 {
		t.Fatalf("value after first observe = %v, want 10", e.value())
	}
}

func TestEWMA_TracksRecentValuesMoreThanOld(t *testing.T) {
	var e ewma
	for i := 0; i < 50; i++ {
		e.observe(100) // long history at 100
	}
	for i := 0; i < 5; i++ {
		e.observe(10) // a few recent observations much lower
	}
	if e.value() >= 100 {
		t.Fatalf("value = %v after 5 low observations, want it to have moved well below 100", e.value())
	}
	if e.value() <= 10 {
		t.Fatalf("value = %v, want it still above the newest single observation (smoothed, not reset)", e.value())
	}
}

func TestShouldUseANN_OutsideBandUsesStaticThresholdUnchanged(t *testing.T) {
	cfg := testConfigWithANN(0) // band disabled
	e := NewEngine(cfg, noVecEmbedder{}, testScorer(cfg), testAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()
	e.annMinDocs = 1000

	if e.shouldUseANN(500) { // well below threshold, band disabled
		t.Fatal("shouldUseANN(500) = true with corpus well below static threshold and band disabled")
	}
	if !e.shouldUseANN(2000) { // well above threshold, band disabled
		t.Fatal("shouldUseANN(2000) = false with corpus well above static threshold and band disabled")
	}
}

func TestShouldUseANN_InBandPicksFasterMeasuredPath(t *testing.T) {
	cfg := testConfigWithANN(0.20) // ±20% band
	e := NewEngine(cfg, noVecEmbedder{}, testScorer(cfg), testAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()
	e.annMinDocs = 1000 // band is [800, 1200]

	// Force the ANN path's rolling average artificially high.
	for i := 0; i < 10; i++ {
		e.annLatency.observe(1000) // 1000ms
		e.exactLatency.observe(1)  // 1ms
	}
	if e.shouldUseANN(1000) { // inside the band
		t.Fatal("shouldUseANN(1000) = true even though exact scan's measured latency is far lower")
	}

	// Flip which path is faster.
	for i := 0; i < 10; i++ {
		e.annLatency.observe(1)
		e.exactLatency.observe(1000)
	}
	if !e.shouldUseANN(1000) {
		t.Fatal("shouldUseANN(1000) = false even though ANN's measured latency is now far lower")
	}
}

func TestShouldUseANN_OutsideBandIgnoresMeasuredLatency(t *testing.T) {
	cfg := testConfigWithANN(0.20)
	e := NewEngine(cfg, noVecEmbedder{}, testScorer(cfg), testAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()
	e.annMinDocs = 1000 // band is [800, 1200]

	// Even with ANN measured as much faster, outside the band the static
	// threshold decides unchanged: well below -> exact; well above -> ANN.
	for i := 0; i < 10; i++ {
		e.annLatency.observe(1)
		e.exactLatency.observe(1000)
	}
	if e.shouldUseANN(500) {
		t.Fatal("shouldUseANN(500) = true outside the band despite corpus being well below the static threshold")
	}
}

func TestEWMA_ConcurrentObserveAndValueIsSafe(t *testing.T) {
	var e ewma
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(v float64) {
			defer wg.Done()
			e.observe(v)
			_ = e.value()
			_ = e.isSet()
		}(float64(i))
	}
	wg.Wait()
	if !e.isSet() {
		t.Fatal("isSet() = false after concurrent observations")
	}
}

func TestElapsedMs_SubMillisecondPrecisionNotTruncated(t *testing.T) {
	got := elapsedMs(500 * time.Microsecond)
	if got != 0.5 {
		t.Fatalf("elapsedMs(500us) = %v, want 0.5 (not truncated to 0)", got)
	}
}

func TestVectorPass_DoesNotObserveLatencyWhenBandDisabled(t *testing.T) {
	cfg := testConfigWithANN(0) // band disabled, the default
	e := NewEngine(cfg, bagEmbedder{}, testScorer(cfg), testAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()

	ctx := context.Background()
	if err := e.AddBatch(ctx, bagDocs(50, 1)); err != nil {
		t.Fatal(err)
	}
	if _, err := e.Search(ctx, bagDocs(1, 1)[0].Text); err != nil {
		t.Fatal(err)
	}
	if e.exactLatency.isSet() {
		t.Fatal("exactLatency.isSet() = true with ANNThresholdBandPct disabled — latency should never be recorded when the feature is off")
	}
}
