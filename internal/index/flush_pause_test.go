package index

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// randVecEmbedder returns a deterministic pseudo-random unit vector per text —
// the same size and storage cost as a real 384-dim embedding, without the cost
// of running a model.
type randVecEmbedder struct{ dims int }

func (e randVecEmbedder) Dimensions() int { return e.dims }
func (e randVecEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	h := uint64(1469598103934665603)
	for i := 0; i < len(text); i++ {
		h = (h ^ uint64(text[i])) * 1099511628211
	}
	r := rand.New(rand.NewSource(int64(h)))
	v := make([]float32, e.dims)
	var ss float64
	for i := range v {
		v[i] = float32(r.NormFloat64())
		ss += float64(v[i]) * float64(v[i])
	}
	n := float32(math.Sqrt(ss))
	for i := range v {
		v[i] /= n
	}
	return v, nil
}
func (e randVecEmbedder) EmbedBatch(ctx context.Context, ts []string) ([][]float32, error) {
	out := make([][]float32, len(ts))
	for i := range ts {
		out[i], _ = e.Embed(ctx, ts[i])
	}
	return out, nil
}

// TestFlushPause reports how long a checkpoint (an incremental flush of the
// in-memory delta) holds up searches, as a function of delta size, on an index
// that already has a large mapped base. It is a measurement, not an assertion —
// the numbers go in ROADMAP.md. Run:
//
//	ZENITH_FLUSH_PAUSE=1 go test ./internal/index -run TestFlushPause -v -timeout 1800s
func TestFlushPause(t *testing.T) {
	if os.Getenv("ZENITH_FLUSH_PAUSE") == "" {
		t.Skip("set ZENITH_FLUSH_PAUSE=1 to run (a measurement, several minutes)")
	}
	ctx := context.Background()
	words := synthVocab(8000)
	r := rand.New(rand.NewSource(3))
	zipf := rand.NewZipf(r, 1.15, 4, uint64(len(words)-1))
	mkDocs := func(prefix string, n int) []BatchDoc {
		docs := make([]BatchDoc, n)
		for i := range docs {
			txt := ""
			for j := 0; j < 80; j++ { // ~500 bytes, like a short passage
				txt += words[zipf.Uint64()] + " "
			}
			docs[i] = BatchDoc{ID: fmt.Sprintf("%s-%07d", prefix, i), Text: txt}
		}
		return docs
	}

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	eng := NewEngine(cfg, randVecEmbedder{dims: 384}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	eng.SetAutoCompact(false)
	defer eng.Close()
	path := filepath.Join(t.TempDir(), "pause.db")

	// A mapped base (default 100k docs), so the flush is measured against a big index.
	// ZENITH_FLUSH_PAUSE_BASE / ZENITH_FLUSH_PAUSE_DELTAS (comma list) shorten a run,
	// e.g. for profiling.
	base := 100_000
	if v, err := strconv.Atoi(os.Getenv("ZENITH_FLUSH_PAUSE_BASE")); err == nil && v > 0 {
		base = v
	}
	deltas := []int{1_000, 10_000, 30_000}
	if v := os.Getenv("ZENITH_FLUSH_PAUSE_DELTAS"); v != "" {
		deltas = nil
		for _, f := range strings.Split(v, ",") {
			if n, err := strconv.Atoi(strings.TrimSpace(f)); err == nil && n > 0 {
				deltas = append(deltas, n)
			}
		}
	}
	if err := eng.AddBatch(ctx, mkDocs("base", base)); err != nil {
		t.Fatal(err)
	}
	if err := eng.Save(path); err != nil {
		t.Fatal(err)
	}
	t.Logf("base: %d docs, %d segment(s)", eng.Count(), eng.SegmentCount())

	for di, delta := range deltas {
		if err := eng.AddBatch(ctx, mkDocs(fmt.Sprintf("d%d-%d", di, delta), delta)); err != nil {
			t.Fatal(err)
		}

		// Two concurrent searchers. The probe runs a query that matches almost
		// nothing (about a millisecond of work), so a long probe latency can only
		// be time spent waiting for the engine lock. The load searcher runs
		// ordinary queries so the flush competes with real traffic. Each records
		// (start, duration) so latencies before and during the flush compare.
		type sample struct {
			at  time.Time
			dur time.Duration
		}
		var stop atomic.Bool
		var wg sync.WaitGroup
		var mu sync.Mutex
		var probeS, loadS []sample
		run := func(qs []string, out *[]sample) {
			defer wg.Done()
			for i := 0; !stop.Load(); i++ {
				t0 := time.Now()
				eng.Search(ctx, qs[i%len(qs)])
				d := time.Since(t0)
				mu.Lock()
				*out = append(*out, sample{t0, d})
				mu.Unlock()
			}
		}
		wg.Add(2)
		go run([]string{"zzqxunique"}, &probeS)
		go run([]string{"kubernetes cluster", "search index ranking", words[10] + " " + words[40]}, &loadS)
		time.Sleep(3 * time.Second) // steady state before the flush

		t0 := time.Now()
		if err := eng.Save(path); err != nil {
			t.Fatal(err)
		}
		flushEnd := time.Now()
		flush := flushEnd.Sub(t0)
		time.Sleep(500 * time.Millisecond)
		stop.Store(true)
		wg.Wait()

		stat := func(ss []sample, from, to time.Time) (n int, p50, max time.Duration) {
			var ds []time.Duration
			for _, s := range ss {
				if !s.at.Before(from) && s.at.Before(to) {
					ds = append(ds, s.dur)
				}
			}
			if len(ds) == 0 {
				return 0, 0, 0
			}
			sort.Slice(ds, func(i, j int) bool { return ds[i] < ds[j] })
			return len(ds), ds[len(ds)/2], ds[len(ds)-1]
		}
		bn, bp50, bmax := stat(probeS, t0.Add(-3*time.Second), t0)
		dn, dp50, dmax := stat(probeS, t0, flushEnd)
		ln, lp50, lmax := stat(loadS, t0.Add(-3*time.Second), t0)
		mn, mp50, mmax := stat(loadS, t0, flushEnd)
		t.Logf("delta %6d docs: Save took %s", delta, flush.Round(time.Millisecond))
		t.Logf("    probe (~1ms query) before: n=%d p50=%s max=%s | during Save: n=%d p50=%s max=%s",
			bn, bp50.Round(10*time.Microsecond), bmax.Round(10*time.Microsecond), dn, dp50.Round(10*time.Microsecond), dmax.Round(10*time.Microsecond))
		t.Logf("    load  (typical queries) before: n=%d p50=%s max=%s | during Save: n=%d p50=%s max=%s",
			ln, lp50.Round(time.Millisecond), lmax.Round(time.Millisecond), mn, mp50.Round(time.Millisecond), mmax.Round(time.Millisecond))
	}

	// Compaction runs outside the engine lock: stall should stay small.
	var stop atomic.Bool
	var wg sync.WaitGroup
	var maxLat atomic.Int64
	wg.Add(1)
	go func() {
		defer wg.Done()
		for !stop.Load() {
			t0 := time.Now()
			eng.Search(ctx, words[10]+" "+words[40])
			if d := int64(time.Since(t0)); d > maxLat.Load() {
				maxLat.Store(d)
			}
		}
	}()
	t0 := time.Now()
	segs := eng.SegmentCount()
	if err := eng.Compact(); err != nil {
		t.Fatal(err)
	}
	comp := time.Since(t0)
	stop.Store(true)
	wg.Wait()
	sizes := []int64{}
	if info, err := Inspect(path); err == nil {
		for _, s := range info.Segments {
			if st, err := os.Stat(s); err == nil {
				sizes = append(sizes, st.Size())
			}
		}
	}
	sort.Slice(sizes, func(i, j int) bool { return sizes[i] < sizes[j] })
	t.Logf("compact %d segments -> %d (%d docs) took %s; longest search stall during it %s; result size %v bytes",
		segs, eng.SegmentCount(), eng.Count(), comp.Round(time.Millisecond), time.Duration(maxLat.Load()).Round(time.Millisecond), sizes)
}
