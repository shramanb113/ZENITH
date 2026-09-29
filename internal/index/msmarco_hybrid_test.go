//go:build cgo

package index

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestMSMARCOHybrid measures the full hybrid pipeline (lexical + dense + RRF) on
// real MS MARCO passages with the real embedded ONNX model:
//
//   - Recall@10 and latency percentiles with the ANN graph active (the default
//     above WithANNThreshold docs) and with the exact vector scan;
//   - the top-10 overlap between the two, i.e. ANN recall on *real* 384-dim
//     embeddings (the roadmap flagged that only synthetic vectors had been
//     measured);
//   - the index's on-disk size and memory-mapped footprint.
//
// The first run embeds ~107k passages (minutes) and caches the index in the
// segment format at $ZENITH_MSMARCO_HYBRID_INDEX; later runs map it in seconds.
//
// Run: ZENITH_MSMARCO_HYBRID=1 ZENITH_MSMARCO_HYBRID_INDEX=<path> \
//
//	go test ./internal/index -run TestMSMARCOHybrid -v -timeout 7200s
//
// Optional: ZENITH_MSMARCO_QUERIES=<n> (default 1000).
func TestMSMARCOHybrid(t *testing.T) {
	if os.Getenv("ZENITH_MSMARCO_HYBRID") == "" {
		t.Skip("set ZENITH_MSMARCO_HYBRID=1 to run (embeds the MS MARCO corpus on first run)")
	}
	const cacheDir = "../../bench/.cache"
	nQueries := 1000
	if n, _ := strconv.Atoi(os.Getenv("ZENITH_MSMARCO_QUERIES")); n > 0 {
		nQueries = n
	}
	idxPath := os.Getenv("ZENITH_MSMARCO_HYBRID_INDEX")

	passages, augmented, queries, qrels := loadMSMARCO(t, cacheDir, 100_000)
	if nQueries < len(queries) {
		queries = queries[:nQueries]
	}

	// ZENITH_MODEL selects a registered model (default: the bundled one); non-bundled
	// models are read from bench/.cache/models (see `zenith models pull`).
	localEmb, err := localembedder.NewByID(os.Getenv("ZENITH_MODEL"), cacheDir+"/models")
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}
	modelID := localEmb.Spec().ID
	emb, err := embedding.NewCachingEmbedder(localEmb, 10_000)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.DefaultConfig()
	cfg.WordVectors = false // word vectors only feed neural expansion; skip the cost here
	eng := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	eng.SetANNThreshold(0) // decide explicitly below
	defer eng.Close()

	loaded := false
	if idxPath != "" {
		if _, err := os.Stat(idxPath); err == nil {
			if err := eng.Load(idxPath); err != nil {
				t.Fatalf("loading cached index: %v", err)
			}
			loaded = true
		}
	}
	if !loaded {
		batch := make([]BatchDoc, 0, len(passages)+len(augmented))
		for id, text := range passages {
			batch = append(batch, BatchDoc{ID: id, Text: text})
		}
		for id, text := range augmented {
			batch = append(batch, BatchDoc{ID: id, Text: text})
		}
		start := time.Now()
		// Flush in slices so a long embedding run leaves a usable index behind
		// and each flush is a modest segment.
		const slice = 20_000
		for lo := 0; lo < len(batch); lo += slice {
			hi := min(lo+slice, len(batch))
			if err := eng.AddBatch(context.Background(), batch[lo:hi]); err != nil {
				t.Fatal(err)
			}
			if idxPath != "" {
				if err := eng.Save(idxPath); err != nil {
					t.Fatal(err)
				}
			}
			t.Logf("indexed %d/%d docs (%s)", hi, len(batch), time.Since(start).Round(time.Second))
		}
		if idxPath != "" {
			if err := eng.Compact(); err != nil {
				t.Fatal(err)
			}
		}
	}
	t.Logf("corpus: %d live docs, %d segment(s)", eng.Count(), eng.SegmentCount())

	ctx := context.Background()
	type run struct {
		name    string
		top     [][]string
		recall  float64
		lat     []time.Duration
		samples int
	}
	measure := func(name string) run {
		r := run{name: name, samples: len(queries)}
		hits := 0
		for _, q := range queries {
			t0 := time.Now()
			res, err := eng.Search(ctx, q.text)
			r.lat = append(r.lat, time.Since(t0))
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]string, 0, 10)
			hit := false
			for i, x := range res {
				if i >= 10 {
					break
				}
				ids = append(ids, x.ID)
				if x.ID == qrels[q.id] {
					hit = true
				}
			}
			if hit {
				hits++
			}
			r.top = append(r.top, ids)
		}
		r.recall = float64(hits) / float64(len(queries))
		sort.Slice(r.lat, func(i, j int) bool { return r.lat[i] < r.lat[j] })
		return r
	}
	pct := func(l []time.Duration, p float64) time.Duration { return l[int(float64(len(l)-1)*p)] }

	// A query costs its embedding (model inference) plus the engine's work. The
	// caching embedder makes a repeated query free, so to price a search honestly
	// every "cold" pass starts from an empty cache, and the "warm" pass right
	// after it (same queries, cache full) isolates the engine's own cost.
	freshCache := func() {
		c, err := embedding.NewCachingEmbedder(localEmb, 10_000)
		if err != nil {
			t.Fatal(err)
		}
		eng.embedder = c
	}
	freshCache()
	embLat := make([]time.Duration, 0, len(queries))
	for _, q := range queries {
		t0 := time.Now()
		if _, err := eng.embedder.Embed(ctx, q.text); err != nil {
			t.Fatal(err)
		}
		embLat = append(embLat, time.Since(t0))
	}
	sort.Slice(embLat, func(i, j int) bool { return embLat[i] < embLat[j] })

	coldWarm := func(label string, serial bool) (cold, warm run) {
		eng.serialEmbed = serial
		freshCache()
		cold = measure(label + ", cold cache")
		warm = measure(label + ", warm cache (engine only)")
		eng.serialEmbed = false
		return cold, warm
	}

	eng.SetANNThreshold(0)
	exactSerial, _ := coldWarm("exact scan, embed then search", true)
	exactCold, exactWarm := coldWarm("exact scan, embed overlapped", false)
	eng.SetANNThreshold(defaultANNMinDocs)
	if !eng.ANNActive() {
		t.Fatal("ANN did not activate")
	}
	annSerial, _ := coldWarm("ANN, embed then search", true)
	annCold, annWarm := coldWarm("ANN, embed overlapped", false)

	var overlap float64
	for i := range queries {
		want := map[string]bool{}
		for _, id := range exactWarm.top[i] {
			want[id] = true
		}
		hit := 0
		for _, id := range annWarm.top[i] {
			if want[id] {
				hit++
			}
		}
		if len(want) > 0 {
			overlap += float64(hit) / float64(len(want))
		}
	}
	overlap /= float64(len(queries))

	t.Logf("query embedding alone (%s): p50 %s  p95 %s  p99 %s", modelID,
		pct(embLat, 0.50).Round(10*time.Microsecond), pct(embLat, 0.95).Round(10*time.Microsecond), pct(embLat, 0.99).Round(10*time.Microsecond))
	for _, r := range []run{exactSerial, exactCold, exactWarm, annSerial, annCold, annWarm} {
		t.Logf("%-52s Recall@10=%.4f  p50=%s p95=%s p99=%s  (%d queries)", r.name, r.recall,
			pct(r.lat, 0.50).Round(10*time.Microsecond), pct(r.lat, 0.95).Round(10*time.Microsecond), pct(r.lat, 0.99).Round(10*time.Microsecond), r.samples)
	}
	t.Logf("ANN vs exact: top-10 overlap %.4f on real %s embeddings", overlap, modelID)
	fmt.Fprintf(os.Stderr, "RESULT hybrid model=%s embed_p50=%s exact_recall=%.4f exact_serial_p50=%s exact_overlapped_p50=%s exact_engine_p50=%s ann_recall=%.4f ann_serial_p50=%s ann_overlapped_p50=%s ann_engine_p50=%s overlap=%.4f\n",
		modelID, pct(embLat, 0.5), exactWarm.recall, pct(exactSerial.lat, 0.5), pct(exactCold.lat, 0.5), pct(exactWarm.lat, 0.5),
		annWarm.recall, pct(annSerial.lat, 0.5), pct(annCold.lat, 0.5), pct(annWarm.lat, 0.5), overlap)
	runtime.KeepAlive(eng)
}
