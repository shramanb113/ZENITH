package index

import (
	"bufio"
	"context"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestScale measures the engine at a corpus size the other tests do not reach.
// Text is real: the first N passages of MS MARCO. Vectors are synthetic —
// clustered unit vectors of the real dimension (384), so the ANN graph has
// realistic structure and the vector path has realistic cost, but no claim is
// made about semantic quality (embedding a million passages takes hours; the
// model comparison is in TestModelComparison / TestSciFactHybrid).
//
//	ZENITH_SCALE=1 [ZENITH_SCALE_DOCS=1000000] [ZENITH_SCALE_QUERIES=3000] \
//	  [ZENITH_SCALE_REUSE=1] [ZENITH_SCALE_REBUILD=1] \
//	  go test ./internal/index -run TestScale -v -timeout 6h
//
// Reported: ingest rate and peak memory, flush and compaction times, open time
// with the persisted ANN graph (and, with ZENITH_SCALE_REBUILD=1, without it),
// resident and private memory after open and after queries, hybrid latency with
// the query vector precomputed (query embedding is model inference, measured
// separately), lexical-only latency, and ANN recall against an exact scan.

const scaleDims = 384

type scaleEmbedder struct {
	centers [][]float32
	lexical atomic.Bool // true: Embed fails, so the search is lexical only
}

func newScaleEmbedder() *scaleEmbedder {
	r := rand.New(rand.NewSource(20260928))
	e := &scaleEmbedder{centers: make([][]float32, 1500)}
	for i := range e.centers {
		c := make([]float32, scaleDims)
		for j := range c {
			c[j] = float32(r.NormFloat64())
		}
		e.centers[i] = c
	}
	return e
}

func (e *scaleEmbedder) Name() string    { return "test:synth384" }
func (e *scaleEmbedder) Dimensions() int { return scaleDims }

func hash64(s string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return h.Sum64()
}

// vecFor is a deterministic clustered unit vector for key: the same key always
// gets the same vector, near the centre its hash selects.
func (e *scaleEmbedder) vecFor(key string, noise float64) []float32 {
	h := hash64(key)
	r := rand.New(rand.NewSource(int64(h)))
	c := e.centers[h%uint64(len(e.centers))]
	v := make([]float32, scaleDims)
	var ss float64
	for i := range v {
		x := float64(c[i]) + noise*r.NormFloat64()
		v[i] = float32(x)
		ss += x * x
	}
	inv := float32(1 / math.Sqrt(ss))
	for i := range v {
		v[i] *= inv
	}
	return v
}

func (e *scaleEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if e.lexical.Load() {
		return nil, fmt.Errorf("scale test: lexical-only mode")
	}
	return e.vecFor("q:"+text, 0.25), nil
}

func (e *scaleEmbedder) EmbedBatch(ctx context.Context, ts []string) ([][]float32, error) {
	out := make([][]float32, len(ts))
	for i, t := range ts {
		v, err := e.Embed(ctx, t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func envInt(name string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(name)); err == nil && v > 0 {
		return v
	}
	return def
}

func mb(b uint64) string { return fmt.Sprintf("%.0f MB", float64(b)/(1<<20)) }

func dirSize(dir, prefix string) int64 {
	var n int64
	ents, _ := os.ReadDir(dir)
	for _, e := range ents {
		if strings.HasPrefix(e.Name(), prefix) {
			if fi, err := e.Info(); err == nil {
				n += fi.Size()
			}
		}
	}
	return n
}

func newScaleEngine(emb *scaleEmbedder) *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	return NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
}

func pctile(l []time.Duration, p float64) time.Duration {
	s := append([]time.Duration(nil), l...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[int(float64(len(s)-1)*p)]
}

func TestScale(t *testing.T) {
	if os.Getenv("ZENITH_SCALE") == "" {
		t.Skip("set ZENITH_SCALE=1 to run (needs bench/.cache; takes tens of minutes)")
	}
	const cacheDir = "../../bench/.cache"
	n := envInt("ZENITH_SCALE_DOCS", 1_000_000)
	nq := envInt("ZENITH_SCALE_QUERIES", 3000)
	ctx := context.Background()
	dir := filepath.Join(cacheDir, fmt.Sprintf("scale-%d", n))
	path := filepath.Join(dir, "scale.db")
	emb := newScaleEmbedder()
	report := func(format string, a ...any) {
		line := fmt.Sprintf(format, a...)
		t.Log(line)
		fmt.Fprintln(os.Stderr, "SCALE "+line)
	}
	memNow := func(label string) {
		ws, priv := procMem()
		var ms runtime.MemStats
		runtime.ReadMemStats(&ms)
		report("memory %-28s working set %s, private %s (file-backed resident ~%s), Go heap in use %s", label, mb(ws), mb(priv), mb(ws-min(ws, priv)), mb(ms.HeapInuse))
	}

	// ---- 1. build (or reuse) the index ----
	_, statErr := os.Stat(path)
	if statErr == nil && os.Getenv("ZENITH_SCALE_REUSE") == "" {
		os.RemoveAll(dir)
		statErr = os.ErrNotExist
	}
	if statErr != nil {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		eng := newScaleEngine(emb)
		t0 := time.Now()
		f, err := os.Open(filepath.Join(cacheDir, "collection.tsv"))
		if err != nil {
			t.Skipf("MS MARCO collection missing: %v", err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<20)
		const batchSize = 10_000
		const flushEvery = 100_000
		batch := make([]BatchDoc, 0, batchSize)
		done := 0
		var flushTotal time.Duration
		flushBatch := func() {
			if len(batch) == 0 {
				return
			}
			if err := eng.AddBatch(ctx, batch); err != nil {
				t.Fatal(err)
			}
			done += len(batch)
			batch = batch[:0]
			if done%flushEvery == 0 {
				ts := time.Now()
				if err := eng.Save(path); err != nil {
					t.Fatal(err)
				}
				flushTotal += time.Since(ts)
				ws, priv := procMem()
				report("ingested %8d docs in %s (%.0f docs/s), last flush %s; working set %s, private %s, %d segments",
					done, time.Since(t0).Round(time.Second), float64(done)/time.Since(t0).Seconds(),
					time.Since(ts).Round(time.Millisecond), mb(ws), mb(priv), eng.SegmentCount())
			}
		}
		for sc.Scan() && done+len(batch) < n {
			parts := strings.SplitN(sc.Text(), "\t", 2)
			if len(parts) < 2 {
				continue
			}
			vec := emb.vecFor("d:"+parts[0], 0.35)
			batch = append(batch, BatchDoc{ID: parts[0], Text: parts[1], Vector: vec})
			if len(batch) == batchSize {
				flushBatch()
			}
		}
		flushBatch()
		f.Close()
		ingest := time.Since(t0)
		if err := eng.Save(path); err != nil {
			t.Fatal(err)
		}
		report("ingest of %d docs: %s total (%.0f docs/s); %s of it in flush calls; %d segments; vectors=%d", done,
			ingest.Round(time.Second), float64(done)/ingest.Seconds(), flushTotal.Round(time.Second), eng.SegmentCount(), eng.vectorCount())
		memNow("after ingest (before compact)")

		tc := time.Now()
		if err := eng.Compact(); err != nil {
			t.Fatal(err)
		}
		report("compaction to %d segment(s): %s", eng.SegmentCount(), time.Since(tc).Round(time.Millisecond))
		ts := time.Now()
		if err := eng.SaveANN(); err != nil {
			t.Fatal(err)
		}
		report("ANN graph save: %s, file %s", time.Since(ts).Round(time.Millisecond), mb(uint64(dirSize(dir, "scale.db.ann"))))
		report("index on disk: %s (%.2f KB/doc)", mb(uint64(dirSize(dir, "scale.db"))), float64(dirSize(dir, "scale.db"))/1024/float64(done))
		if err := eng.Close(); err != nil {
			t.Fatal(err)
		}
	}

	// ---- 2. open ----
	openIndex := func(label string) *Engine {
		runtime.GC()
		eng := newScaleEngine(emb)
		t0 := time.Now()
		if err := eng.Load(path); err != nil {
			t.Fatal(err)
		}
		report("open (%s): %s; docs=%d segments=%d ANN active=%v restored-from-file=%v", label,
			time.Since(t0).Round(time.Millisecond), eng.Count(), eng.SegmentCount(), eng.ANNActive(), eng.ANNLoadedFromDisk())
		memNow("after open (" + label + ")")
		return eng
	}
	eng := openIndex("graph restored from its file")
	defer eng.Close()
	if !eng.ANNLoadedFromDisk() {
		t.Fatal("the ANN graph was not restored from its file")
	}

	// ---- 3. queries ----
	var queries []string
	scanFile(t, cacheDir+`\queries.dev.small.tsv`, func(parts []string) bool {
		if len(parts) >= 2 {
			queries = append(queries, parts[1])
		}
		return len(queries) < nq
	})
	run := func(label string, qs []string) []time.Duration {
		lat := make([]time.Duration, 0, len(qs))
		for _, q := range qs {
			t0 := time.Now()
			if _, err := eng.Search(ctx, q); err != nil {
				t.Fatal(err)
			}
			lat = append(lat, time.Since(t0))
		}
		report("%-34s p50 %s  p95 %s  p99 %s  (%d queries)", label,
			pctile(lat, 0.5).Round(10*time.Microsecond), pctile(lat, 0.95).Round(10*time.Microsecond), pctile(lat, 0.99).Round(10*time.Microsecond), len(lat))
		return lat
	}
	warm := min(200, len(queries))
	run("hybrid, first 200 queries (cold)", queries[:warm])
	memNow("after 200 queries")
	run("hybrid, steady state", queries)
	memNow("after all queries")
	emb.lexical.Store(true)
	run("lexical only", queries)
	emb.lexical.Store(false)

	// ---- 4. ANN recall against an exact scan ----
	sample := queries[:min(100, len(queries))]
	var sum float64
	for _, q := range sample {
		qv := normalizeVector(emb.vecFor("q:"+q, 0.25))
		type hit struct {
			id uint64
			s  float64
		}
		top := make([]hit, 0, 11)
		eng.mu.RLock()
		eng.eachVector(func(id uint64, v []uint16) {
			s := ann.DotF32F16(qv, v)
			if len(top) < 10 || s > top[len(top)-1].s {
				top = append(top, hit{id, s})
				sort.Slice(top, func(i, j int) bool { return top[i].s > top[j].s })
				if len(top) > 10 {
					top = top[:10]
				}
			}
		})
		got, ok := eng.annSearch(qv, nil)
		eng.mu.RUnlock()
		if !ok {
			t.Fatal("ANN declined an unfiltered search")
		}
		want := map[uint64]bool{}
		for _, h := range top {
			want[h.id] = true
		}
		hits := 0
		for i, h := range got {
			if i >= 10 {
				break
			}
			if want[h.ID] {
				hits++
			}
		}
		sum += float64(hits) / 10
	}
	report("ANN recall@10 vs exact scan over %d docs: %.4f (%d queries)", eng.Count(), sum/float64(len(sample)), len(sample))

	// ---- 5. open without the graph file: the cost persistence removes ----
	if os.Getenv("ZENITH_SCALE_REBUILD") != "" {
		eng.Close()
		os.Rename(annFilePath(path), annFilePath(path)+".off")
		rebuilt := openIndex("graph rebuilt from vectors")
		defer rebuilt.Close()
		os.Rename(annFilePath(path)+".off", annFilePath(path))
	}
	runtime.KeepAlive(sync.Mutex{})
}
