//go:build cgo

package index

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// TestANNSidecarOpenTime measures, on the cached 107k-passage MS MARCO hybrid
// index with real gte-small vectors, what the persisted HNSW graph buys at open:
// the time to open when the graph must be built from the vectors, the time when
// it is restored from `<db>.ann`, and whether the two answer 300 real queries
// with identical results. It works on a copy, so the cache is not touched.
//
//	ZENITH_HYBRID_ENGINE=1 ZENITH_MSMARCO_HYBRID_INDEX=../../bench/.cache/hybrid-gte-small.db \
//	  go test ./internal/index -run TestANNSidecarOpenTime -v
func TestANNSidecarOpenTime(t *testing.T) {
	if os.Getenv("ZENITH_HYBRID_ENGINE") == "" {
		t.Skip("set ZENITH_HYBRID_ENGINE=1 and ZENITH_MSMARCO_HYBRID_INDEX=<cached hybrid index> to run")
	}
	const cacheDir = "../../bench/.cache"
	src := os.Getenv("ZENITH_MSMARCO_HYBRID_INDEX")
	if _, err := os.Stat(src); src == "" || err != nil {
		t.Skip("cached hybrid index missing (build it with TestMSMARCOHybrid)")
	}
	dir := t.TempDir()
	base := filepath.Base(src)
	entries, _ := os.ReadDir(filepath.Dir(src))
	for _, en := range entries {
		n := en.Name()
		if !strings.HasPrefix(n, base) || strings.HasSuffix(n, ".ann") || strings.HasSuffix(n, ".tmp") {
			continue
		}
		in, err := os.Open(filepath.Join(filepath.Dir(src), n))
		if err != nil {
			t.Fatal(err)
		}
		out, err := os.Create(filepath.Join(dir, n))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(out, in); err != nil {
			t.Fatal(err)
		}
		in.Close()
		out.Close()
	}
	path := filepath.Join(dir, base)

	_, _, queries, _ := loadMSMARCO(t, cacheDir, 100_000)
	queries = queries[:min(300, len(queries))]
	localEmb, err := localembedder.NewByID(os.Getenv("ZENITH_MODEL"), cacheDir+"/models")
	if err != nil {
		t.Fatalf("localembedder: %v", err)
	}
	cache, err := embedding.NewCachingEmbedder(localEmb, 10_000)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	open := func() (*Engine, time.Duration) {
		cfg := config.DefaultConfig()
		cfg.WordVectors = false
		eng := NewEngine(cfg, cache, ranking.NewWeightedRRFRanker(cfg.RRFConstant, 0, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
		eng.SetANNThreshold(defaultANNMinDocs)
		t0 := time.Now()
		if err := eng.Load(path); err != nil {
			t.Fatal(err)
		}
		return eng, time.Since(t0)
	}
	answers := func(eng *Engine) []string {
		var out []string
		for _, q := range queries {
			res, err := eng.Search(ctx, q.text)
			if err != nil {
				t.Fatal(err)
			}
			ids := make([]string, 0, 10)
			for i, r := range res {
				if i >= 10 {
					break
				}
				ids = append(ids, r.ID)
			}
			out = append(out, strings.Join(ids, ","))
		}
		return out
	}

	a, buildOpen := open()
	if !a.ANNActive() || a.ANNLoadedFromDisk() {
		t.Fatalf("first open: ANN active=%v fromDisk=%v; want built from vectors", a.ANNActive(), a.ANNLoadedFromDisk())
	}
	if err := a.SaveANN(); err != nil { // waits behind the background save open started
		t.Fatal(err)
	}
	want := answers(a)
	a.Close()
	fi, err := os.Stat(path + ".ann")
	if err != nil {
		t.Fatalf("no sidecar after SaveANN: %v", err)
	}

	b, loadOpen := open()
	defer b.Close()
	if !b.ANNLoadedFromDisk() {
		t.Fatal("second open did not restore the graph from the sidecar")
	}
	got := answers(b)
	diff := 0
	for i := range want {
		if want[i] != got[i] {
			diff++
		}
	}
	t.Logf("open, graph built from %d vectors: %s", b.Count(), buildOpen.Round(time.Millisecond))
	t.Logf("open, graph restored from %s (%.1f MB): %s", filepath.Base(path)+".ann", float64(fi.Size())/(1<<20), loadOpen.Round(time.Millisecond))
	t.Logf("results of %d real queries, restored vs built graph: %d differ", len(queries), diff)
	fmt.Fprintf(os.Stderr, "RESULT ann-open build=%s restore=%s file=%.1fMB differing=%d/%d\n", buildOpen, loadOpen, float64(fi.Size())/(1<<20), diff, len(queries))
	if diff != 0 {
		t.Errorf("%d of %d queries answered differently after restoring the graph", diff, len(queries))
	}
}
