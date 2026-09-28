package index

import (
	"bufio"
	"context"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// Golden-file tests pin the on-disk format contract in FORMAT.md.
//
// testdata/golden-v5 is a REAL index written by the previous release's own
// library (format v5, a single gob file) together with expected.tsv: the search
// results that release returned for it. testdata/golden-v6 is the same corpus
// in the current segment format. Both must keep opening — v5 through Migrate —
// and keep returning those results. An accidental format break, or a migration
// that changes what a search returns, fails here.
//
// The expected scores come from the previous release, so the search config
// below pins the lexical behaviour that release had (exhaustive prefixes, fixed
// edit distance); ranking changes made since are tested elsewhere and must not
// be able to mask a format regression.

const goldenDim = 32

// goldenBag is the embedder the golden files were built with: a deterministic
// hashed bag-of-words with real vectors.
type goldenBag struct{}

func (goldenBag) Name() string    { return "test:bag32" }
func (goldenBag) Dimensions() int { return goldenDim }
func (b goldenBag) Embed(_ context.Context, text string) ([]float32, error) {
	v := make([]float32, goldenDim)
	for _, w := range strings.Fields(strings.ToLower(text)) {
		h := fnv.New32a()
		h.Write([]byte(w))
		s := h.Sum32()
		v[s%goldenDim] += 1
		v[(s>>8)%goldenDim] += 0.5
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
func (b goldenBag) EmbedBatch(ctx context.Context, t []string) ([][]float32, error) {
	out := make([][]float32, len(t))
	for i := range t {
		out[i], _ = b.Embed(ctx, t[i])
	}
	return out, nil
}

func goldenEngine() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = true
	cfg.FuzzyByLength = false  // the previous release applied FuzzyMaxDist to every word
	cfg.PrefixFragmentCap = -1 // ... and expanded every prefix
	e := NewEngine(cfg, goldenBag{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	e.SetANNThreshold(0)
	return e
}

type goldenCase struct {
	kind, query string
	want        []SearchResponse
}

func readGoldenExpected(t *testing.T, path string) []goldenCase {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var cases []goldenCase
	sc := bufio.NewScanner(io.Reader(f))
	for sc.Scan() {
		parts := strings.Split(sc.Text(), "\t")
		if len(parts) < 2 || (parts[0] != "Q" && parts[0] != "F") {
			continue
		}
		c := goldenCase{kind: parts[0], query: parts[1]}
		for _, kv := range parts[2:] {
			i := strings.LastIndex(kv, "=")
			score, err := strconv.ParseFloat(kv[i+1:], 64)
			if err != nil {
				t.Fatalf("bad expected entry %q: %v", kv, err)
			}
			c.want = append(c.want, SearchResponse{ID: kv[:i], Score: score})
		}
		cases = append(cases, c)
	}
	if len(cases) == 0 {
		t.Fatal("no expected cases in " + path)
	}
	return cases
}

func checkGolden(t *testing.T, e *Engine, cases []goldenCase) {
	t.Helper()
	ctx := context.Background()
	for _, c := range cases {
		var pred Predicate
		q := c.query
		if c.kind == "F" { // "<query>|topic=<value>"
			qq, cond, _ := strings.Cut(c.query, "|")
			q = qq
			_, val, _ := strings.Cut(cond, "=")
			pred = func(a Attrs) bool { return a["topic"].S == val }
		}
		got, err := e.SearchWithFilter(ctx, q, pred)
		if err != nil {
			t.Fatal(err)
		}
		if len(got) > 5 {
			got = got[:5] // expectations were recorded with Limit(5)
		}
		// pkg/zenith reports scores relative to the best hit (top = 1.0); the engine
		// returns raw RRF scores. Compare in the same units.
		if len(got) > 0 && got[0].Score > 0 {
			top := got[0].Score
			norm := make([]SearchResponse, len(got))
			for i, g := range got {
				norm[i] = SearchResponse{ID: g.ID, Score: g.Score / top}
			}
			got = norm
		}
		if len(got) != len(c.want) {
			t.Errorf("%q: %d results, want %d  (got %v)", c.query, len(got), len(c.want), got)
			continue
		}
		for i := range got {
			if got[i].ID != c.want[i].ID || math.Abs(got[i].Score-c.want[i].Score) > 1e-6 {
				t.Errorf("%q rank %d: got %s/%.10f, previous release returned %s/%.10f",
					c.query, i, got[i].ID, got[i].Score, c.want[i].ID, c.want[i].Score)
			}
		}
	}
}

func copyDir(t *testing.T, src string) string {
	t.Helper()
	dst := t.TempDir()
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, en := range entries {
		b, err := os.ReadFile(filepath.Join(src, en.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, en.Name()), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

// The version-5 file written by the previous release migrates and answers every
// recorded query exactly as that release did.
func TestGolden_V5_MigratesAndMatchesPreviousRelease(t *testing.T) {
	cases := readGoldenExpected(t, filepath.Join("testdata", "golden-v5", "expected.tsv"))
	dir := copyDir(t, filepath.Join("testdata", "golden-v5"))
	path := filepath.Join(dir, "golden.db")

	e := goldenEngine()
	if err := e.Load(path); err == nil {
		t.Fatal("a v5 file must be refused until it is migrated")
	}
	e.Close()

	res, err := Migrate(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if res.FromVersion != 5 || res.Embedder != "test:bag32" || res.Dims != goldenDim {
		t.Fatalf("migration result %+v", res)
	}
	if err := Verify(path); err != nil {
		t.Fatal(err)
	}
	m := goldenEngine()
	t.Cleanup(func() { m.Close() })
	if err := m.Load(path); err != nil {
		t.Fatal(err)
	}
	if got, want := m.Count(), 19; got != want { // 20 added, 1 deleted
		t.Fatalf("Count = %d, want %d", got, want)
	}
	checkGolden(t, m, cases)
}

// The version-6 index committed to the repo keeps opening and answering the same.
func TestGolden_V6_OpensAndMatches(t *testing.T) {
	cases := readGoldenExpected(t, filepath.Join("testdata", "golden-v5", "expected.tsv"))
	dir := copyDir(t, filepath.Join("testdata", "golden-v6"))
	info, err := Inspect(filepath.Join(dir, "golden.db"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Version != 6 || len(info.Segments) != 1 {
		t.Fatalf("golden-v6 header: %+v", info)
	}
	if err := Verify(filepath.Join(dir, "golden.db")); err != nil {
		t.Fatal(err)
	}
	e := goldenEngine()
	t.Cleanup(func() { e.Close() })
	if err := e.Load(filepath.Join(dir, "golden.db")); err != nil {
		t.Fatalf("the committed format-6 index no longer opens: %v", err)
	}
	checkGolden(t, e, cases)
	_ = fmt.Sprint // keep fmt for debugging additions
}
