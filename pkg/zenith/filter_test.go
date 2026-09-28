package zenith_test

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/shramanb113/ZENITH/internal/storage/wal"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

func idsOf(rs []zenith.Result) []string {
	out := make([]string, len(rs))
	for i, r := range rs {
		out[i] = r.ID
	}
	sort.Strings(out)
	return out
}

func sameIDs(t *testing.T, got []zenith.Result, want ...string) {
	t.Helper()
	sort.Strings(want)
	g := idsOf(got)
	if len(g) != len(want) {
		t.Fatalf("ids = %v, want %v", g, want)
	}
	for i := range g {
		if g[i] != want[i] {
			t.Fatalf("ids = %v, want %v", g, want)
		}
	}
}

func seedFilterDocs(t *testing.T, db *zenith.DB) {
	t.Helper()
	docs := []struct {
		id, text string
		attrs    zenith.Attrs
	}{
		{"en2020", "kubernetes cluster networking guide", zenith.Attrs{"lang": "en", "year": 2020, "public": true}},
		{"en2024", "kubernetes cluster networking guide", zenith.Attrs{"lang": "en", "year": 2024, "public": false}},
		{"fr2022", "kubernetes cluster networking guide", zenith.Attrs{"lang": "fr", "year": 2022, "public": true}},
		{"noattrs", "kubernetes cluster networking guide", nil},
	}
	for _, d := range docs {
		if err := db.AddWithAttrs(bgCtx(), d.id, d.text, d.attrs); err != nil {
			t.Fatalf("AddWithAttrs(%s): %v", d.id, err)
		}
	}
}

func TestFilter_EqInRangeAndCombinators(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)

	cases := []struct {
		name string
		f    zenith.Filter
		want []string
	}{
		{"eq string", zenith.Eq("lang", "en"), []string{"en2020", "en2024"}},
		{"eq number", zenith.Eq("year", 2022), []string{"fr2022"}},
		{"eq bool", zenith.Eq("public", true), []string{"en2020", "fr2022"}},
		{"eq type mismatch", zenith.Eq("year", "2022"), nil},
		{"in", zenith.In("lang", "fr", "de"), []string{"fr2022"}},
		{"range inclusive", zenith.Range("year", 2020, 2022), []string{"en2020", "fr2022"}},
		{"range open end", zenith.Range("year", 2023, math.Inf(1)), []string{"en2024"}},
		{"exists", zenith.Exists("lang"), []string{"en2020", "en2024", "fr2022"}},
		{"and", zenith.And(zenith.Eq("lang", "en"), zenith.Eq("public", true)), []string{"en2020"}},
		{"or", zenith.Or(zenith.Eq("lang", "fr"), zenith.Range("year", 2024, 2024)), []string{"en2024", "fr2022"}},
		{"not includes attr-less", zenith.Not(zenith.Eq("lang", "en")), []string{"fr2022", "noattrs"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rs, err := db.Search(bgCtx(), "kubernetes networking", zenith.WithFilter(tc.f))
			if err != nil {
				t.Fatalf("Search: %v", err)
			}
			sameIDs(t, rs, tc.want...)
		})
	}
}

func TestFilter_NoFilterReturnsEverything(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	rs, err := db.Search(bgCtx(), "kubernetes networking")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	sameIDs(t, rs, "en2020", "en2024", "fr2022", "noattrs")
}

// The filter must shape the candidate set before ranking, so Limit(1) returns
// the best *matching* document even when the global best is filtered out.
func TestFilter_AppliesBeforeLimit(t *testing.T) {
	db := openMem(t)
	if err := db.AddWithAttrs(bgCtx(), "best", "zebra zebra zebra zebra", zenith.Attrs{"tier": "gold"}); err != nil {
		t.Fatal(err)
	}
	if err := db.AddWithAttrs(bgCtx(), "worse", "zebra crossing", zenith.Attrs{"tier": "silver"}); err != nil {
		t.Fatal(err)
	}
	rs, err := db.Search(bgCtx(), "zebra", zenith.WithFilter(zenith.Eq("tier", "silver")), zenith.Limit(1))
	if err != nil {
		t.Fatal(err)
	}
	sameIDs(t, rs, "worse")
}

// In hybrid mode the vector pass returns every document; the filter must
// prune that set too, not just the lexical one.
func TestFilter_HybridVectorPassIsFiltered(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithEmbedder(&customEmbedder{}))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.AddWithAttrs(bgCtx(), "en", "alpha bravo", zenith.Attrs{"lang": "en"}); err != nil {
		t.Fatal(err)
	}
	if err := db.AddWithAttrs(bgCtx(), "fr", "charlie delta", zenith.Attrs{"lang": "fr"}); err != nil {
		t.Fatal(err)
	}
	rs, err := db.Search(bgCtx(), "alpha", zenith.WithFilter(zenith.Eq("lang", "fr")))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range rs {
		if r.ID == "en" {
			t.Fatalf("filtered-out document %q leaked through the vector pass", r.ID)
		}
	}
}

func TestFilter_ReAddReplacesAttrs(t *testing.T) {
	db := openMem(t)
	if err := db.AddWithAttrs(bgCtx(), "d", "some text here", zenith.Attrs{"k": "v1"}); err != nil {
		t.Fatal(err)
	}
	if err := db.AddWithAttrs(bgCtx(), "d", "some text here", zenith.Attrs{"k": "v2"}); err != nil {
		t.Fatal(err)
	}
	rs, _ := db.Search(bgCtx(), "text", zenith.WithFilter(zenith.Eq("k", "v1")))
	sameIDs(t, rs)
	rs, _ = db.Search(bgCtx(), "text", zenith.WithFilter(zenith.Eq("k", "v2")))
	sameIDs(t, rs, "d")

	mustAdd(t, db, "d", "some text here")
	rs, _ = db.Search(bgCtx(), "text", zenith.WithFilter(zenith.Exists("k")))
	sameIDs(t, rs)
}

func TestFilter_DeleteRemovesDoc(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	if err := db.Delete(bgCtx(), "en2020"); err != nil {
		t.Fatal(err)
	}
	rs, _ := db.Search(bgCtx(), "kubernetes", zenith.WithFilter(zenith.Eq("lang", "en")))
	sameIDs(t, rs, "en2024")
}

func TestFilter_AddBatchWithAttrs(t *testing.T) {
	db := openMem(t)
	err := db.AddBatchWithAttrs(bgCtx(),
		map[string]string{"a": "shared words here", "b": "shared words here", "c": "shared words here"},
		map[string]zenith.Attrs{"a": {"grp": "x"}, "b": {"grp": "y"}})
	if err != nil {
		t.Fatal(err)
	}
	rs, _ := db.Search(bgCtx(), "shared", zenith.WithFilter(zenith.Eq("grp", "x")))
	sameIDs(t, rs, "a")
	rs, _ = db.Search(bgCtx(), "shared", zenith.WithFilter(zenith.Not(zenith.Exists("grp"))))
	sameIDs(t, rs, "c")
}

func TestFilter_InvalidAttrs(t *testing.T) {
	db := openMem(t)
	for name, a := range map[string]zenith.Attrs{
		"empty key":   {"": "x"},
		"slice value": {"k": []string{"a"}},
		"nil value":   {"k": nil},
		"nan":         {"k": math.NaN()},
	} {
		if err := db.AddWithAttrs(bgCtx(), "d", "text", a); !errors.Is(err, zenith.ErrInvalidAttrs) {
			t.Errorf("%s: err = %v, want ErrInvalidAttrs", name, err)
		}
	}
}

func TestFilter_CannotCombineWithExplain(t *testing.T) {
	db := openMem(t)
	mustAdd(t, db, "d", "text here")
	_, err := db.Search(bgCtx(), "text", zenith.Explain(), zenith.WithFilter(zenith.Exists("k")))
	if !errors.Is(err, zenith.ErrInvalidOption) {
		t.Fatalf("err = %v, want ErrInvalidOption", err)
	}
}

func TestFilter_SurvivesCleanReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "attrs.db")
	db1, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	seedFilterDocs(t, db1)
	if err := db1.Close(); err != nil {
		t.Fatal(err)
	}
	db2, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	rs, _ := db2.Search(bgCtx(), "kubernetes", zenith.WithFilter(zenith.Range("year", 2021, 2025)))
	sameIDs(t, rs, "en2024", "fr2022")
}

// Attributes written to the WAL by AddWithAttrs must be replayed after a
// crash (no checkpoint): simulate by writing the WAL via a live DB, then
// copying it aside so Close's checkpoint cannot mask the replay path.
func TestFilter_SurvivesWALReplay(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "walattrs.db")

	base, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	if err := base.Close(); err != nil {
		t.Fatal(err)
	}

	live, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	if err := live.AddWithAttrs(bgCtx(), "wal1", "crash recovered attribute doc", zenith.Attrs{"src": "wal"}); err != nil {
		t.Fatal(err)
	}
	walBytes := readFileBytes(t, path+".wal")
	if err := live.Close(); err != nil {
		t.Fatal(err)
	}
	// Restore the pre-checkpoint WAL and an empty baseline gob.
	writeFileBytes(t, path+".wal", walBytes)
	removeFile(t, path)
	w, recs, err := wal.OpenWAL(path+".wal", wal.WALConfig{SyncMode: wal.SyncAlways})
	if err != nil || len(recs) != 1 {
		t.Fatalf("WAL fixture: recs=%d err=%v", len(recs), err)
	}
	_ = w.Close()

	db, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer db.Close()
	rs, _ := db.Search(bgCtx(), "recovered", zenith.WithFilter(zenith.Eq("src", "wal")))
	sameIDs(t, rs, "wal1")
}

func readFileBytes(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func writeFileBytes(t *testing.T, p string, b []byte) {
	t.Helper()
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
}

func removeFile(t *testing.T, p string) {
	t.Helper()
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
}
