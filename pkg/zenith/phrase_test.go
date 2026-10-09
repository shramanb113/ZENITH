package zenith_test

import (
	"context"
	"path/filepath"
	"sort"
	"testing"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

var phraseDocs = map[string]string{
	"hit1":     "an introduction to machine learning",
	"hit2":     "Machine-Learning in production",
	"reversed": "learning machine design",
	"gap":      "machine based learning",
	"other":    "kubernetes cluster scheduling",
}

func phraseIDs(t *testing.T, db *zenith.DB, q string, opts ...zenith.SearchOption) []string {
	t.Helper()
	res, err := db.Search(context.Background(), q, append([]zenith.SearchOption{zenith.Limit(100)}, opts...)...)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, len(res))
	for i, r := range res {
		ids[i] = r.ID
	}
	sort.Strings(ids)
	return ids
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// A quoted phrase is honoured by DB.Search, with and without Explain (the HTTP
// sidecar always searches with Explain), and across a close and reopen.
func TestSearch_QuotedPhrase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "phrase.db")
	db, err := zenith.Open(path, zenith.WithBM25Only(), zenith.WithoutWordVectors())
	if err != nil {
		t.Fatal(err)
	}
	if err := db.AddBatch(context.Background(), phraseDocs); err != nil {
		t.Fatal(err)
	}

	want := []string{"hit1", "hit2"}
	check := func(stage string, db *zenith.DB) {
		t.Helper()
		if got := phraseIDs(t, db, `"machine learning"`); !sameStrings(got, want) {
			t.Fatalf("%s: phrase search = %v, want %v", stage, got, want)
		}
		if got := phraseIDs(t, db, `"machine learning"`, zenith.Explain()); !sameStrings(got, want) {
			t.Fatalf("%s: phrase search with Explain = %v, want %v", stage, got, want)
		}
		if got := phraseIDs(t, db, "machine learning"); len(got) < 4 {
			t.Fatalf("%s: plain search = %v, want the near-misses too", stage, got)
		}
		if got := phraseIDs(t, db, `"learning machine"`); !sameStrings(got, []string{"reversed"}) {
			t.Fatalf("%s: reversed phrase = %v", stage, got)
		}
	}
	check("open", db)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	db2, err := zenith.Open(path, zenith.WithBM25Only(), zenith.WithoutWordVectors())
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()
	check("reopened", db2)
}
