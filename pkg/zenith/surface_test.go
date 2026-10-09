package zenith_test

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// Result.Attrs and DB.GetAttrs return what AddWithAttrs stored, in plain Go
// types (every number as float64, an array as []any).
func TestResultAttrsAndGetAttrsRoundTrip(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	if err := db.AddWithAttrs(bgCtx(), "tagged", "kubernetes cluster networking guide",
		zenith.Attrs{"tags": []string{"go", "k8s"}, "n": 3}); err != nil {
		t.Fatal(err)
	}

	rs, err := db.Search(bgCtx(), "kubernetes networking", zenith.Limit(50))
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]zenith.Result{}
	for _, r := range rs {
		byID[r.ID] = r
	}
	want := zenith.Attrs{"lang": "en", "year": float64(2020), "public": true}
	if got := byID["en2020"].Attrs; !reflect.DeepEqual(got, want) {
		t.Fatalf("en2020 Attrs = %#v, want %#v", got, want)
	}
	if got := byID["noattrs"].Attrs; got != nil {
		t.Fatalf("noattrs Attrs = %#v, want nil", got)
	}
	wantTagged := zenith.Attrs{"tags": []any{"go", "k8s"}, "n": float64(3)}
	if got := byID["tagged"].Attrs; !reflect.DeepEqual(got, wantTagged) {
		t.Fatalf("tagged Attrs = %#v, want %#v", got, wantTagged)
	}

	// Explain results carry attrs too.
	ex, err := db.Search(bgCtx(), "kubernetes", zenith.Explain(), zenith.Limit(50))
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range ex {
		if r.ID == "fr2022" && r.Attrs["lang"] != "fr" {
			t.Fatalf("explain fr2022 Attrs = %#v", r.Attrs)
		}
	}

	got, err := db.GetAttrs("tagged")
	if err != nil || !reflect.DeepEqual(got, wantTagged) {
		t.Fatalf("GetAttrs(tagged) = %#v, %v", got, err)
	}
	if got, err := db.GetAttrs("does-not-exist"); err != nil || got != nil {
		t.Fatalf("GetAttrs(missing) = %#v, %v; want nil, nil", got, err)
	}
	if _, err := db.GetAttrs(""); !errors.Is(err, zenith.ErrInvalidID) {
		t.Fatalf("GetAttrs(\"\") err = %v, want ErrInvalidID", err)
	}
}

func TestWithWeightsRejectsNonFiniteAndNegative(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	for _, w := range [][3]float64{{math.NaN(), 0, 0}, {0, math.Inf(1), 0}, {0, 0, -1}, {-2, 0, 0}} {
		if _, err := db.Search(bgCtx(), "kubernetes", zenith.WithWeights(w[0], w[1], w[2])); !errors.Is(err, zenith.ErrInvalidOption) {
			t.Fatalf("WithWeights%v: err = %v, want ErrInvalidOption", w, err)
		}
	}
	if _, err := db.Search(bgCtx(), "kubernetes", zenith.WithWeights(3, 0.5, 10)); err != nil {
		t.Fatalf("valid weights: %v", err)
	}
}

// Facets count every match, not just the returned page, and respect the filter.
func TestSearchWithFacetsAndCorpusFacets(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)

	rs, facets, err := db.SearchWithFacets(bgCtx(), "kubernetes networking", []string{"lang", "public"}, 0, zenith.Limit(1))
	if err != nil {
		t.Fatal(err)
	}
	if len(rs) != 1 {
		t.Fatalf("Limit(1) returned %d results", len(rs))
	}
	wantLang := []zenith.FacetCount{{Value: "en", Count: 2}, {Value: "fr", Count: 1}}
	if !reflect.DeepEqual(facets["lang"], wantLang) {
		t.Fatalf("lang facets = %#v, want %#v", facets["lang"], wantLang)
	}
	// public is true on en2020 and fr2022, false on en2024.
	wantPublic := []zenith.FacetCount{{Value: true, Count: 2}, {Value: false, Count: 1}}
	if !reflect.DeepEqual(facets["public"], wantPublic) {
		t.Fatalf("public facets = %#v, want %#v", facets["public"], wantPublic)
	}

	_, filtered, err := db.SearchWithFacets(bgCtx(), "kubernetes networking", []string{"lang"}, 0,
		zenith.WithFilter(zenith.Eq("public", true)))
	if err != nil {
		t.Fatal(err)
	}
	wantFiltered := []zenith.FacetCount{{Value: "en", Count: 1}, {Value: "fr", Count: 1}}
	if !reflect.DeepEqual(filtered["lang"], wantFiltered) {
		t.Fatalf("filtered lang facets = %#v, want %#v", filtered["lang"], wantFiltered)
	}

	corpus, err := db.Facets([]string{"lang", "year"}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(corpus["lang"], []zenith.FacetCount{{Value: "en", Count: 2}}) {
		t.Fatalf("corpus lang (top 1) = %#v", corpus["lang"])
	}
	if len(corpus["year"]) != 1 || corpus["year"][0].Count != 1 {
		t.Fatalf("corpus year (top 1) = %#v", corpus["year"])
	}

	// Plain Search ignores facets entirely and is unchanged.
	if _, err := db.Search(bgCtx(), "kubernetes"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Facets([]string{""}, 0); err == nil {
		t.Fatal("Facets with an empty field name: want error")
	}
}

func TestSuggest(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	got, err := db.Suggest("Netw", 10)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"network"}) {
		t.Fatalf("Suggest(Netw) = %v, want [network] (stemmed)", got)
	}
	if got, err := db.Suggest("", 10); err != nil || len(got) != 0 {
		t.Fatalf("Suggest(\"\") = %v, %v", got, err)
	}
}
