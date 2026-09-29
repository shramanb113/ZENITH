package zenith_test

import (
	"math"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// A filter encoded to JSON and decoded again must select exactly the same
// documents as the builder-made one it came from.
func TestFilter_JSONRoundTripSelectsTheSameDocuments(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	filters := map[string]zenith.Filter{
		"eq string":     zenith.Eq("lang", "en"),
		"eq number":     zenith.Eq("year", 2022),
		"eq bool":       zenith.Eq("public", true),
		"in":            zenith.In("lang", "en", "fr"),
		"range":         zenith.Range("year", 2021, 2024),
		"open range":    zenith.Range("year", math.Inf(-1), 2021),
		"exists":        zenith.Exists("year"),
		"and":           zenith.And(zenith.Eq("lang", "en"), zenith.Range("year", 2021, 2030)),
		"or":            zenith.Or(zenith.Eq("lang", "fr"), zenith.Eq("year", 2024)),
		"not":           zenith.Not(zenith.Eq("lang", "en")),
		"nested":        zenith.And(zenith.Not(zenith.Eq("public", false)), zenith.Or(zenith.Eq("lang", "en"), zenith.Eq("lang", "fr"))),
		"type mismatch": zenith.Eq("year", "2022"),
		"empty in":      zenith.In("lang"),
		"empty and":     zenith.And(),
		"empty or":      zenith.Or(),
	}
	for name, f := range filters {
		data, err := f.JSON()
		if err != nil {
			t.Fatalf("%s: encode: %v", name, err)
		}
		back, err := zenith.FilterFromJSON(data)
		if err != nil {
			t.Fatalf("%s: decode %s: %v", name, data, err)
		}
		want, err := db.Search(bgCtx(), "kubernetes networking", zenith.Limit(50), zenith.WithFilter(f))
		if err != nil {
			t.Fatal(err)
		}
		got, err := db.Search(bgCtx(), "kubernetes networking", zenith.Limit(50), zenith.WithFilter(back))
		if err != nil {
			t.Fatal(err)
		}
		sameIDs(t, got, idsOf(want)...)
		t.Logf("%-14s %s -> %v", name, data, idsOf(got))
	}
}

func TestFilterFromJSON_DocumentedExample(t *testing.T) {
	db := openMem(t)
	seedFilterDocs(t, db)
	f, err := zenith.FilterFromJSON([]byte(`{"op":"and","args":[
		{"op":"eq","field":"lang","value":"en"},
		{"op":"range","field":"year","min":2021}]}`))
	if err != nil {
		t.Fatal(err)
	}
	res, err := db.Search(bgCtx(), "kubernetes networking", zenith.Limit(50), zenith.WithFilter(f))
	if err != nil {
		t.Fatal(err)
	}
	sameIDs(t, res, "en2024")
}

func TestFilterFromJSON_RejectsMalformedAndOversizedInput(t *testing.T) {
	deep := strings.Repeat(`{"op":"not","args":[`, 40) + `{"op":"exists","field":"a"}` + strings.Repeat(`]}`, 40)
	wide := `{"op":"or","args":[` + strings.Repeat(`{"op":"exists","field":"a"},`, 600) + `{"op":"exists","field":"a"}]}`
	for name, in := range map[string]string{
		"not json":             `{`,
		"unknown op":           `{"op":"regex","field":"a","value":"x"}`,
		"eq without value":     `{"op":"eq","field":"a"}`,
		"eq without field":     `{"op":"eq","value":1}`,
		"object value":         `{"op":"eq","field":"a","value":{"x":1}}`,
		"null value":           `{"op":"eq","field":"a","value":null}`,
		"array value":          `{"op":"in","field":"a","values":[[1]]}`,
		"range min>max":        `{"op":"range","field":"a","min":5,"max":1}`,
		"not with two args":    `{"op":"not","args":[{"op":"exists","field":"a"},{"op":"exists","field":"b"}]}`,
		"not with no args":     `{"op":"not"}`,
		"exists without field": `{"op":"exists"}`,
		"too deep":             deep,
		"too many nodes":       wide,
	} {
		if _, err := zenith.FilterFromJSON([]byte(in)); err == nil {
			t.Errorf("%s: accepted %.60q", name, in)
		}
	}
}
