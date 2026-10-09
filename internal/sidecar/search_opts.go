package sidecar

import (
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// searchExtras are the optional search-body fields shared by the namespace and
// collection search endpoints, beyond queries/limit/min_semantic/filter:
//
//	"offset":  10                                         skip this many hits per query
//	"weights": {"vector": 2, "phonetic": 0.3, "rrf_k": 20} per-call ranking overrides (0/absent = default)
//	"sort":    {"field": "year", "desc": true}            order by an attribute instead of score
//	"facets":  {"fields": ["lang", "tags"], "top_k": 10}  per-value counts over each query's matches
type searchExtras struct {
	Offset  int `json:"offset"`
	Weights *struct {
		Vector   float64 `json:"vector"`
		Phonetic float64 `json:"phonetic"`
		RRFK     float64 `json:"rrf_k"`
	} `json:"weights"`
	Sort *struct {
		Field string `json:"field"`
		Desc  bool   `json:"desc"`
	} `json:"sort"`
	Facets *struct {
		Fields []string `json:"fields"`
		TopK   int      `json:"top_k"`
	} `json:"facets"`
}

// maxOffset bounds offset (the engine never returns more than its internal
// candidate cap anyway — 1000 by default — so a larger offset only overflows).
const maxOffset = 100_000

// maxFacetFields bounds how many fields one request may facet on.
const maxFacetFields = 32

// options validates the extras and turns them into search options, plus the
// facet request (nil fields = no facets). Weight values are validated by
// zenith.WithWeights itself (Search then fails with ErrInvalidOption).
func (x *searchExtras) options() (opts []zenith.SearchOption, facetFields []string, facetTopK int, err error) {
	if x.Offset < 0 || x.Offset > maxOffset {
		return nil, nil, 0, errors.New("offset must be between 0 and 100000")
	}
	if x.Weights != nil {
		if x.Weights.Vector < 0 || x.Weights.Phonetic < 0 || x.Weights.RRFK < 0 {
			return nil, nil, 0, errors.New("weights must not be negative")
		}
		opts = append(opts, zenith.WithWeights(x.Weights.Vector, x.Weights.Phonetic, x.Weights.RRFK))
	}
	if x.Sort != nil && x.Sort.Field != "" {
		opts = append(opts, zenith.SortBy(x.Sort.Field, x.Sort.Desc))
	}
	if x.Facets != nil && len(x.Facets.Fields) > 0 {
		if len(x.Facets.Fields) > maxFacetFields {
			return nil, nil, 0, errors.New("facets.fields has too many entries")
		}
		for _, f := range x.Facets.Fields {
			if f == "" {
				return nil, nil, 0, errors.New("facets.fields must not contain an empty name")
			}
		}
		if x.Facets.TopK < 0 {
			return nil, nil, 0, errors.New("facets.top_k must not be negative")
		}
		facetTopK = x.Facets.TopK
		if facetTopK == 0 {
			facetTopK = 10
		}
		facetFields = x.Facets.Fields
	}
	return opts, facetFields, facetTopK, nil
}

// skip drops the first offset results (the page is fetched as offset+limit).
func skip(res []zenith.Result, offset int) []zenith.Result {
	if offset >= len(res) {
		return res[:0]
	}
	return res[offset:]
}

type facetOut struct {
	Value any `json:"value"`
	Count int `json:"count"`
}

func facetsOut(f zenith.Facets) map[string][]facetOut {
	if f == nil {
		return nil
	}
	out := make(map[string][]facetOut, len(f))
	for field, list := range f {
		vals := make([]facetOut, len(list))
		for i, c := range list {
			vals[i] = facetOut{c.Value, c.Count}
		}
		out[field] = vals
	}
	return out
}

// suggestParams reads ?q=<prefix>&n=<count> for the suggest endpoints.
func suggestParams(r *http.Request) (prefix string, n int, err error) {
	prefix = r.URL.Query().Get("q")
	if strings.TrimSpace(prefix) == "" {
		return "", 0, errors.New("q (the prefix) is required")
	}
	if s := r.URL.Query().Get("n"); s != "" {
		if n, err = strconv.Atoi(s); err != nil || n < 0 {
			return "", 0, errors.New("n must be a non-negative integer")
		}
	}
	return prefix, n, nil
}

// facetParams reads ?fields=a,b&top_k=<k> for the corpus-wide facet endpoints.
func facetParams(r *http.Request) (fields []string, topK int, err error) {
	for _, f := range strings.Split(r.URL.Query().Get("fields"), ",") {
		if f = strings.TrimSpace(f); f != "" {
			fields = append(fields, f)
		}
	}
	if len(fields) == 0 {
		return nil, 0, errors.New("fields (comma-separated attribute names) is required")
	}
	if len(fields) > maxFacetFields {
		return nil, 0, errors.New("too many fields")
	}
	if s := r.URL.Query().Get("top_k"); s != "" {
		if topK, err = strconv.Atoi(s); err != nil || topK < 0 {
			return nil, 0, errors.New("top_k must be a non-negative integer")
		}
	}
	if topK == 0 {
		topK = 10 // same default as a search body's facets.top_k
	}
	return fields, topK, nil
}
