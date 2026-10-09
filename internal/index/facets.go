package index

import (
	"errors"
	"sort"
)

// FacetCount is how many documents carry one value of a faceted attribute.
type FacetCount struct {
	Value AttrValue // never AttrArray: an array attribute is counted per element
	Count int
}

// Facets maps a field name to its value counts, highest count first (ties by
// value, so the order is deterministic).
type Facets map[string][]FacetCount

// ErrEmptyFacetField is returned when a requested facet field name is empty.
var ErrEmptyFacetField = errors.New("index: facet field names must not be empty")

// FacetCounts counts, for each field, how many of the documents in resultIDs
// carry each value of that attribute. resultIDs is meant to be a bounded
// candidate list — typically the fused search result IDs, before pagination —
// not the whole corpus (use CorpusFacetCounts for that). An array attribute
// counts once per distinct element, matching the "any element matches"
// semantics filters use (anyMatch), so a document tagged [a, b] adds one to
// both a and b. Duplicate or unknown IDs, and documents without the field,
// contribute nothing. topK <= 0 keeps every value; otherwise only the topK
// highest counts per field are returned. A field no document carries maps to
// an empty slice.
//
// Holds Engine.mu.RLock for the duration.
func (e *Engine) FacetCounts(resultIDs []string, fields []string, topK int) (Facets, error) {
	if err := checkFacetFields(fields); err != nil {
		return nil, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()

	counts := make([]map[attrKey]int, len(fields))
	for i := range counts {
		counts[i] = make(map[attrKey]int)
	}
	seenDoc := make(map[uint64]struct{}, len(resultIDs))
	for _, raw := range resultIDs {
		id := internalDocID(raw)
		if _, dup := seenDoc[id]; dup {
			continue
		}
		seenDoc[id] = struct{}{}
		a := e.attrs[id]
		if len(a) == 0 {
			continue
		}
		for i, f := range fields {
			v, ok := a[f]
			if !ok {
				continue
			}
			if v.Kind != AttrArray {
				counts[i][keyOf(v)]++
				continue
			}
			// Count each distinct element once per document: [a, a] is one
			// document carrying a, not two.
			var seen map[attrKey]struct{}
			if len(v.Arr) > 1 {
				seen = make(map[attrKey]struct{}, len(v.Arr))
			}
			for _, el := range v.Arr {
				k := keyOf(el)
				if seen != nil {
					if _, dup := seen[k]; dup {
						continue
					}
					seen[k] = struct{}{}
				}
				counts[i][k]++
			}
		}
	}
	return buildFacets(fields, counts, topK), nil
}

// CorpusFacetCounts is FacetCounts over every live document, with no query:
// it reads the attribute index's value → document postings directly (their
// live entries only) instead of visiting documents, so its cost is
// proportional to the postings of the requested fields.
//
// Holds Engine.mu.RLock for the duration.
func (e *Engine) CorpusFacetCounts(fields []string, topK int) (Facets, error) {
	if err := checkFacetFields(fields); err != nil {
		return nil, err
	}
	e.mu.RLock()
	defer e.mu.RUnlock()

	x := e.attrIdx
	counts := make([]map[attrKey]int, len(fields))
	for i, f := range fields {
		counts[i] = make(map[attrKey]int)
		for k, list := range x.eq[f] {
			// A posting list is ascending; an array holding the same element
			// twice appended its ordinal twice in a row, so skipping an
			// ordinal equal to the previous one counts each document once.
			n, prev := 0, uint32(0)
			for j, o := range list {
				if j > 0 && o == prev {
					continue
				}
				prev = o
				if x.isLive(o) {
					n++
				}
			}
			if n > 0 {
				counts[i][k] = n
			}
		}
	}
	return buildFacets(fields, counts, topK), nil
}

func checkFacetFields(fields []string) error {
	for _, f := range fields {
		if f == "" {
			return ErrEmptyFacetField
		}
	}
	return nil
}

func buildFacets(fields []string, counts []map[attrKey]int, topK int) Facets {
	out := make(Facets, len(fields))
	for i, f := range fields {
		list := make([]FacetCount, 0, len(counts[i]))
		for k, n := range counts[i] {
			list = append(list, FacetCount{Value: AttrValue{Kind: k.kind, S: k.s, N: k.n}, Count: n})
		}
		sort.Slice(list, func(a, b int) bool {
			if list[a].Count != list[b].Count {
				return list[a].Count > list[b].Count
			}
			return lessAttrKey(keyOf(list[a].Value), keyOf(list[b].Value))
		})
		if topK > 0 && len(list) > topK {
			list = list[:topK]
		}
		out[f] = list
	}
	return out
}

// lessAttrKey orders values by kind, then string, then number — only to make
// ties in a facet listing deterministic.
func lessAttrKey(a, b attrKey) bool {
	if a.kind != b.kind {
		return a.kind < b.kind
	}
	if a.s != b.s {
		return a.s < b.s
	}
	return a.n < b.n
}
