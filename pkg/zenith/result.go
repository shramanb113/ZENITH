package zenith

// Result is a single search result returned by Search.
type Result struct {
	ID      string   // original document ID passed to Add
	Score   float64  // normalised to [0.0, 1.0]; never NaN, never Inf
	Chunks  []Chunk  // non-nil only for chunked documents (PDF pages)
	Signals *Signals // filled only when Search is called with Explain()
	// Attrs is the document's metadata as given to AddWithAttrs (nil if it
	// has none). Values come back as string, float64 (every number), bool,
	// or []any of those for an array attribute.
	Attrs Attrs
}

// FacetCount is how many matching documents carry one value of a faceted
// attribute. Value is a string, float64 or bool (an array attribute is
// counted per distinct element, never as a whole array).
type FacetCount struct {
	Value any
	Count int
}

// Facets maps each requested field to its value counts, highest count first.
// A field no matching document carries maps to an empty slice.
type Facets map[string][]FacetCount

// Signals is the raw, absolute evidence behind an Explain result.
type Signals struct {
	QueryTerms []string    // analysed base query terms (without synonyms)
	Lexical    float64     // raw BM25 of this document for the query
	Semantic   float64     // cosine similarity in [0, 1]; 0 without an embedding model
	Terms      []TermMatch // best match per query term found in this document
}

// TermMatch records how one analysed query term was found in the document.
// Dist is 0 for exact and synonym matches and the edit distance for fuzzy matches.
type TermMatch struct {
	Term    string
	Matched string
	Dist    int
	Synonym bool
}

// Chunk holds page and position metadata for a result from a chunked document.
type Chunk struct {
	Page            int
	Index           int
	BboxX, BboxY    float32
	BboxW, BboxH    float32
}
