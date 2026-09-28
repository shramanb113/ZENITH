package ranking

type ScoredResult struct {
	ID    string
	Score float64
}

type Candidate struct {
	ID    uint64
	Score float64
}

type Scorer interface {
    Score(keywordIDs []uint64, keywordScores map[uint64]float64, vectorIDs []uint64, vectorScores map[uint64]float64, idMapping IDLookup) []ScoredResult
}

// IDLookup resolves an internal document ID to the caller-facing document ID.
// It is a function rather than a map so the engine can resolve IDs that live in
// memory-mapped segments without materialising a map of every document.
type IDLookup func(id uint64) string

// MapLookup adapts a map to an IDLookup; IDs missing from the map resolve to "".
func MapLookup(m map[uint64]string) IDLookup {
	return func(id uint64) string { return m[id] }
}
