package ranking

import (
	"math"
	"sort"
	"sync"
)

// TFIDFScorer implements the seroost TF-IDF ranking algorithm in Go.
//
// This is a direct port of the seroost search algorithm by tsoding:
//
//	TF(term, doc)     = count(term in doc) / total_terms_in_doc
//	IDF(term, corpus) = log( N / df(term) )
//	score(query, doc) = Σ TF(t,doc) * IDF(t)  for t in query_terms
//
// It sits alongside BM25Scorer and RRFRanker in the ranking package.
// All three implement the Scorer interface and can be swapped in the Engine.
//
// The key difference from BM25: TF-IDF has no length normalisation and no
// term saturation. It rewards documents where query terms appear frequently
// relative to the document's total length — simpler, faster, slightly less
// precise on long documents.
//
// Thread safety: Index()/Remove() must be serialised. Query()/Score() are
// safe for concurrent use.
type TFIDFScorer struct {
	mu sync.RWMutex

	// Per-document state
	termFreqs  map[uint64]map[string]int // internalID → term → count
	docLengths map[uint64]int            // internalID → total term count

	// Corpus-level state
	docFreq   map[string]int // term → number of docs containing it
	totalDocs int
}

// NewTFIDFScorer creates an empty TFIDFScorer.
func NewTFIDFScorer() *TFIDFScorer {
	return &TFIDFScorer{
		termFreqs:  make(map[uint64]map[string]int),
		docLengths: make(map[uint64]int),
		docFreq:    make(map[string]int),
	}
}

// Index records term frequencies for docID.
// tokens must already be stemmed/normalised — use analysis.TokenizeString.
// Re-indexing an existing docID cleanly replaces its old stats.
func (s *TFIDFScorer) Index(docID uint64, tokens []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove old stats if re-indexing
	if old, exists := s.termFreqs[docID]; exists {
		for term := range old {
			s.docFreq[term]--
			if s.docFreq[term] <= 0 {
				delete(s.docFreq, term)
			}
		}
		s.totalDocs--
	}

	tf := make(map[string]int, len(tokens))
	for _, tok := range tokens {
		tf[tok]++
	}

	s.termFreqs[docID] = tf
	s.docLengths[docID] = len(tokens)
	s.totalDocs++

	for term := range tf {
		s.docFreq[term]++
	}
}

// Remove deletes a document from the TF-IDF index.
func (s *TFIDFScorer) Remove(docID uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	old, exists := s.termFreqs[docID]
	if !exists {
		return
	}
	for term := range old {
		s.docFreq[term]--
		if s.docFreq[term] <= 0 {
			delete(s.docFreq, term)
		}
	}
	s.totalDocs--
	delete(s.termFreqs, docID)
	delete(s.docLengths, docID)
}

// State returns a snapshot of TF-IDF corpus state for serialisation.
func (s *TFIDFScorer) State() (map[uint64]int, map[uint64]map[string]int, map[string]int, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.docLengths, s.termFreqs, s.docFreq, s.totalDocs
}

// LoadState restores TF-IDF corpus state after deserialisation.
func (s *TFIDFScorer) LoadState(docLengths map[uint64]int, termFreqs map[uint64]map[string]int, docFreq map[string]int, totalDocs int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docLengths = docLengths
	s.termFreqs = termFreqs
	s.docFreq = docFreq
	s.totalDocs = totalDocs
}

// tf computes the term frequency of term in docID.
// TF = count(term in doc) / total_terms_in_doc — exactly as seroost computes it.
func (s *TFIDFScorer) tf(term string, docID uint64) float64 {
	dl := s.docLengths[docID]
	if dl == 0 {
		return 0
	}
	return float64(s.termFreqs[docID][term]) / float64(dl)
}

// idf computes the inverse document frequency for term.
// IDF = log( N / df(term) ) — exactly as seroost computes it.
// Returns 0 if term is not in any document.
func (s *TFIDFScorer) idf(term string) float64 {
	df := s.docFreq[term]
	if df == 0 || s.totalDocs == 0 {
		return 0
	}
	return math.Log(float64(s.totalDocs) / float64(df))
}

// TFIDFResult is a scored document from a TF-IDF query.
type TFIDFResult struct {
	DocID uint64
	Score float64
}

// Query scores all indexed documents against queryTerms.
// queryTerms must be stemmed the same way as at index time.
// Results are sorted descending by score; zero-score docs are excluded.
// This is the direct equivalent of seroost's search function.
func (s *TFIDFScorer) Query(queryTerms []string) []TFIDFResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(queryTerms) == 0 || s.totalDocs == 0 {
		return nil
	}

	results := make([]TFIDFResult, 0, len(s.termFreqs))
	for docID := range s.termFreqs {
		var score float64
		for _, term := range queryTerms {
			score += s.tf(term, docID) * s.idf(term)
		}
		if score > 0 {
			results = append(results, TFIDFResult{DocID: docID, Score: score})
		}
	}

	// Sort descending — same order seroost produces
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// Score implements the ranking.Scorer interface.
// Blends TF-IDF lexical scores with vector scores using weighted normalisation:
//
//	final = 0.6 * tfidf_normalised + 0.4 * vector_normalised
//
// This mirrors BM25Scorer.Score so both can be swapped transparently.
func (s *TFIDFScorer) Score(
	keywordIDs []uint64,
	keywordScores map[uint64]float64,
	vectorIDs []uint64,
	vectorScores map[uint64]float64,
	idMapping IDLookup,
) []ScoredResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	seen := make(map[uint64]struct{})
	for _, id := range keywordIDs {
		seen[id] = struct{}{}
	}
	for _, id := range vectorIDs {
		seen[id] = struct{}{}
	}

	var maxKW, maxVec float64
	for _, id := range keywordIDs {
		if keywordScores[id] > maxKW {
			maxKW = keywordScores[id]
		}
	}
	for _, id := range vectorIDs {
		if vectorScores[id] > maxVec {
			maxVec = vectorScores[id]
		}
	}

	results := make([]ScoredResult, 0, len(seen))
	for id := range seen {
		var normKW, normVec float64
		if maxKW > 0 {
			normKW = keywordScores[id] / maxKW
		}
		if maxVec > 0 {
			normVec = vectorScores[id] / maxVec
		}
		combined := 0.6*normKW + 0.4*normVec
		if combined > 0 {
			results = append(results, ScoredResult{
				ID:    idMapping(id),
				Score: combined,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].ID < results[j].ID
	})

	if len(results) > 10 {
		results = results[:10]
	}
	return results
}

// DocCount returns the number of indexed documents.
func (s *TFIDFScorer) DocCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.totalDocs
}
