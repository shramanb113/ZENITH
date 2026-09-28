package ranking

import (
	"math"
	"sort"
	"sync"
)

// BM25Backing supplies corpus statistics and postings for documents that live
// in immutable, memory-mapped segments instead of the scorer's own maps. The
// scorer combines it with the documents it holds itself (recent additions), so
// scores are identical to a single in-memory index over all of them. Every
// method must reflect only *live* backing documents (deleted ones excluded).
type BM25Backing interface {
	// Totals returns the number of live backing documents and the sum of their lengths.
	Totals() (docs, totalLen int)
	// DocFreq returns how many live backing documents contain term.
	DocFreq(term string) int
	// EachPosting calls fn once per live backing document containing term.
	EachPosting(term string, fn func(docID uint64, tf, docLen int))
	// TermFreq returns a live backing document's frequency of term and its length.
	TermFreq(docID uint64, term string) (tf, docLen int, ok bool)
}

// BM25Scorer implements the BM25 (Okapi BM25) ranking function.
//
// BM25 improves on TF-IDF with two key parameters:
//   - k1 (term saturation): controls how much repeated terms matter.
//     After a point, more occurrences stop helping. Typical: 1.2–2.0.
//   - b (length normalisation): penalises long documents. 0 = no normalisation,
//     1 = full normalisation. Typical: 0.75.
//
// Formula per term t in document d:
//
//	IDF(t) = log( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
//	TF_norm(t,d) = freq(t,d) * (k1+1) / (freq(t,d) + k1*(1 - b + b*(|d|/avgdl)))
//	BM25(d,Q) = Σ IDF(t) * TF_norm(t,d)  for t in Q
//
// It implements the Scorer interface so it can be used interchangeably with
// RRFRanker inside the Engine.
//
// Thread safety: Index() and Remove() mutate state — serialise these.
// Query() and Score() only read — safe for concurrent use after indexing.
type BM25Scorer struct {
	mu sync.RWMutex

	k1 float64 // term saturation parameter (default 1.2)
	b  float64 // length normalisation parameter (default 0.75)

	// Per-document state
	docLengths map[uint64]int            // internalID → term count
	termFreqs  map[uint64]map[string]int // internalID → term → frequency

	// Inverted posting lists: term → IDs of documents containing it.
	// Lets Query touch only documents containing at least one query term —
	// O(hits) instead of the O(N) full-corpus scan that made every query
	// evaluate BM25 against all 100k documents (~280ms p50 at 100k docs).
	// Rebuilt from termFreqs on LoadState, so the serialized format is
	// unchanged.
	postings map[string][]uint64

	// Corpus-level state
	docFreq   map[string]int // term → number of documents containing it
	totalDocs int
	totalLen  int // sum of all document lengths (for avgdl)

	back BM25Backing // optional; nil = everything is in the maps above
}

// SetBacking attaches (or, with nil, detaches) immutable-segment statistics.
func (s *BM25Scorer) SetBacking(b BM25Backing) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.back = b
}

// Reset drops every document the scorer holds itself (not the backing).
func (s *BM25Scorer) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docLengths = make(map[uint64]int)
	s.termFreqs = make(map[uint64]map[string]int)
	s.postings = make(map[string][]uint64)
	s.docFreq = make(map[string]int)
	s.totalDocs, s.totalLen = 0, 0
}

// corpus returns the combined live document count and total length.
// Caller holds s.mu.
func (s *BM25Scorer) corpus() (docs, totalLen int) {
	docs, totalLen = s.totalDocs, s.totalLen
	if s.back != nil {
		d, l := s.back.Totals()
		docs += d
		totalLen += l
	}
	return docs, totalLen
}

// dfOf returns the combined document frequency of term. Caller holds s.mu.
func (s *BM25Scorer) dfOf(term string) int {
	df := s.docFreq[term]
	if s.back != nil {
		df += s.back.DocFreq(term)
	}
	return df
}

// BM25Params allows tuning k1 and b. Zero value uses defaults.
type BM25Params struct {
	K1 float64 // default 1.2
	B  float64 // default 0.75
}

// NewBM25Scorer creates a BM25Scorer with the given parameters.
// Pass zero BM25Params{} to use defaults (k1=1.2, b=0.75).
func NewBM25Scorer(p BM25Params) *BM25Scorer {
	k1 := p.K1
	if k1 == 0 {
		k1 = 1.2
	}
	b := p.B
	if b == 0 {
		b = 0.75
	}
	return &BM25Scorer{
		k1:         k1,
		b:          b,
		docLengths: make(map[uint64]int),
		termFreqs:  make(map[uint64]map[string]int),
		postings:   make(map[string][]uint64),
		docFreq:    make(map[string]int),
	}
}

// Index records term frequencies for a document.
// tokens must already be stemmed/normalised — use analysis.TokenizeString.
// If the document was previously indexed it is cleanly re-indexed.
func (s *BM25Scorer) Index(docID uint64, tokens []string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Remove old stats if re-indexing
	if old, exists := s.termFreqs[docID]; exists {
		for term := range old {
			s.docFreq[term]--
			if s.docFreq[term] <= 0 {
				delete(s.docFreq, term)
			}
			s.removePosting(term, docID)
		}
		s.totalLen -= s.docLengths[docID]
		s.totalDocs--
	}

	tf := make(map[string]int, len(tokens))
	for _, tok := range tokens {
		tf[tok]++
	}

	s.termFreqs[docID] = tf
	s.docLengths[docID] = len(tokens)
	s.totalLen += len(tokens)
	s.totalDocs++

	for term := range tf {
		s.docFreq[term]++
		s.postings[term] = append(s.postings[term], docID)
	}
}

// removePosting deletes docID from term's posting list.
// Caller must hold s.mu.
func (s *BM25Scorer) removePosting(term string, docID uint64) {
	list := s.postings[term]
	for i, id := range list {
		if id == docID {
			list[i] = list[len(list)-1]
			list = list[:len(list)-1]
			break
		}
	}
	if len(list) == 0 {
		delete(s.postings, term)
	} else {
		s.postings[term] = list
	}
}

// Remove deletes a document from the BM25 index.
func (s *BM25Scorer) Remove(docID uint64) {
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
		s.removePosting(term, docID)
	}
	s.totalLen -= s.docLengths[docID]
	s.totalDocs--
	delete(s.termFreqs, docID)
	delete(s.docLengths, docID)
}

// State returns a snapshot of BM25 corpus state for serialisation.
func (s *BM25Scorer) State() (map[uint64]int, map[uint64]map[string]int, map[string]int, int, int) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.docLengths, s.termFreqs, s.docFreq, s.totalDocs, s.totalLen
}

// LoadState restores BM25 corpus state after deserialisation.
// Posting lists are not part of the serialized format — they are derived
// state, rebuilt here from termFreqs in one pass.
func (s *BM25Scorer) LoadState(docLengths map[uint64]int, termFreqs map[uint64]map[string]int, docFreq map[string]int, totalDocs, totalLen int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.docLengths = docLengths
	s.termFreqs = termFreqs
	s.docFreq = docFreq
	s.totalDocs = totalDocs
	s.totalLen = totalLen

	s.postings = make(map[string][]uint64, len(docFreq))
	for docID, tf := range termFreqs {
		for term := range tf {
			s.postings[term] = append(s.postings[term], docID)
		}
	}
}

// avgdl returns the average document length across the corpus.
func (s *BM25Scorer) avgdl() float64 {
	docs, total := s.corpus()
	if docs == 0 {
		return 0
	}
	return float64(total) / float64(docs)
}

// idf computes the IDF component for a term.
// Uses the Robertson-Walker IDF variant with +1 smoothing to avoid
// negative values for terms appearing in more than half the corpus.
func (s *BM25Scorer) idf(term string) float64 {
	n, _ := s.corpus()
	df := float64(s.dfOf(term))
	return math.Log((float64(n)-df+0.5)/(df+0.5) + 1)
}

// scoreDocParams computes the BM25 score for a single document with explicit
// k1/b, so parameter variants can be
// evaluated against a built index without mutating the scorer.
func (s *BM25Scorer) scoreDocParams(docID uint64, queryTerms []string, k1, b float64) float64 {
	tf, held := s.termFreqs[docID]
	dl := float64(s.docLengths[docID])
	avgdl := s.avgdl()

	var score float64
	for _, term := range queryTerms {
		var freq float64
		if held {
			freq = float64(tf[term])
		} else if s.back != nil {
			if f, l, ok := s.back.TermFreq(docID, term); ok {
				freq, dl = float64(f), float64(l)
			}
		}
		if freq == 0 {
			continue
		}
		idf := s.idf(term)
		// BM25 TF normalisation
		tfNorm := freq * (k1 + 1) / (freq + k1*(1-b+b*(dl/avgdl)))
		score += idf * tfNorm
	}
	return score
}

// ScoreDocs computes BM25 scores for the listed documents only, with explicit
// k1/b parameters. O(len(docIDs) × len(queryTerms)) — used for offline
// parameter evaluation where scoring the whole corpus per variant would be
// prohibitive. Documents scoring 0 are omitted from the result.
func (s *BM25Scorer) ScoreDocs(docIDs []uint64, queryTerms []string, k1, b float64) map[uint64]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make(map[uint64]float64, len(docIDs))
	for _, id := range docIDs {
		if sc := s.scoreDocParams(id, queryTerms, k1, b); sc > 0 {
			out[id] = sc
		}
	}
	return out
}

// BM25Result is a scored document from a BM25 query.
type BM25Result struct {
	DocID uint64
	Score float64
}

// Query scores documents containing at least one query term and returns
// results sorted descending by score. Documents with score 0 are excluded.
// queryTerms must be stemmed/normalised the same way as at index time.
//
// Cost is O(Σ posting-list lengths of the query's terms) — only documents
// containing a query term are evaluated. The previous implementation scanned
// every indexed document per query (O(N)); at 100k docs that was ~700M BM25
// evaluations across the MS MARCO query set and dominated query latency.
// Scores are identical: documents without any query term always scored 0
// and were discarded anyway. Repeated query terms contribute once per
// occurrence, matching the original term-loop semantics.
func (s *BM25Scorer) Query(queryTerms []string) []BM25Result {
	s.mu.RLock()
	defer s.mu.RUnlock()

	nDocs, _ := s.corpus()
	if len(queryTerms) == 0 || nDocs == 0 {
		return nil
	}

	// Collapse duplicate terms but preserve their multiplicity so repeated
	// terms score exactly as the original per-occurrence loop did.
	termCount := make(map[string]int, len(queryTerms))
	for _, term := range queryTerms {
		termCount[term]++
	}

	avgdl := s.avgdl()
	scores := make(map[uint64]float64)
	for term, count := range termCount {
		idf := s.idf(term) * float64(count)
		for _, docID := range s.postings[term] {
			freq := float64(s.termFreqs[docID][term])
			dl := float64(s.docLengths[docID])
			tfNorm := freq * (s.k1 + 1) / (freq + s.k1*(1-s.b+s.b*(dl/avgdl)))
			scores[docID] += idf * tfNorm
		}
		if s.back != nil {
			s.back.EachPosting(term, func(docID uint64, tf, docLen int) {
				freq := float64(tf)
				dl := float64(docLen)
				tfNorm := freq * (s.k1 + 1) / (freq + s.k1*(1-s.b+s.b*(dl/avgdl)))
				scores[docID] += idf * tfNorm
			})
		}
	}

	results := make([]BM25Result, 0, len(scores))
	for docID, sc := range scores {
		if sc > 0 {
			results = append(results, BM25Result{DocID: docID, Score: sc})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results
}

// Score implements the ranking.Scorer interface so BM25Scorer can be used
// as a drop-in replacement for RRFRanker inside the Engine.
//
// When used as a Scorer, it blends its own BM25 keyword scores with the
// provided vectorScores using a simple weighted sum:
//
//	final = 0.6 * bm25_normalised + 0.4 * vector_normalised
//
// This preserves the hybrid nature of the Engine while using BM25 for the
// lexical component instead of the raw n-gram scoring.
func (s *BM25Scorer) Score(
	keywordIDs []uint64,
	keywordScores map[uint64]float64,
	vectorIDs []uint64,
	vectorScores map[uint64]float64,
	idMapping IDLookup,
) []ScoredResult {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Collect all candidate document IDs
	seen := make(map[uint64]struct{})
	for _, id := range keywordIDs {
		seen[id] = struct{}{}
	}
	for _, id := range vectorIDs {
		seen[id] = struct{}{}
	}

	// Normalise BM25 keyword scores to [0,1]
	var maxKW float64
	for _, id := range keywordIDs {
		if keywordScores[id] > maxKW {
			maxKW = keywordScores[id]
		}
	}

	// Normalise vector scores to [0,1]
	var maxVec float64
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
func (s *BM25Scorer) DocCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	n, _ := s.corpus()
	return n
}
