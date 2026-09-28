package index

import (
	"context"
	"log/slog"
	"sort"
	"unicode/utf8"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// SearchWithFilter is Search restricted to documents whose attributes satisfy
// pred (nil = no restriction). The predicate is applied to the lexical and
// vector candidate sets before rank fusion.
func (e *Engine) SearchWithFilter(ctx context.Context, query string, pred Predicate) ([]SearchResponse, error) {
	queryVec, embErr := embedding.EmbedQuery(ctx, e.embedder, query)
	if embErr != nil {
		slog.Warn("Search vectors degraded — embedder unreachable", "error", embErr)
	}
	queryVec = normalizeVector(queryVec)

	e.mu.RLock()
	defer e.mu.RUnlock()

	var tokens []analysis.Token
	if qa, ok := e.analyzer.(analysis.QueryAnalyzer); ok {
		tokens = qa.AnalyzeQuery(query)
	} else {
		tokens = e.analyzer.Analyze(query)
	}
	rawTokens := make([]string, 0, len(tokens))
	for _, t := range tokens {
		rawTokens = append(rawTokens, t.Term)
	}

	// A blank/whitespace/stop-word-only query analyses to zero tokens.
	// Embedding "" still produces a valid vector that happens to be closest
	// to whatever the fallback/embedder considers "nothing", which returned
	// arbitrary top-N results instead of no results.
	if len(rawTokens) == 0 {
		return nil, nil
	}

	e.inverted.RLock()
	e.phonetics.RLock()
	keywordScores := e.lexicalPass(rawTokens)
	e.phonetics.RUnlock()
	e.inverted.RUnlock()
	e.filterCandidates(pred, keywordScores)

	e.vectors.RLock()
	vectorScores := e.vectorPass(queryVec, pred)
	e.vectors.RUnlock()

	// BM25 over the literal query terms is needed by every fusion below and by
	// the weak-result check; compute it once.
	bm25Results := e.bm25.Query(rawTokens)

	ranks := e.rankAndFuse(keywordScores, bm25Results, vectorScores)

	// Neural expansion is meant to catch queries whose literal terms aren't
	// in the vocabulary (typos, unusual phrasing) by pulling in embedding
	// neighbors. Gating it on len(ranks)==0 alone means it almost never
	// fires in hybrid mode: vectorPass keeps every document with a positive
	// dot product against the query vector (roughly half the corpus for a
	// real embedder), so ranks is essentially never empty even when the
	// literal query terms match nothing. Instead, treat "no real BM25 hit
	// for the literal terms" as weak — that's independent of how permissive
	// the vector pass was.
	weakResults := len(ranks) == 0
	if !weakResults && e.config.WordVectors && len(bm25Results) == 0 {
		weakResults = true
	}

	if weakResults && e.config.WordVectors {
		expandedTokens := e.expandTokens(rawTokens)

		e.inverted.RLock()
		expandedKeywords := e.neuralExpand(expandedTokens)
		e.inverted.RUnlock()
		e.filterCandidates(pred, expandedKeywords)

		for id, score := range keywordScores {
			expandedKeywords[id] += score
		}

		ranks = e.rankAndFuse(expandedKeywords, bm25Results, vectorScores)
	}

	return ranks, nil
}

func (e *Engine) expandTokens(rawTokens []string) []string {
	var expanded []string
	for _, token := range rawTokens {
		if len(token) < 3 {
			continue
		}
		neighbors := e.getSemanticNeighbors(token, 5, 0.70)
		for _, n := range neighbors {
			neighborTokens := e.analyzer.Analyze(n)
			if len(neighborTokens) > 0 {
				expanded = append(expanded, neighborTokens[0].Term)
			}
		}
	}
	return expanded
}

// lexicalPass scores candidate documents by edge-n-gram coverage, phonetic
// code and fuzzy (edit-distance) matches of the query tokens. Posting lists
// are read from the delta and every segment (see eachFragDoc).
func (e *Engine) lexicalPass(queryTokens []string) map[uint64]float64 {
	keywordScores := make(map[uint64]float64)
	cap := e.prefixCap()

	for _, token := range queryTokens {
		Q := len(token)

		var frags []string
		if Q >= 3 {
			frags = generateEdgeNgrams(token)
		} else {
			frags = []string{token}
		}

		for _, frag := range frags {
			// A short prefix shared by a large part of the corpus adds ~everything
			// as a candidate and tells us nothing (see Config.PrefixFragmentCap).
			// The full-token fragment is exempt: it finds every document that
			// actually contains the term.
			if cap > 0 && len(frag) < Q && e.fragCount(frag) > cap {
				continue
			}
			w := (float64(len(frag)) / float64(Q)) * 100.0
			e.eachFragDoc(frag, func(id uint64) { keywordScores[id] += w })
		}

		if phon := analysis.Soundex(token); phon != "" {
			w := e.config.PhoneticWeight
			e.eachPhonDoc(phon, func(id uint64) { keywordScores[id] += w })
		}

		if Q >= 2 {
			for _, match := range e.fuzzyMatches(token) {
				if match.Distance == 0 {
					continue
				}
				w := 60.0 / float64(match.Distance)
				e.eachFragDoc(match.Word, func(id uint64) { keywordScores[id] += w })
			}
		}
	}
	return keywordScores
}

// fuzzyMatches returns vocabulary terms within FuzzyMaxDist edits of token.
// It walks the FST with a Levenshtein automaton when the FST is built and the
// distance is supported (cost ~ matches, not vocabulary size); otherwise it
// falls back to the BK-tree, which returns the identical set and distances.
func (e *Engine) fuzzyMatches(token string) []analysis.FuzzyMatch {
	d := e.fuzzyDist(token)
	if d <= 0 {
		return nil
	}
	if m, ok, err := e.fst.FuzzySearch(token, d); err == nil && ok {
		return m
	}
	return e.bkTree().Search(token, d)
}

// fuzzyDist is the edit distance allowed when correcting token. With
// Config.FuzzyByLength (the default) it grows with the word: nothing within
// two edits of a 2–3 letter word is a plausible misspelling of it — such a word
// is within 2 edits of hundreds of unrelated vocabulary terms — so short words
// must match exactly, mid-length words may be off by one, and only longer
// words get the full FuzzyMaxDist. This is the "AUTO" fuzziness rule of
// Lucene/Elasticsearch.
func (e *Engine) fuzzyDist(token string) int {
	d := e.config.FuzzyMaxDist
	if !e.config.FuzzyByLength {
		return d
	}
	switch n := utf8.RuneCountInString(token); {
	case n <= 3:
		return 0
	case n <= 5:
		return min(d, 1)
	default:
		return d
	}
}

// bkTree returns the BK-tree fuzzy fallback, building it from the live
// vocabulary on first use. Safe under Engine.mu.RLock: builds are serialised
// by bkMu, and vocabulary changes (which need the write lock) cannot overlap.
func (e *Engine) bkTree() *analysis.BKTree {
	e.bkMu.Lock()
	defer e.bkMu.Unlock()
	if e.bk == nil {
		t := analysis.NewBKTree()
		for _, term := range e.liveTerms() {
			t.Add(term)
		}
		e.bk = t
	}
	return e.bk
}

// vectorPass scores documents by dot product with the query vector.
// Negative dot products are clamped to 0 — a document pointing away from
// the query has zero semantic relevance, not negative relevance.
//
// Small corpora (or highly selective filters) are scored exactly by scanning;
// once an ANN graph exists and the filter is broad, the graph returns the
// top annK candidates instead of scoring every document. pred (may be nil)
// is applied here, before fusion.
func (e *Engine) vectorPass(queryVec []float32, pred Predicate) map[uint64]float64 {
	scores := make(map[uint64]float64)
	if len(queryVec) == 0 {
		return scores
	}

	if e.ann != nil && e.ann.Len() > 0 {
		if hits, ok := e.annSearch(queryVec, pred); ok {
			for _, h := range hits {
				if h.Score > 0 {
					scores[h.ID] = h.Score
				}
			}
			return scores
		}
	}

	e.eachVector(func(id uint64, v []uint16) {
		if pred != nil && !pred(e.attrs[id]) {
			return
		}
		if s := ann.DotF32F16(queryVec, v); s > 0 {
			scores[id] = s
		}
	})
	return scores
}

func (e *Engine) neuralExpand(expandedTokens []string) map[uint64]float64 {
	keywordScores := make(map[uint64]float64)

	for _, neighbor := range expandedTokens {
		targets := make(map[uint64]bool)
		e.eachFragDoc(neighbor, func(id uint64) { targets[id] = true })
		if runes := []rune(neighbor); len(runes) > 3 {
			e.eachFragDoc(string(runes[:3]), func(id uint64) { targets[id] = true })
		}
		for id := range targets {
			keywordScores[id] += 20000.0
		}
	}
	return keywordScores
}

func (e *Engine) rankAndFuse(
	kwScores map[uint64]float64,
	bm25Results []ranking.BM25Result,
	vScores map[uint64]float64,
) []SearchResponse {
	bm25ByID := make(map[uint64]float64, len(bm25Results))
	for _, r := range bm25Results {
		bm25ByID[r.DocID] = r.Score
	}
	kwRank := buildKwRank(kwScores, bm25ByID)
	kwIDs := make([]uint64, 0, len(kwRank))
	for id := range kwRank {
		kwIDs = append(kwIDs, id)
	}

	// BM25-only mode: no vector scores are present.
	if len(vScores) == 0 {
		scored := e.scorer.Score(kwIDs, kwRank, nil, nil, e.origID)
		results := make([]SearchResponse, len(scored))
		for i, r := range scored {
			results[i] = SearchResponse{ID: r.ID, Score: r.Score}
		}
		return results
	}

	// Hybrid mode: rank the lexical RRF list by kwRank (BM25-weighted, with
	// a coverage-based fallback for candidates BM25 doesn't score) fused
	// against the vector list.
	vcIDs := make([]uint64, 0, len(vScores))
	for id := range vScores {
		vcIDs = append(vcIDs, id)
	}

	// e.origID is safe here — Engine.mu.RLock() (Search) or Lock() (others) is held.
	scored := e.scorer.Score(kwIDs, kwRank, vcIDs, vScores, e.origID)

	results := make([]SearchResponse, len(scored))
	for i, r := range scored {
		results[i] = SearchResponse{ID: r.ID, Score: r.Score}
	}
	return results
}

// getSemanticNeighbors returns the topN most similar words to token by dot
// product, sorted descending by similarity. Previously truncated without
// sorting — nondeterministic under Go's randomised map iteration.
func (e *Engine) getSemanticNeighbors(token string, topN int, threshold float32) []string {
	e.vectors.RLock()
	defer e.vectors.RUnlock()

	tokenBits, ok := e.wordVec(token)
	if !ok {
		return nil
	}

	tokenVec := Float16ToFloats(tokenBits)
	var candidates []neighborCandidate
	e.eachWordVector(func(word string, v []uint16) {
		if word == token {
			return
		}
		s := float32(ranking.DotProduct(tokenVec, Float16ToFloats(v)))
		if s >= threshold {
			candidates = append(candidates, neighborCandidate{word: word, score: s})
		}
	})

	// Sort descending by similarity so topN is deterministic.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	if topN > 0 && len(candidates) > topN {
		candidates = candidates[:topN]
	}

	out := make([]string, len(candidates))
	for i, c := range candidates {
		out[i] = c.word
	}
	return out
}
