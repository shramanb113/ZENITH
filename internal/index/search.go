package index

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// / SearchWithFilter is Search restricted to documents whose attributes satisfy
// pred (nil = no restriction). The predicate is applied to the lexical and
// vector candidate sets before rank fusion.
func (e *Engine) SearchWithFilter(ctx context.Context, query string, pred Predicate) ([]SearchResponse, error) {
	if pred == nil {
		return e.SearchFiltered(ctx, query, nil)
	}
	return e.SearchFiltered(ctx, query, &Filter{Pred: pred})
}

// embedHoldMax bounds how long a search holds the engine read lock waiting for
// the query embedding. A pending writer makes new readers queue behind it, so a
// slow embedder (a network call) must never be waited for with the lock held.
const embedHoldMax = 100 * time.Millisecond

type queryEmbedding struct {
	vec []float32
	err error
}

// lexicalPhase is everything a search does before it needs the query vector:
// analysis, the lexical candidate pass, attribute filtering and BM25.
// Engine.mu held for reading.
func (e *Engine) lexicalPhase(ctx context.Context, query string, f *Filter, phoneticWeight float64) (rawTokens []string, keywordScores map[uint64]float64, bm25Results []ranking.BM25Result, err error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, nil, err
	}
	var tokens []analysis.Token
	if qa, ok := e.analyzer.(analysis.QueryAnalyzer); ok {
		tokens = qa.AnalyzeQuery(query)
	} else {
		tokens = e.analyzer.Analyze(query)
	}
	rawTokens = make([]string, 0, len(tokens))
	for _, t := range tokens {
		rawTokens = append(rawTokens, t.Term)
	}

	// A blank/whitespace/stop-word-only query analyses to zero tokens.
	// Embedding "" still produces a valid vector that happens to be closest
	// to whatever the fallback/embedder considers "nothing", which returned
	// arbitrary top-N results instead of no results.
	if len(rawTokens) == 0 {
		return nil, nil, nil, nil
	}

	e.inverted.RLock()
	e.phonetics.RLock()
	keywordScores = e.lexicalPass(rawTokens, phoneticWeight)
	e.phonetics.RUnlock()
	e.inverted.RUnlock()
	e.filterCandidates(f.pred(), keywordScores)

	// BM25 over the literal query terms is needed by every fusion below and by
	// the weak-result check; compute it once.
	bm25Results = e.bm25.Query(rawTokens)
	return rawTokens, keywordScores, bm25Results, nil
}

// Weights overrides the per-list ranking weights for a single SearchFilteredWeighted
// call. A zero field uses the engine's configured default (Config.VectorWeight,
// Config.PhoneticWeight, Config.RRFConstant respectively) — mirroring the
// zero-means-default convention ranking.NewWeightedRRFRanker already uses for
// its own weight arguments. The keyword/lexical list weight is not
// overridable: it stays fixed at 1.0, same as every engine-construction call
// site (see cmd/server/main.go, pkg/zenith/zenith.go).
type Weights struct {
	Vector   float64
	Phonetic float64
	RRF      float64
}

// ErrInvalidWeights is returned for a Weights override holding a NaN,
// infinite or negative value.
var ErrInvalidWeights = errors.New("index: ranking weights must be finite and non-negative")

// Validate rejects a NaN, infinite or negative weight. Zero is valid and
// means "engine default". A NaN would otherwise poison every fused score (and
// the query-cache key), and a negative RRF k can make k+rank zero.
func (w Weights) Validate() error {
	for _, f := range [...]struct {
		name string
		v    float64
	}{{"vector", w.Vector}, {"phonetic", w.Phonetic}, {"rrf_k", w.RRF}} {
		if math.IsNaN(f.v) || math.IsInf(f.v, 0) || f.v < 0 {
			return fmt.Errorf("%w: %s = %v", ErrInvalidWeights, f.name, f.v)
		}
	}
	return nil
}

// scorer resolves the ranking.Scorer to use for a single query: the engine's
// shared scorer when w carries no override (the common, allocation-free
// case), or a fresh RRFRanker built from w's overrides layered onto the
// engine defaults otherwise. The engine's own e.scorer is never mutated, so
// concurrent searches with different overrides never interfere with each
// other or with un-overridden callers.
func (e *Engine) scorerFor(w Weights) ranking.Scorer {
	if w.Vector == 0 && w.RRF == 0 {
		return e.scorer
	}
	vec := w.Vector
	if vec == 0 {
		vec = e.config.VectorWeight
	}
	k := w.RRF
	if k == 0 {
		k = e.config.RRFConstant
	}
	return ranking.NewWeightedRRFRanker(k, e.config.MaxResults, 1.0, vec)
}

// phoneticWeight resolves the phonetic match weight to use for a single
// query: w's override, or the engine's configured default.
func (e *Engine) phoneticWeightFor(w Weights) float64 {
	if w.Phonetic != 0 {
		return w.Phonetic
	}
	return e.config.PhoneticWeight
}

// SearchFiltered is SearchWithFilter with the filter as data: when f carries a
// Spec, selective conditions are answered from the attribute index instead of
// by testing documents one by one. f may be nil. Equivalent to
// SearchFilteredWeighted with a zero Weights (every weight at its engine
// default).
func (e *Engine) SearchFiltered(ctx context.Context, query string, f *Filter) ([]SearchResponse, error) {
	return e.SearchFilteredWeighted(ctx, query, f, Weights{})
}

// SearchFilteredWeighted is SearchFiltered with per-query overrides for the
// ranking weights (see Weights) used only for this call — the engine's
// configured defaults, and every other concurrent search, are unaffected.
//
// This is the query-result cache's entry point (QUERYCACHE.md): a filter
// with no structured Spec (a raw Predicate closure — see Filter.spec) can't
// be hashed into a key, so it bypasses the cache entirely, computing fresh
// exactly as before caching existed. Every other call is keyed on
// (QueryCacheNamespace, writeGen, query, filter spec JSON, weights); a write
// bumps writeGen, making every previously-cached key for this engine
// unreachable without any active invalidation. Concurrent identical calls
// share one real computation via searchSF.
func (e *Engine) SearchFilteredWeighted(ctx context.Context, query string, f *Filter, w Weights) ([]SearchResponse, error) {
	if err := w.Validate(); err != nil {
		return nil, err
	}
	bypass := f != nil && f.pred() != nil && f.spec() == nil
	if bypass || e.cache == nil {
		entry, err := e.searchUncached(ctx, query, f, w)
		if err != nil {
			return nil, err
		}
		return entry.Results, nil
	}

	specJSON, err := json.Marshal(f.spec()) // nil *FilterSpec marshals to "null"
	if err != nil {
		// An unmarshalable spec (e.g. a non-finite range bound, or an array
		// value JSON can't represent) must never fall back to a shared
		// zero-value key: two different unmarshalable filters would then
		// collide on the same cache/singleflight key and one caller could
		// get another caller's results. Bypass the cache entirely instead,
		// the same way a raw-Predicate filter does.
		entry, err := e.searchUncached(ctx, query, f, w)
		if err != nil {
			return nil, err
		}
		return entry.Results, nil
	}
	e.mu.RLock()
	gen := e.writeGen
	e.mu.RUnlock()
	bucket := bucketKey(e.config.QueryCacheNamespace, gen, specJSON, w, e.processEpoch)
	key := fullCacheKey(bucket, query)

	if entry, tier, ok := e.cache.Get(ctx, key); ok {
		e.cacheObserver.ObserveQueryCacheHit(tier)
		return cloneResponses(entry.Results), nil
	}

	if e.config.QueryCacheSemanticThreshold > 0 {
		if qVec, ok := e.embedForSemanticScan(ctx, query); ok {
			if entry, ok := e.semanticScan(bucket, qVec); ok {
				e.cacheObserver.ObserveQueryCacheHit("semantic")
				return cloneResponses(entry.Results), nil
			}
		}
	}

	e.cacheObserver.ObserveQueryCacheMiss()
	// The shared computation runs on a ctx that can't be canceled by any one
	// caller: DoChan fans the same result out to every concurrent caller of
	// this key, so if it ran on the leader's own ctx, that caller hitting its
	// deadline or disconnecting would hand every other waiter for the same
	// popular query a spurious context.Canceled/DeadlineExceeded error too.
	// Each caller still honors its own ctx below by racing it against the
	// shared result instead of blocking on it unconditionally.
	ch := e.searchSF.DoChan(key, func() (any, error) {
		return e.searchUncached(context.WithoutCancel(ctx), query, f, w)
	})
	select {
	case res := <-ch:
		if res.Err != nil {
			return nil, res.Err
		}
		entry := res.Val.(cacheResult)
		entry.BucketKey = bucket
		e.cache.Set(ctx, key, entry)
		return cloneResponses(entry.Results), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// searchUncached is SearchFilteredWeighted's actual computation — the
// lexical pass, vector pass, rank fusion, and neural-expansion fallback,
// unchanged from before the query-result cache existed. Every caller goes
// through SearchFilteredWeighted above, never this directly.
//
// The query embedding runs concurrently with the lexical phase (the two are
// independent), so a hybrid search costs the longer of them instead of their
// sum. The embedding starts before the engine lock is taken and is waited for
// with the lock held only briefly (see embedHoldMax).
func (e *Engine) searchUncached(ctx context.Context, query string, f *Filter, w Weights) (cacheResult, error) {
	embedded := make(chan queryEmbedding, 1)
	go func() {
		v, err := embedding.EmbedQuery(ctx, e.embedder, query)
		embedded <- queryEmbedding{v, err}
	}()
	var early *queryEmbedding
	if e.serialEmbed { // test hook: embed first, as searches did before the overlap
		qe := <-embedded
		early = &qe
	}

	e.mu.RLock()
	locked := true
	defer func() {
		if locked {
			e.mu.RUnlock()
		}
	}()

	phoneticWeight := e.phoneticWeightFor(w)
	rawTokens, keywordScores, bm25Results, err := e.lexicalPhase(ctx, query, f, phoneticWeight)
	if err != nil {
		return cacheResult{}, err
	}
	if len(rawTokens) == 0 {
		return cacheResult{}, nil
	}

	var qe queryEmbedding
	if early != nil {
		qe = *early
	} else {
		select {
		case qe = <-embedded:
		default:
			timer := time.NewTimer(embedHoldMax)
			select {
			case qe = <-embedded:
				timer.Stop()
			case <-timer.C:
				// Slow embedder: let writers in while it finishes, then redo the
				// lexical phase against whatever the index has become.
				e.mu.RUnlock()
				locked = false
				qe = <-embedded
				e.mu.RLock()
				locked = true
				rawTokens, keywordScores, bm25Results, err = e.lexicalPhase(ctx, query, f, phoneticWeight)
				if err != nil {
					return cacheResult{}, err
				}
				if len(rawTokens) == 0 {
					return cacheResult{}, nil
				}
			}
		}
	}
	if qe.err != nil {
		slog.Warn("Search vectors degraded — embedder unreachable", "error", qe.err)
	}
	queryVec := normalizeVector(qe.vec)

	e.vectors.RLock()
	vectorScores, err := e.vectorPass(ctx, queryVec, f)
	e.vectors.RUnlock()
	if err != nil {
		return cacheResult{}, err
	}

	scorer := e.scorerFor(w)
	ranks, err := e.rankAndFuse(ctx, keywordScores, bm25Results, vectorScores, scorer)
	if err != nil {
		return cacheResult{}, err
	}

	// Neural expansion is meant to catch queries whose literal terms aren't
	// in the vocabulary (typos, unusual phrasing) by pulling in embedding
	// neighbors. Gating it on len(ranks)==0 alone means it almost never
	// fires in hybrid mode: vectorPass keeps every document with a positive
	// dot product against the query vector (roughly half the corpus for a
	// real embedder), so ranks is essentially never empty even when the
	// literal query terms match nothing. Instead, treat "no real BM25 hit for the
	// literal terms" as weak — that's independent of how permissive
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
		e.filterCandidates(f.pred(), expandedKeywords)

		for id, score := range keywordScores {
			expandedKeywords[id] += score
		}

		ranks, err = e.rankAndFuse(ctx, expandedKeywords, bm25Results, vectorScores, scorer)
		if err != nil {
			return cacheResult{}, err
		}
	}

	return cacheResult{Results: ranks, QueryVec: queryVec}, nil
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
func (e *Engine) lexicalPass(queryTokens []string, phoneticWeight float64) map[uint64]float64 {
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
			w := phoneticWeight
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
func (e *Engine) vectorPass(ctx context.Context, queryVec []float32, f *Filter) (map[uint64]float64, error) {
	scores := make(map[uint64]float64)
	if len(queryVec) == 0 {
		return scores, nil
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	pred := f.pred()

	// A selective filter with a structured description is answered from the
	// attribute index: the few matching documents are scored exactly, which is
	// both faster than a graph search that has to wander to find them and exact.
	if spec := f.spec(); pred != nil && spec != nil && e.attrIdx != nil {
		limit := max(2000, e.docCountLocked()/50)
		if est, ok := e.attrIdx.estimate(spec); ok && est <= limit {
			if ords, exact, ok := e.attrIdx.candidates(spec); ok {
				for _, id := range e.attrIdx.docIDs(ords) {
					if !exact && !pred(e.attrs[id]) {
						continue
					}
					if v := e.vecOf(id); v != nil {
						if s := ann.DotF32F16(queryVec, v); s > 0 {
							scores[id] = s
						}
					}
				}
				return scores, nil
			}
		}
	}

	banded := e.config.ANNThresholdBandPct > 0
	if e.ann != nil && e.ann.Len() > 0 && e.shouldUseANN(e.docCountLocked()) {
		start := time.Now()
		hits, ok := e.annSearch(queryVec, pred)
		// Only a successful ANN search is a real latency sample for this
		// path — a failed search (ok == false) falls through to the exact
		// scan below and recording it here would misrepresent the ANN
		// path's speed with a result it never actually produced.
		if banded && ok {
			e.annLatency.observe(elapsedMs(time.Since(start)))
		}
		if ok {
			for _, h := range hits {
				if h.Score > 0 {
					scores[h.ID] = h.Score
				}
			}
			return scores, nil
		}
	}

	start := time.Now()
	count := 0
	var canceled error
	e.eachVector(func(id uint64, v []uint16) bool {
		count++
		if count%2048 == 0 {
			if err := ctx.Err(); err != nil {
				canceled = err
				return false
			}
		}
		if pred != nil && !pred(e.attrs[id]) {
			return true
		}
		if s := ann.DotF32F16(queryVec, v); s > 0 {
			scores[id] = s
		}
		return true
	})
	if banded {
		e.exactLatency.observe(elapsedMs(time.Since(start)))
	}
	if canceled != nil {
		return nil, canceled
	}
	return scores, nil
}

// elapsedMs is d in fractional milliseconds. Sub-millisecond searches are
// common (small corpora, warm caches); time.Duration.Milliseconds truncates
// those to 0, which would make every fast search look equally fast to the
// adaptive-threshold comparison in shouldUseANN.
func elapsedMs(d time.Duration) float64 {
	return float64(d.Nanoseconds()) / 1e6
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
	ctx context.Context,
	kwScores map[uint64]float64,
	bm25Results []ranking.BM25Result,
	vScores map[uint64]float64,
	scorer ranking.Scorer,
) ([]SearchResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
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
		scored := scorer.Score(kwIDs, kwRank, nil, nil, e.origID)
		results := make([]SearchResponse, len(scored))
		for i, r := range scored {
			results[i] = SearchResponse{ID: r.ID, Score: r.Score}
		}
		return results, nil
	}

	// Hybrid mode: rank the lexical RRF list by kwRank (BM25-weighted, with
	// a coverage-based fallback for candidates BM25 doesn't score) fused
	// against the vector list.
	vcIDs := make([]uint64, 0, len(vScores))
	for id := range vScores {
		vcIDs = append(vcIDs, id)
	}

	// e.origID is safe here — Engine.mu.RLock() (Search) or Lock() (others) is held.
	scored := scorer.Score(kwIDs, kwRank, vcIDs, vScores, e.origID)

	results := make([]SearchResponse, len(scored))
	for i, r := range scored {
		results[i] = SearchResponse{ID: r.ID, Score: r.Score}
	}
	return results, nil
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

// attrSortKey is the per-result sort key SortByAttribute compares. A
// document missing the field, or holding an array value (kind is left at
// its zero value, which is never AttrString/AttrNumber/AttrBool), has
// has == false and always sorts last.
type attrSortKey struct {
	has  bool
	kind AttrKind
	s    string
	n    float64
}

func attrSortKeyFor(a Attrs, field string) attrSortKey {
	v, ok := a[field]
	if !ok || v.Kind == AttrArray {
		return attrSortKey{}
	}
	k := attrSortKey{has: true, kind: v.Kind}
	if v.Kind == AttrString {
		k.s = v.S
	} else {
		k.n = v.N
	}
	return k
}

// SortByAttribute reorders results by the named document attribute's value,
// replacing their existing order entirely — not a secondary tiebreak. The
// sort is stable, so documents tied on the attribute's value (including two
// documents both missing it, both holding an array value, or holding
// differently-typed values for it, which are incomparable) keep their
// relative order from results as passed in. A document missing the field,
// or holding an array value, always sorts after every document with a
// comparable scalar (string/number/bool) value, regardless of desc.
// field == "" is a no-op.
func (e *Engine) SortByAttribute(results []SearchResponse, field string, desc bool) {
	if field == "" || len(results) < 2 {
		return
	}
	keys := make([]attrSortKey, len(results))
	for i, r := range results {
		keys[i] = attrSortKeyFor(e.GetAttrs(r.ID), field)
	}
	idx := make([]int, len(results))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool {
		a, b := keys[idx[i]], keys[idx[j]]
		if a.has != b.has {
			return a.has
		}
		if !a.has || a.kind != b.kind {
			return false
		}
		var cmp int
		if a.kind == AttrString {
			cmp = strings.Compare(a.s, b.s)
		} else {
			switch {
			case a.n < b.n:
				cmp = -1
			case a.n > b.n:
				cmp = 1
			}
		}
		if desc {
			cmp = -cmp
		}
		return cmp < 0
	})
	sorted := make([]SearchResponse, len(results))
	for i, j := range idx {
		sorted[i] = results[j]
	}
	copy(results, sorted)
}
