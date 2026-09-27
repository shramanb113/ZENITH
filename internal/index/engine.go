package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"log/slog"
	"maps"
	"os"
	pathutil "path/filepath" // aliased: Save/Load use "filepath" as a parameter name
	"sort"
	"sync"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// SearchResponse holds a single search result.
type SearchResponse struct {
	ID    string
	Score float64
}

// BatchDoc is a single entry for AddBatch.
type BatchDoc struct {
	ID     string
	Text   string
	Vector []float32
}

// TermStore is implemented by storage backends that maintain a term vocabulary.
type TermStore interface {
	AddTerms([]string)
}

// DocumentJournal durably records document mutations before they touch the
// in-memory index. Satisfied by *storage.Engine — its Put/Delete signatures
// match exactly. Set via SetDocumentJournal; nil means no journaling.
type DocumentJournal interface {
	Put(ctx context.Context, key, value []byte) error
	Delete(ctx context.Context, key []byte) error
}

// Engine is the central orchestrator — it owns all sub-indexes and the
// scoring pipeline.
//
// Concurrency model: Engine.mu is the top-level gate.
//   - Add, AddWithVector, AddBatch, Remove, Load, RebuildFST: take mu.Lock()
//   - Search, Save: take mu.RLock()
//
// Sub-index locks (inverted.mu, vectors.mu, phonetics.mu, bm25.mu) are kept
// as defence-in-depth but are no longer the primary concurrency boundary.
// Lock ordering is always: Engine.mu → sub-index lock. Never reversed.
type Engine struct {
	mu sync.RWMutex // primary concurrency gate — see comment above

	// saveMu serialises Save calls so two concurrent Saves can't both write
	// to the same fixed ".tmp" path. Save only takes mu.RLock() (to allow
	// concurrent Search), so this is the only thing preventing that race.
	saveMu sync.Mutex

	config    *config.Config
	inverted  *InvertedIndex
	vectors   *VectorStore
	phonetics *PhoneticIndex
	bkTree    *analysis.BKTree

	embedder embedding.Embedder
	scorer   ranking.Scorer
	analyzer analysis.Analyzer

	bm25  *ranking.BM25Scorer
	tfidf *ranking.TFIDFScorer

	idMapping map[uint64]string
	docText   map[uint64]string // original full text per document, for GetText

	fst      *analysis.FSTDictionary
	fstSize  int  // informational only — last rebuild's term count
	fstDirty bool // true when the vocabulary has changed since the last FST build
	fstPath  string

	termStore TermStore
	journal   DocumentJournal
}

// NewEngine constructs a fully initialised Engine.
func NewEngine(cfg *config.Config, emb embedding.Embedder, scr ranking.Scorer, ana analysis.Analyzer) *Engine {
	return &Engine{
		config:    cfg,
		inverted:  NewInvertedIndex(),
		vectors:   NewVectorStore(),
		phonetics: NewPhoneticIndex(),
		bkTree:    analysis.NewBKTree(),
		embedder:  emb,
		scorer:    scr,
		analyzer:  ana,
		idMapping: make(map[uint64]string),
		docText:   make(map[uint64]string),
		bm25:      ranking.NewBM25Scorer(ranking.BM25Params{}),
		tfidf:     ranking.NewTFIDFScorer(),
		fst:       analysis.NewFSTDictionary(),
	}
}

func (e *Engine) SetTermStore(s TermStore)             { e.termStore = s }
func (e *Engine) SetFSTPath(path string)               { e.fstPath = path }
func (e *Engine) SetDocumentJournal(j DocumentJournal) { e.journal = j }

// RebuildFST rebuilds the FST from the current global vocabulary.
// Takes Engine.mu.Lock() — safe to call from outside the engine.
func (e *Engine) RebuildFST() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.rebuildFSTLocked()
}

// rebuildFSTLocked is the internal FST rebuild — no lock taken.
// MUST be called while Engine.mu.Lock() is held.
func (e *Engine) rebuildFSTLocked() error {
	e.inverted.RLock()
	glob := e.inverted.GetGlobalSeen()
	terms := make([]string, 0, len(glob))
	for t := range glob {
		terms = append(terms, t)
	}
	e.fstSize = len(glob)
	e.inverted.RUnlock()

	var buildErr error
	if e.fstPath != "" {
		buildErr = e.fst.BuildToFile(terms, e.fstPath)
	} else {
		buildErr = e.fst.Build(terms)
	}
	if buildErr != nil {
		return fmt.Errorf("index: fst build: %w", buildErr)
	}

	if w, ok := e.analyzer.(analysis.FSTWirer); ok {
		w.SetFST(e.fst)
	}
	if e.termStore != nil {
		e.termStore.AddTerms(terms)
	}

	e.fstDirty = false
	slog.Info("index: FST rebuilt", "terms", len(terms))
	return nil
}

// rebuildFSTIfNeeded rebuilds only when the vocabulary has changed since the
// last build. MUST be called while Engine.mu.Lock() is held.
//
// This used to compare len(globalSeen) against a high-water mark (fstSize),
// which only detects growth. Remove() shrinks globalSeen without ever
// rebuilding, so a delete-then-add-fewer-terms sequence left fstSize too
// high and silently skipped rebuilds — new terms became unresolvable by FST
// prefix lookup until the vocabulary grew back past its historical peak. A
// dirty flag set on both add and remove doesn't have that blind spot.
func (e *Engine) rebuildFSTIfNeeded() {
	if !e.fstDirty {
		return
	}
	if err := e.rebuildFSTLocked(); err != nil {
		slog.Warn("index: FST rebuild failed", "error", err)
	}
}

func (e *Engine) FSTContains(term string) bool { return e.fst.Contains(term) }
func (e *Engine) FSTPrefixSearch(prefix string, maxResults int) ([]string, error) {
	return e.fst.PrefixSearch(prefix, maxResults)
}

// Add indexes a document.
//
// The document embedding is computed before Engine.mu is taken. Embed is a
// network/inference call (HTTP round-trip to Ollama, or ONNX inference);
// running it while holding the lock previously stalled every other Add and
// Search for its full duration — Go's RWMutex blocks new readers once a
// writer is waiting, so a single slow embed call could serialise the whole
// engine behind it.
func (e *Engine) Add(ctx context.Context, originalID string, fullText string) error {
	docVec, err := e.embedder.Embed(ctx, fullText)
	if err != nil {
		slog.With("doc_id", originalID).Warn("Embedding failed, indexing purely lexically", "error", err)
		docVec = nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.addInternal(ctx, originalID, fullText, docVec); err != nil {
		return err
	}
	e.rebuildFSTIfNeeded()
	return nil
}

// AddWithVector indexes a document with a pre-computed embedding.
func (e *Engine) AddWithVector(ctx context.Context, originalID string, fullText string, docVec []float32) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.addInternal(ctx, originalID, fullText, docVec); err != nil {
		return err
	}
	e.rebuildFSTIfNeeded()
	return nil
}

// docEmbedBatch is the ONNX inference sweet spot measured on 12-thread
// consumer hardware: batch 128 hit an int8 GEMM cliff (12× slower per doc)
// and concurrent sessions oversubscribed the cores (5× slower).
const docEmbedBatch = 64

// AddBatch indexes all documents and rebuilds the FST once at the end.
// Takes Engine.mu.Lock() for its full duration.
//
// Document vectors are computed in batched ONNX forward passes on a
// producer goroutine that runs one chunk ahead of index construction, so
// embedding and lexical indexing overlap instead of alternating.
func (e *Engine) AddBatch(ctx context.Context, docs []BatchDoc) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Stable sort by ID: sort.Slice is not stable, so with duplicate IDs in
	// the input the relative order of the duplicates (and therefore which
	// one addInternal's idempotent overwrite leaves as the final version)
	// was nondeterministic. Stable sort preserves the caller's original
	// relative order for equal IDs, so "last occurrence in the input wins"
	// — deterministic and matches how a single-doc Add/Add/Add sequence
	// would behave.
	sorted := make([]BatchDoc, len(docs))
	copy(sorted, docs)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].ID < sorted[j].ID })
	docs = sorted

	if e.config.WordVectors {
		e.warmWordVectors(ctx, docs)
	}

	const chunkN = 1024
	type embChunk struct {
		start, end int
		vecs       [][]float32
	}
	done := make(chan struct{})
	defer close(done)
	ch := make(chan embChunk, 1)
	// The producer only reads docs and calls the thread-safe embedder; all
	// index mutation stays on this goroutine, which holds Engine.mu.
	go func() {
		defer close(ch)
		for start := 0; start < len(docs); start += chunkN {
			end := min(start+chunkN, len(docs))
			c := embChunk{start: start, end: end, vecs: e.embedDocs(ctx, docs[start:end])}
			select {
			case ch <- c:
			case <-done:
				return
			}
		}
	}()

	for c := range ch {
		for i := c.start; i < c.end; i++ {
			docVec := c.vecs[i-c.start]
			if docVec == nil {
				// Batch embedding failed for this doc (or its whole chunk) —
				// addInternal no longer retries this itself (see Add's
				// doc comment for why embedding moved out of the locked
				// path), so retry once here, single-doc, before indexing.
				if v, err := e.embedder.Embed(ctx, docs[i].Text); err == nil {
					docVec = v
				} else {
					slog.With("doc_id", docs[i].ID).Warn("Embedding failed, indexing purely lexically", "error", err)
				}
			}
			if err := e.addInternal(ctx, docs[i].ID, docs[i].Text, docVec); err != nil {
				// Documents before i are already applied to the live index.
				// Rebuild the FST so their new terms are still resolvable
				// even though the batch overall reports an error, instead
				// of leaving the vocabulary silently out of sync.
				if rebuildErr := e.rebuildFSTLocked(); rebuildErr != nil {
					slog.Warn("index: FST rebuild after partial AddBatch failure failed", "error", rebuildErr)
				}
				return err
			}
		}
	}
	return e.rebuildFSTLocked()
}

// warmWordVectors embeds every vocabulary token in docs that has no stored
// word vector yet and writes the result directly into the vector store.
// The previous implementation only warmed the LRU embed cache: with a
// vocabulary much larger than the cache (70k terms vs 10k entries) the
// early entries were evicted before addInternal read them back, so most
// words were embedded twice at full cost.
func (e *Engine) warmWordVectors(ctx context.Context, docs []BatchDoc) {
	tokenSet := make(map[string]struct{})
	for _, d := range docs {
		for _, t := range e.analyzer.Analyze(d.Text) {
			if !e.vectors.HasWordVector(t.Term) {
				tokenSet[t.Term] = struct{}{}
			}
		}
	}
	if len(tokenSet) == 0 {
		return
	}
	tokens := make([]string, 0, len(tokenSet))
	for t := range tokenSet {
		tokens = append(tokens, t)
	}
	sort.Strings(tokens)

	const warmBatch = 512
	for i := 0; i < len(tokens); i += warmBatch {
		end := min(i+warmBatch, len(tokens))
		chunk := tokens[i:end]
		vecs, err := e.embedder.EmbedBatch(ctx, chunk)
		if err != nil || len(vecs) != len(chunk) {
			slog.Warn("index: word-vector warm-up failed", "error", err)
			continue
		}
		e.vectors.Lock()
		wordVecs := e.vectors.GetWordVectors()
		for j, t := range chunk {
			nv := normalizeVector(vecs[j])
			if nv == nil {
				continue
			}
			wordVecs[t] = VectorEntry{
				Vector:    FloatsToFloat16(nv),
				Magnitude: ranking.Magnitude(nv),
			}
		}
		e.vectors.Unlock()
	}
}

// embedDocs returns one vector per doc, aligned by index. Docs with a
// caller-provided vector keep it; docs whose batch fails stay nil and fall
// back to single-doc embedding inside addInternal. Texts are embedded in
// length-sorted batches so each batch pads to its own longest member rather
// than the corpus worst case (2.1× less ONNX compute on MS MARCO).
func (e *Engine) embedDocs(ctx context.Context, docs []BatchDoc) [][]float32 {
	out := make([][]float32, len(docs))
	var need []int
	for i, d := range docs {
		if d.Vector != nil {
			out[i] = d.Vector
		} else {
			need = append(need, i)
		}
	}
	sort.Slice(need, func(a, b int) bool {
		la, lb := len(docs[need[a]].Text), len(docs[need[b]].Text)
		if la != lb {
			return la < lb
		}
		return need[a] < need[b]
	})
	for i := 0; i < len(need); i += docEmbedBatch {
		end := min(i+docEmbedBatch, len(need))
		texts := make([]string, end-i)
		for j, idx := range need[i:end] {
			texts[j] = docs[idx].Text
		}
		vecs, err := e.embedder.EmbedBatch(ctx, texts)
		if err != nil || len(vecs) != len(texts) {
			slog.Warn("index: batch document embedding failed; falling back to per-doc", "error", err)
			continue
		}
		for j, idx := range need[i:end] {
			out[idx] = vecs[j]
		}
	}
	return out
}

// Remove deletes all index entries for originalID.
// Takes Engine.mu.Lock() for its full duration.
func (e *Engine) Remove(ctx context.Context, originalID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.journal != nil {
		if err := e.journal.Delete(ctx, []byte(originalID)); err != nil {
			return fmt.Errorf("index: journal delete: %w", err)
		}
	}

	h := fnv.New64a()
	h.Write([]byte(originalID))
	internalID := h.Sum64()

	e.inverted.Lock()
	e.vectors.Lock()
	e.phonetics.Lock()
	defer e.inverted.Unlock()
	defer e.vectors.Unlock()
	defer e.phonetics.Unlock()

	idxData := e.inverted.GetData()
	idxPhon := e.phonetics.GetData()
	idxFrags := e.inverted.GetDocFragments()
	docVecStore := e.vectors.GetVectors()
	glob := e.inverted.GetGlobalSeen()
	docToks := e.inverted.GetDocTokens()

	oldFrags, exists := idxFrags[internalID]
	if !exists {
		return nil
	}

	for _, frag := range oldFrags {
		if idList, ok := idxData[frag]; ok {
			idxData[frag] = removeID(idList, internalID)
		}
		if idList, ok := idxPhon[frag]; ok {
			idxPhon[frag] = removeID(idList, internalID)
		}
	}
	delete(idxFrags, internalID)
	delete(docVecStore, internalID)
	delete(e.idMapping, internalID)
	delete(e.docText, internalID)

	// Decrement globalSeen reference counts for this document's raw tokens.
	if rawToks, ok := docToks[internalID]; ok {
		for _, tok := range rawToks {
			if n := glob[tok]; n <= 1 {
				delete(glob, tok)
				e.fstDirty = true
			} else {
				glob[tok] = n - 1
			}
		}
		delete(docToks, internalID)
	}

	e.bm25.Remove(internalID)
	e.tfidf.Remove(internalID)

	return nil
}

// GetText returns the original full text last indexed under originalID, and
// whether a document with that ID currently exists in the index.
func (e *Engine) GetText(originalID string) (string, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	h := fnv.New64a()
	h.Write([]byte(originalID))
	internalID := h.Sum64()

	text, ok := e.docText[internalID]
	return text, ok
}

// Count returns the number of documents currently held in the index.
func (e *Engine) Count() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.docText)
}

func (e *Engine) addInternal(ctx context.Context, originalID string, fullText string, preVec []float32) error {
	if e.journal != nil {
		if err := e.journal.Put(ctx, []byte(originalID), []byte(fullText)); err != nil {
			return fmt.Errorf("index: journal write: %w", err)
		}
	}

	logger := slog.With("doc_id", originalID)

	tokens := e.analyzer.Analyze(fullText)
	rawTokens := make([]string, 0, len(tokens))
	for _, t := range tokens {
		rawTokens = append(rawTokens, t.Term)
	}

	// preVec is already computed by the caller (Add, AddWithVector and
	// AddBatch all embed before calling addInternal) — this function never
	// makes its own embedding call, so it never holds Engine.mu across one.
	docVec := normalizeVector(preVec)

	tempWordVectors := make(map[string]VectorEntry)
	if e.config.WordVectors {
		var tokensToEmbed []string
		for _, t := range rawTokens {
			if _, exists := tempWordVectors[t]; !exists && !e.vectors.HasWordVector(t) {
				tokensToEmbed = append(tokensToEmbed, t)
				tempWordVectors[t] = VectorEntry{}
			}
		}

		const embedBatchSize = 512
		for i := 0; i < len(tokensToEmbed); i += embedBatchSize {
			end := i + embedBatchSize
			if end > len(tokensToEmbed) {
				end = len(tokensToEmbed)
			}
			chunk := tokensToEmbed[i:end]
			batchVecs, err := e.embedder.EmbedBatch(ctx, chunk)
			if err == nil && len(batchVecs) == len(chunk) {
				for j, t := range chunk {
					nv := normalizeVector(batchVecs[j])
					if nv == nil {
						// Per-item embedding failure inside an otherwise-ok
						// batch: don't store an empty entry — that would
						// make HasWordVector permanently true for a word
						// that was never actually embedded.
						delete(tempWordVectors, t)
						continue
					}
					tempWordVectors[t] = VectorEntry{
						Vector:    FloatsToFloat16(nv),
						Magnitude: ranking.Magnitude(nv),
					}
				}
			} else {
				logger.Warn("Batch embedding failed for tokens", "error", err)
				for _, t := range chunk {
					delete(tempWordVectors, t)
				}
			}
		}
	}

	h := fnv.New64a()
	h.Write([]byte(originalID))
	internalID := h.Sum64()

	e.inverted.Lock()
	e.vectors.Lock()
	e.phonetics.Lock()
	defer e.inverted.Unlock()
	defer e.vectors.Unlock()
	defer e.phonetics.Unlock()

	idxData := e.inverted.GetData()
	idxPhon := e.phonetics.GetData()
	idxFrags := e.inverted.GetDocFragments()
	wordVecs := e.vectors.GetWordVectors()
	docVecStore := e.vectors.GetVectors()
	glob := e.inverted.GetGlobalSeen()
	docToks := e.inverted.GetDocTokens()

	// Detect a 64-bit doc-ID hash collision: internalID already maps to a
	// *different* originalID. Without this check the second document would
	// silently overwrite the first's postings, vector and BM25 state below.
	if existing, ok := e.idMapping[internalID]; ok && existing != originalID {
		return fmt.Errorf("index: id hash collision: %q and %q both hash to %d", existing, originalID, internalID)
	}

	// Idempotency: remove previous postings for this document.
	if oldFrags, exists := idxFrags[internalID]; exists {
		for _, frag := range oldFrags {
			if idList, ok := idxData[frag]; ok {
				idxData[frag] = removeID(idList, internalID)
			}
			if idList, ok := idxPhon[frag]; ok {
				idxPhon[frag] = removeID(idList, internalID)
			}
		}
		// Decrement globalSeen for the old tokens before overwriting.
		if oldToks, ok := docToks[internalID]; ok {
			for _, tok := range oldToks {
				if n := glob[tok]; n <= 1 {
					delete(glob, tok)
					e.fstDirty = true
				} else {
					glob[tok] = n - 1
				}
			}
		}
		e.bm25.Remove(internalID)
		e.tfidf.Remove(internalID)
	}

	e.idMapping[internalID] = originalID
	e.docText[internalID] = fullText

	if docVec != nil {
		docVecStore[internalID] = VectorEntry{
			Vector:    FloatsToFloat16(docVec),
			Magnitude: ranking.Magnitude(docVec),
		}
	} else {
		// Re-indexing with no usable vector (embedder down, or the caller
		// didn't supply one) must drop any vector left over from a previous
		// version of this document — otherwise semantic search keeps
		// matching content the document no longer has.
		delete(docVecStore, internalID)
	}
	maps.Copy(wordVecs, tempWordVectors)

	seenInDoc := make(map[string]bool)
	var docFrags []string

	tokCnt := e.inverted.GetTokenCounts()
	vocab := e.inverted.GetVocabulary()

	for _, token := range rawTokens {
		tokCnt[token]++

		for _, frag := range generateEdgeNgrams(token) {
			if seenInDoc[frag] {
				continue
			}
			seenInDoc[frag] = true
			idxData[frag] = append(idxData[frag], internalID)
			docFrags = append(docFrags, frag)
		}

		if phon := analysis.Soundex(token); phon != "" && !seenInDoc[phon] {
			idxPhon[phon] = append(idxPhon[phon], internalID)
			seenInDoc[phon] = true
			docFrags = append(docFrags, phon)
		}

		// Increment globalSeen reference count; add to BKTree on first occurrence.
		if glob[token] == 0 {
			vocab[len(token)] = append(vocab[len(token)], token)
			e.bkTree.Add(token)
			e.fstDirty = true
		}
		glob[token]++
	}

	idxFrags[internalID] = docFrags
	docToks[internalID] = append([]string(nil), rawTokens...) // snapshot

	e.bm25.Index(internalID, rawTokens)
	e.tfidf.Index(internalID, rawTokens)

	return nil
}

// Search executes a hybrid query.
//
// The query embedding is computed before Engine.mu is taken, for the same
// reason as in Add: Embed is a network/inference call, and running it while
// holding even RLock previously stalled Add/Remove (which need the write
// lock) and, transitively, every other Search queued behind them.
func (e *Engine) Search(ctx context.Context, query string) ([]SearchResponse, error) {
	queryVec, embErr := e.embedder.Embed(ctx, query)
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
	keywordScores, matchTokens := e.lexicalPass(rawTokens)
	e.phonetics.RUnlock()
	e.inverted.RUnlock()

	e.vectors.RLock()
	vectorScores := e.vectorPass(queryVec)
	e.vectors.RUnlock()

	ranks := e.rankAndFuse(keywordScores, matchTokens, rawTokens, vectorScores)

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
	if !weakResults && e.config.WordVectors && len(e.bm25.Query(rawTokens)) == 0 {
		weakResults = true
	}

	if weakResults && e.config.WordVectors {
		expandedTokens := e.expandTokens(rawTokens)

		e.inverted.RLock()
		expandedKeywords, expandedMatches := e.neuralExpand(rawTokens, expandedTokens)
		e.inverted.RUnlock()

		for id, score := range keywordScores {
			expandedKeywords[id] += score
			if expandedMatches[id] == nil {
				expandedMatches[id] = make(map[string]bool)
			}
			for mt := range matchTokens[id] {
				expandedMatches[id][mt] = true
			}
		}

		ranks = e.rankAndFuse(expandedKeywords, expandedMatches, rawTokens, vectorScores)
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

func (e *Engine) lexicalPass(queryTokens []string) (map[uint64]float64, map[uint64]map[string]bool) {
	keywordScores := make(map[uint64]float64)
	matchTokens := make(map[uint64]map[string]bool)

	idxData := e.inverted.GetData()
	idxPhon := e.phonetics.GetData()

	for _, token := range queryTokens {
		Q := len(token)

		var frags []string
		if Q >= 3 {
			frags = generateEdgeNgrams(token)
		} else {
			frags = []string{token}
		}

		for _, frag := range frags {
			if ids, ok := idxData[frag]; ok {
				for _, id := range ids {
					keywordScores[id] += (float64(len(frag)) / float64(Q)) * 100.0
					if matchTokens[id] == nil {
						matchTokens[id] = make(map[string]bool)
					}
					matchTokens[id][token] = true
				}
			}
		}

		if phon := analysis.Soundex(token); phon != "" {
			if ids, ok := idxPhon[phon]; ok {
				for _, id := range ids {
					keywordScores[id] += e.config.PhoneticWeight
					if matchTokens[id] == nil {
						matchTokens[id] = make(map[string]bool)
					}
					matchTokens[id][token] = true
				}
			}
		}

		if Q >= 2 {
			for _, match := range e.bkTree.Search(token, e.config.FuzzyMaxDist) {
				if match.Distance == 0 {
					continue
				}
				if ids, ok := idxData[match.Word]; ok {
					for _, id := range ids {
						keywordScores[id] += 60.0 / float64(match.Distance)
						if matchTokens[id] == nil {
							matchTokens[id] = make(map[string]bool)
						}
						matchTokens[id][token] = true
					}
				}
			}
		}
	}
	return keywordScores, matchTokens
}

// vectorPass scores all documents by dot product with the query vector.
// Negative dot products are clamped to 0 — a document pointing away from
// the query has zero semantic relevance, not negative relevance.
func (e *Engine) vectorPass(queryVec []float32) map[uint64]float64 {
	scores := make(map[uint64]float64)
	if len(queryVec) == 0 {
		return scores
	}
	for id, entry := range e.vectors.GetVectors() {
		s := ranking.DotProduct(queryVec, Float16ToFloats(entry.Vector))
		if s > 0 {
			scores[id] = s
		}
	}
	return scores
}

func (e *Engine) neuralExpand(originalTokens []string, expandedTokens []string) (map[uint64]float64, map[uint64]map[string]bool) {
	keywordScores := make(map[uint64]float64)
	matchTokens := make(map[uint64]map[string]bool)
	idxData := e.inverted.GetData()

	for _, neighbor := range expandedTokens {
		targets := make(map[uint64]bool)
		if ids, ok := idxData[neighbor]; ok {
			for _, id := range ids {
				targets[id] = true
			}
		}
		if runes := []rune(neighbor); len(runes) > 3 {
			if ids, ok := idxData[string(runes[:3])]; ok {
				for _, id := range ids {
					targets[id] = true
				}
			}
		}
		for id := range targets {
			keywordScores[id] += 20000.0
			if matchTokens[id] == nil {
				matchTokens[id] = make(map[string]bool)
			}
			if len(originalTokens) > 0 {
				matchTokens[id][originalTokens[0]] = true
			}
		}
	}
	return keywordScores, matchTokens
}

// buildKwRank turns raw n-gram/phonetic/fuzzy coverage scores into a single
// per-document ranking key that prefers real BM25 relevance when it exists.
// A document with a BM25 score for the query terms ranks in BM25's positive
// range (BM25 is +1 smoothed, so always > 0). Everything else — fuzzy or
// phonetic-only hits, or neural-expansion hits that only ever match neighbor
// terms rather than the literal query — has no BM25 signal for the literal
// query terms, and ranks below the BM25-scored documents by its own coverage
// score rather than being collapsed into a single alphabetical tie.
//
// This used to be inlined separately in each branch of rankAndFuse, and the
// BM25-only branch didn't have it at all (it passed bm25ByID straight to the
// scorer, so any candidate with no BM25 score silently got a keyword score
// of 0 — losing fuzzy/phonetic hits, and losing every neural-expansion
// candidate's score, since expansion always queries BM25 with the original,
// intentionally-non-matching query terms).
func buildKwRank(kwScores map[uint64]float64, bm25ByID map[uint64]float64) map[uint64]float64 {
	kwRank := make(map[uint64]float64, len(kwScores))
	for id, cov := range kwScores {
		if cov <= 0 {
			continue
		}
		if s, ok := bm25ByID[id]; ok {
			kwRank[id] = 1.0 + s
		} else {
			kwRank[id] = cov * 1e-9
		}
	}
	return kwRank
}

func (e *Engine) rankAndFuse(
	kwScores map[uint64]float64,
	matchToks map[uint64]map[string]bool,
	qryToks []string,
	vScores map[uint64]float64,
) []SearchResponse {

	bm25Results := e.bm25.Query(qryToks)
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
		scored := e.scorer.Score(kwIDs, kwRank, nil, nil, e.idMapping)
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

	// e.idMapping is safe here — Engine.mu.RLock() (Search) or Lock() (others) is held.
	scored := e.scorer.Score(kwIDs, kwRank, vcIDs, vScores, e.idMapping)

	results := make([]SearchResponse, len(scored))
	for i, r := range scored {
		results[i] = SearchResponse{ID: r.ID, Score: r.Score}
	}
	return results
}

// neighborCandidate pairs a word with its similarity score for sorting.
type neighborCandidate struct {
	word  string
	score float32
}

// getSemanticNeighbors returns the topN most similar words to token by dot
// product, sorted descending by similarity. Previously truncated without
// sorting — nondeterministic under Go's randomised map iteration.
func (e *Engine) getSemanticNeighbors(token string, topN int, threshold float32) []string {
	e.vectors.RLock()
	defer e.vectors.RUnlock()

	wordVecs := e.vectors.GetWordVectors()
	tokenEntry, ok := wordVecs[token]
	if !ok {
		return nil
	}

	tokenVec := Float16ToFloats(tokenEntry.Vector)
	var candidates []neighborCandidate
	for word, entry := range wordVecs {
		if word == token {
			continue
		}
		s := float32(ranking.DotProduct(tokenVec, Float16ToFloats(entry.Vector)))
		if s >= threshold {
			candidates = append(candidates, neighborCandidate{word: word, score: s})
		}
	}

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

func generateEdgeNgrams(token string) []string {
	const (
		MinGram = 3
		MaxGram = 10
	)
	runes := []rune(token)
	n := len(runes)

	if n < MinGram {
		return []string{token}
	}

	limit := n
	if limit > MaxGram {
		limit = MaxGram
	}

	cap := 1 + (limit - MinGram)
	results := make([]string, 0, cap)
	results = append(results, token)

	for i := MinGram; i < limit; i++ {
		results = append(results, string(runes[0:i]))
	}

	return results
}

func removeID(ids []uint64, target uint64) []uint64 {
	out := ids[:0]
	for _, id := range ids {
		if id != target {
			out = append(out, id)
		}
	}
	return out
}

var saveFormatMagic = [4]byte{'Z', 'N', 'T', 'H'}

// saveFormatVersion v3: globalSeen is now map[string]int (ref count),
// docTokens map[uint64][]string added for globalSeen management on Remove.
// v4: docText map[uint64]string added so GetText/GetDocument can return the
// original indexed text instead of just IDs and scores.
const saveFormatVersion uint16 = 4

var ErrIncompatibleVersion = fmt.Errorf("index: incompatible file version — rebuild the index with the current binary")

// Save serialises all index state to filepath. Takes Engine.mu.RLock() so
// concurrent Searches can proceed during save, but Add/Remove/Load block.
//
// Save additionally takes saveMu, a dedicated mutex: mu.RLock() alone allows
// two Saves to run concurrently (both are readers), and both would write to
// the same fixed ".tmp" path, corrupting each other's output.
func (e *Engine) Save(filepath string) error {
	e.saveMu.Lock()
	defer e.saveMu.Unlock()

	e.mu.RLock()
	defer e.mu.RUnlock()

	start := time.Now()
	e.inverted.RLock()
	e.vectors.RLock()
	e.phonetics.RLock()
	defer e.inverted.RUnlock()
	defer e.vectors.RUnlock()
	defer e.phonetics.RUnlock()

	slog.Info("Saving index state", "path", filepath)

	tmp := filepath + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}

	if _, err := file.Write(saveFormatMagic[:]); err != nil {
		file.Close()
		os.Remove(tmp)
		return err
	}
	var vbuf [2]byte
	vbuf[0] = byte(saveFormatVersion >> 8)
	vbuf[1] = byte(saveFormatVersion)
	if _, err := file.Write(vbuf[:]); err != nil {
		file.Close()
		os.Remove(tmp)
		return err
	}

	enc := gob.NewEncoder(file)

	bm25Lengths, bm25TermFreqs, bm25DocFreq, bm25TotalDocs, bm25TotalLen := e.bm25.State()
	tfidfLengths, tfidfTermFreqs, tfidfDocFreq, tfidfTotalDocs := e.tfidf.State()

	state := []any{
		e.inverted.GetData(), e.idMapping, e.vectors.GetVectors(),
		e.inverted.GetTokenCounts(), e.phonetics.GetData(), e.inverted.GetVocabulary(),
		e.inverted.GetGlobalSeen(), e.vectors.GetWordVectors(), e.inverted.GetDocFragments(),
		bm25Lengths, bm25TermFreqs, bm25DocFreq, bm25TotalDocs, bm25TotalLen,
		tfidfLengths, tfidfTermFreqs, tfidfDocFreq, tfidfTotalDocs,
		// v3: docTokens for globalSeen reference counting
		e.inverted.GetDocTokens(),
		// v4: original document text, for GetText/GetDocument
		e.docText,
	}
	for _, s := range state {
		if err := enc.Encode(s); err != nil {
			file.Close()
			os.Remove(tmp)
			return err
		}
	}

	// fsync the temp file's contents before rename, and fsync the containing
	// directory after rename. Without the first, a crash right after Close
	// can leave the renamed file truncated (the rename itself is durable,
	// but the data it points at might not be). Without the second, on most
	// filesystems the rename operation itself isn't guaranteed durable until
	// the directory entry is synced, so a crash could leave the old file's
	// name pointing at nothing or at stale data.
	if err := file.Sync(); err != nil {
		file.Close()
		os.Remove(tmp)
		return fmt.Errorf("index: fsync temp file: %w", err)
	}
	if err := file.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, filepath); err != nil {
		os.Remove(tmp)
		return err
	}
	if dir, err := os.Open(pathutil.Dir(filepath)); err == nil {
		if syncErr := dir.Sync(); syncErr != nil {
			// Directory fsync is expected to fail on Windows (no support for
			// syncing a directory handle) — best-effort only, log at Debug
			// so it doesn't look like an operational problem there. On
			// platforms where it's supposed to work, Debug is still visible
			// with verbose logging enabled.
			slog.Debug("index: directory fsync after save failed", "error", syncErr)
		}
		dir.Close()
	}

	slog.Info("Index saved", "entries", len(e.inverted.GetData()), "duration", time.Since(start))
	return nil
}

// Load restores index state from a gob file. Takes Engine.mu.Lock().
func (e *Engine) Load(filepath string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if err := e.load(filepath); err != nil {
		return err
	}

	e.inverted.RLock()
	vocabSize := len(e.inverted.GetGlobalSeen())
	e.inverted.RUnlock()

	if e.fstPath != "" {
		if err := e.fst.OpenFromFile(e.fstPath); err == nil {
			// The FST file on disk isn't guaranteed to match the vocabulary
			// we just loaded (stale file from a previous run, or a save
			// that didn't complete). A term-count mismatch is a cheap,
			// effective check — trusting the file blindly let a stale FST
			// silently survive restarts (queries for real terms in the
			// loaded index would fail prefix resolution, or vice versa).
			if e.fst.Size() == vocabSize {
				e.fstSize = vocabSize
				e.fstDirty = false
				if w, ok := e.analyzer.(analysis.FSTWirer); ok {
					w.SetFST(e.fst)
				}
				slog.Info("index: FST loaded from disk", "path", e.fstPath, "terms", e.fst.Size())
				return nil
			}
			slog.Warn("index: FST on disk does not match loaded vocabulary, rebuilding",
				"fst_terms", e.fst.Size(), "vocab_terms", vocabSize)
		} else {
			slog.Info("index: FST file not found, rebuilding", "path", e.fstPath)
		}
	}

	if err := e.rebuildFSTLocked(); err != nil {
		slog.Warn("index: FST rebuild after load failed", "error", err)
	}
	return nil
}

func (e *Engine) load(filepath string) error {
	start := time.Now()

	f, err := os.Open(filepath)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("No persistence file found, starting fresh", "path", filepath)
		}
		return err
	}
	defer f.Close()

	var magic [4]byte
	if _, err := f.Read(magic[:]); err != nil {
		return fmt.Errorf("index: failed to read file header: %w", err)
	}
	if magic != saveFormatMagic {
		return fmt.Errorf("index: not a ZENITH index file (bad magic bytes)")
	}
	var vbuf [2]byte
	if _, err := f.Read(vbuf[:]); err != nil {
		return fmt.Errorf("index: failed to read version: %w", err)
	}
	version := uint16(vbuf[0])<<8 | uint16(vbuf[1])
	if version != saveFormatVersion {
		return ErrIncompatibleVersion
	}

	dec := gob.NewDecoder(f)

	// Decode into brand-new, empty structures — never into the live engine's
	// maps. gob.Decode into an already-populated map merges entries into it
	// rather than replacing it, so decoding straight into e.inverted's live
	// maps (the previous approach) silently combined the loaded file with
	// whatever was already in memory instead of replacing it, and a decode
	// failure partway through left the engine in a half-loaded, inconsistent
	// state (e.g. postings and BM25 out of sync) that then got persisted on
	// the next Save. Only after every field decodes successfully do we swap
	// these into the live engine, atomically under the sub-index locks.
	vData := make(map[string][]uint64)
	vVectors := make(map[uint64]VectorEntry)
	vToken := make(map[string]int)
	vPhon := make(map[string][]uint64)
	vVocab := make(map[int][]string)
	vSeen := make(map[string]int)
	vWordVectors := make(map[string]VectorEntry)
	vFrag := make(map[uint64][]string)
	vDocToks := make(map[uint64][]string)
	vIDMapping := make(map[uint64]string)
	vDocText := make(map[uint64]string)

	var bm25Lengths map[uint64]int
	var bm25TermFreqs map[uint64]map[string]int
	var bm25DocFreq map[string]int
	var bm25TotalDocs, bm25TotalLen int

	var tfidfLengths map[uint64]int
	var tfidfTermFreqs map[uint64]map[string]int
	var tfidfDocFreq map[string]int
	var tfidfTotalDocs int

	state := []any{
		&vData, &vIDMapping, &vVectors,
		&vToken, &vPhon, &vVocab,
		&vSeen, &vWordVectors, &vFrag,
		&bm25Lengths, &bm25TermFreqs, &bm25DocFreq, &bm25TotalDocs, &bm25TotalLen,
		&tfidfLengths, &tfidfTermFreqs, &tfidfDocFreq, &tfidfTotalDocs,
		&vDocToks, // v3
		&vDocText, // v4
	}
	for _, s := range state {
		if err := dec.Decode(s); err != nil {
			return fmt.Errorf("index: decode index state: %w", err)
		}
	}

	// Every field decoded successfully — swap it all in now.
	e.inverted.Lock()
	e.vectors.Lock()
	e.phonetics.Lock()
	e.inverted.ReplaceAll(vData, vToken, vVocab, vSeen, vFrag, vDocToks)
	e.vectors.ReplaceAll(vVectors, vWordVectors)
	e.phonetics.ReplaceAll(vPhon)
	e.idMapping = vIDMapping
	e.docText = vDocText
	e.inverted.Unlock()
	e.vectors.Unlock()
	e.phonetics.Unlock()

	e.bm25.LoadState(bm25Lengths, bm25TermFreqs, bm25DocFreq, bm25TotalDocs, bm25TotalLen)
	e.tfidf.LoadState(tfidfLengths, tfidfTermFreqs, tfidfDocFreq, tfidfTotalDocs)

	// The BK-tree (fuzzy search) is never persisted — rebuild it from the
	// loaded vocabulary. Previously it was simply never repopulated on Load
	// at all: fuzzy search returned nothing for the lifetime of the process
	// after any restart.
	newBK := analysis.NewBKTree()
	for term := range vSeen {
		newBK.Add(term)
	}
	e.bkTree = newBK

	slog.Info("Index loaded", "docs", len(e.idMapping), "duration", time.Since(start))
	return nil
}

// ExplainHit is the raw per-signal evidence for one document (Explain mode). Unlike fused RRF scores,
// these values are absolute and can be thresholded by callers.
type ExplainHit struct {
	ID       string
	Lexical  float64   // raw BM25 for the analysed query terms plus their synonyms; 0 if none occur
	Semantic float64   // cosine(query, doc): vectors are L2-normalised, so the dot product; 0 without vectors
	Terms    []TermHit // best match per base query term that the document contains
}

// TermHit records how one base query term was found in a document.
type TermHit struct {
	Term    string // analysed base query term
	Matched string // analysed document term that satisfied it
	Dist    int    // 0 for exact and synonym matches; Levenshtein distance for BK-tree matches
	Synonym bool
}

type exactTokenizer interface{ TokenizeExact(text string) []string }

func dedupe(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			out = append(out, s)
		}
	}
	return out
}

// Explain returns the analysed base query terms and, for every document with at least one term hit or
// a positive semantic score, its raw signals. It scans every document, so it is meant for small
// per-request namespaces (hundreds of documents), not the persistent index.
func (e *Engine) Explain(ctx context.Context, query string) ([]string, []ExplainHit, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var base []string
	if et, ok := e.analyzer.(exactTokenizer); ok {
		base = et.TokenizeExact(query)
	} else {
		for _, t := range e.analyzer.Analyze(query) {
			base = append(base, t.Term)
		}
	}
	base = dedupe(base)

	type cand struct {
		tok  string
		dist int
		syn  bool
	}
	rank := func(c cand) int {
		switch {
		case c.dist == 0 && !c.syn:
			return 0
		case c.syn:
			return 1
		default:
			return 1 + c.dist
		}
	}
	cands := make(map[string][]cand, len(base))
	all := append([]string(nil), base...)
	for _, t := range base {
		cs := []cand{{tok: t}}
		for _, s := range analysis.Synonyms(t) {
			cs = append(cs, cand{tok: s, syn: true})
			all = append(all, s)
		}
		if len([]rune(t)) >= 2 {
			for _, m := range e.bkTree.Search(t, e.config.FuzzyMaxDist) {
				if m.Distance > 0 {
					cs = append(cs, cand{tok: m.Word, dist: m.Distance})
				}
			}
		}
		sort.SliceStable(cs, func(i, j int) bool {
			if rank(cs[i]) != rank(cs[j]) {
				return rank(cs[i]) < rank(cs[j])
			}
			return cs[i].tok < cs[j].tok
		})
		cands[t] = cs
	}

	lex := make(map[uint64]float64)
	for _, r := range e.bm25.Query(dedupe(all)) {
		lex[r.DocID] = r.Score
	}

	var queryVec []float32
	if e.embedder != nil {
		var err error
		queryVec, err = e.embedder.Embed(ctx, query)
		if err != nil {
			slog.Warn("explain: semantic signal unavailable — embedder failed", "error", err)
			queryVec = nil
		}
	}

	e.inverted.RLock()
	e.vectors.RLock()
	defer e.vectors.RUnlock()
	defer e.inverted.RUnlock()
	docVecs := e.vectors.GetVectors()

	hits := make([]ExplainHit, 0)
	for id, toks := range e.inverted.GetDocTokens() {
		set := make(map[string]struct{}, len(toks))
		for _, tk := range toks {
			set[tk] = struct{}{}
		}
		var terms []TermHit
		for _, t := range base {
			for _, c := range cands[t] {
				if _, ok := set[c.tok]; ok {
					terms = append(terms, TermHit{Term: t, Matched: c.tok, Dist: c.dist, Synonym: c.syn})
					break
				}
			}
		}
		sem := 0.0
		if len(queryVec) > 0 {
			if v, ok := docVecs[id]; ok {
				if s := ranking.DotProduct(queryVec, Float16ToFloats(v.Vector)); s > 0 {
					sem = s
				}
			}
		}
		if len(terms) == 0 && sem <= 0 {
			continue
		}
		hits = append(hits, ExplainHit{ID: e.idMapping[id], Lexical: lex[id], Semantic: sem, Terms: terms})
	}
	sort.Slice(hits, func(i, j int) bool {
		a, b := hits[i], hits[j]
		if len(a.Terms) != len(b.Terms) {
			return len(a.Terms) > len(b.Terms)
		}
		if a.Lexical != b.Lexical {
			return a.Lexical > b.Lexical
		}
		if a.Semantic != b.Semantic {
			return a.Semantic > b.Semantic
		}
		return a.ID < b.ID
	})
	return base, hits, nil
}
