package index

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"io"
	"log/slog"
	"maps"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/ann"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/querycache"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"golang.org/x/sync/singleflight"
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
	Attrs  Attrs
}

// DocumentJournal durably records document mutations before they touch the
// in-memory index. Satisfied by *storage.Engine — its Put/Delete signatures
// match exactly. Set via SetDocumentJournal; nil means no journaling.
type DocumentJournal interface {
	Put(ctx context.Context, key, value []byte) error
	Delete(ctx context.Context, key []byte) error
}

// Txn stages a batch of document mutations for one atomic commit. Satisfied
// by *storage.Txn (internal/storage) — every method here uses only []byte,
// context.Context, and error, so no import of internal/storage is needed in
// either direction (the same pattern DocumentJournal already uses).
type Txn interface {
	Put(key, value []byte) error
	Delete(key []byte) error
	Commit(ctx context.Context) error
	Discard() error
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

	// saveMu serialises Save calls (and Close) so two flushes never interleave.
	saveMu sync.Mutex
	// compactMu serialises compactions and keeps Load/Close from unmapping
	// segments a running compaction is still reading. Lock order:
	// compactMu → mu.
	compactMu sync.Mutex

	config    *config.Config
	inverted  *InvertedIndex
	vectors   *VectorStore
	phonetics *PhoneticIndex

	// bk is the BK-tree fuzzy-match fallback (used only when the FST cannot
	// serve the query: FST not built, or edit distance above 3). It is built
	// lazily from the live vocabulary, since the FST automaton normally answers
	// fuzzy queries and the tree costs ~200 bytes per term.
	bk   *analysis.BKTree
	bkMu sync.Mutex

	// Persistence state. See layers.go: documents live either in segs
	// (immutable, memory-mapped) or in the in-memory maps below (the delta).
	segs        []*segLayer
	dbPath      string // manifest path the engine is bound to; "" until first Load/Save
	nextGen     uint64 // next segment file number
	manifestGen uint64 // generation of the last committed manifest
	// writeGen counts mutations (Add/AddBatch/Remove) to the live delta,
	// independent of manifestGen (which only bumps on a committed flush).
	// Embedded into every query-cache key (see internal/querycache wiring in
	// search.go): a write makes every previously-cached key for this engine
	// permanently unreachable, with no enumeration or active invalidation
	// logic required. Guarded by e.mu — every mutator already holds
	// e.mu.Lock() when it bumps this, so no new lock or atomic is needed.
	writeGen uint64
	// processEpoch is a random value generated once per Engine instance and
	// mixed into every query-cache key (see cache_key.go's bucketKey). It
	// exists only to protect the optional L2 (Redis) tier's cross-process
	// correctness: writeGen alone is meaningless across a restart (it
	// starts at 0 again) or between replicas, so without this, a different
	// process could read back another process's stale L2 entries at the
	// same (namespace, writeGen) pair. L1 never needs this (it never
	// outlives the process that wrote it); it's harmless there too, since
	// bucketKey includes it unconditionally rather than branching per tier.
	processEpoch string
	// cache is the query-result cache (nil when Config.QueryCacheSize <= 0,
	// which is a complete no-op, not a degraded mode). See search.go for how
	// SearchFilteredWeighted uses it.
	cache *querycache.Tiered[cacheResult]
	// searchSF de-duplicates concurrent cache misses for the same key: N
	// goroutines racing on the same uncached query+filter+weights would
	// otherwise each pay the full lexical+vector+fusion pipeline
	// independently. Mirrors embedding.CachingEmbedder's embedSF/querySF.
	searchSF singleflight.Group
	// cacheObserver receives query-cache hit/miss events; defaults to a
	// no-op so instrumentation is opt-in (see SetCacheObserver).
	cacheObserver CacheObserver
	// semanticScans counts how many times the semantic near-duplicate scan
	// actually ran. Test-only visibility (package index tests read it
	// directly); proves Config.QueryCacheSemanticThreshold == 0 truly skips
	// the scan, not just that it found no hits.
	semanticScans atomic.Int64

	pendingDels map[uint64]struct{} // segment docs deleted since the last flush
	// frozen is the delta a flush is writing (nil when none): read-only, still
	// searched, and replaced by a segment when the flush commits. See frozen.go.
	frozen *memLayer
	// needGC: the engine was just bound to a path, so the first commit must
	// delete segment files the new manifest does not name.
	needGC bool
	// ANN sidecar bookkeeping (ann_persist.go). annChanges counts graph
	// mutations since the file was last written; guarded by mu.
	annSaveMu   sync.Mutex
	annSaving   atomic.Bool
	annWG       sync.WaitGroup // outstanding background SaveANN goroutines; Close waits on this
	annChanges  int64
	annSaved    bool // the file on disk reflects the graph as of the last save
	annFromDisk bool // the current graph was restored from the file
	autoCompact bool
	// replaceLegacy lets the first Save replace an old-format file at dbPath.
	// Only Migrate sets it, after it has copied the original aside.
	replaceLegacy bool
	compacting    atomic.Bool

	embedder embedding.Embedder
	scorer   ranking.Scorer
	analyzer analysis.Analyzer

	bm25 *ranking.BM25Scorer

	idMapping map[uint64]string
	docText   map[uint64]string // original full text per document, for GetText
	attrs     map[uint64]Attrs  // per-document metadata for filtered search; absent = no attrs
	attrIdx   *attrIndex        // inverted index over attrs, for selective filters (attrindex.go)

	// ann is the HNSW graph over document vectors. nil while the corpus is
	// below annMinDocs (brute force is exact and fast enough there); built
	// once the threshold is crossed and maintained incrementally after. It is
	// not persisted: Load rebuilds it from the stored vectors.
	ann        *ann.Index
	annMinDocs int
	// annLatency/exactLatency are rolling-average per-path search latencies
	// (milliseconds), fed by every search that actually took that path.
	// Used only when Config.ANNThresholdBandPct > 0 — see ann_adaptive.go.
	annLatency, exactLatency ewma
	// serialEmbed makes a search embed its query before starting the lexical
	// phase instead of overlapping them. Test hook for measuring the overlap.
	serialEmbed bool

	fst      *analysis.FSTDictionary
	fstSize  int  // informational only — last rebuild's term count
	fstDirty bool // true when the vocabulary has changed since the last FST build
	fstPath  string

	journal DocumentJournal
}

// NewEngine constructs a fully initialised Engine.
func NewEngine(cfg *config.Config, emb embedding.Embedder, scr ranking.Scorer, ana analysis.Analyzer) *Engine {
	e := &Engine{
		config:        cfg,
		inverted:      NewInvertedIndex(),
		vectors:       NewVectorStore(),
		phonetics:     NewPhoneticIndex(),
		embedder:      emb,
		scorer:        scr,
		analyzer:      ana,
		idMapping:     make(map[uint64]string),
		docText:       make(map[uint64]string),
		attrs:         make(map[uint64]Attrs),
		attrIdx:       newAttrIndex(),
		annMinDocs:    defaultANNMinDocs,
		autoCompact:   true,
		nextGen:       1,
		bm25:          ranking.NewBM25Scorer(ranking.BM25Params{}),
		fst:           analysis.NewFSTDictionary(),
		cacheObserver: noopCacheObserver{},
		processEpoch:  newProcessEpoch(),
	}
	e.bm25.SetBacking(segBacking{e})

	if cfg.QueryCacheSize > 0 {
		l1, err := querycache.NewMemCache[cacheResult](cfg.QueryCacheSize)
		if err != nil {
			// Unreachable: cfg.QueryCacheSize > 0 is guaranteed by the guard
			// above, and that is NewMemCache's only error condition.
			panic(err)
		}
		var l2 *querycache.BytesCache
		if cfg.QueryCacheRedisAddr != "" {
			ttl := cfg.QueryCacheTTL
			if ttl <= 0 {
				ttl = 5 * time.Minute
			}
			l2 = querycache.NewBytesCache(cfg.QueryCacheRedisAddr, ttl)
		}
		e.cache = querycache.NewTiered(l1, l2, encodeCacheResult, decodeCacheResult)
	}

	return e
}

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
	terms := e.liveTerms()
	e.fstSize = len(terms)
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
	if err := e.addInternal(ctx, originalID, fullText, docVec, nil); err != nil {
		return err
	}
	e.rebuildFSTIfNeeded()
	return nil
}

// AddWithVector indexes a document with a pre-computed embedding.
func (e *Engine) AddWithVector(ctx context.Context, originalID string, fullText string, docVec []float32) error {
	return e.AddWithVectorAttrs(ctx, originalID, fullText, docVec, nil)
}

// AddWithVectorAttrs is AddWithVector plus per-document metadata used by
// SearchWithFilter. Re-adding an ID replaces its attributes (nil clears them).
func (e *Engine) AddWithVectorAttrs(ctx context.Context, originalID string, fullText string, docVec []float32, attrs Attrs) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if err := e.addInternal(ctx, originalID, fullText, docVec, attrs); err != nil {
		return err
	}
	e.rebuildFSTIfNeeded()
	return nil
}

// EmbedText computes an embedding for text without indexing anything. Like
// Add, an embedding failure is non-fatal — it returns nil so the caller can
// still index (or persist) the document purely lexically. Callers that also
// need to index the text should pass the result to AddWithVector rather than
// calling Add and embedding twice.
func (e *Engine) EmbedText(ctx context.Context, text string) []float32 {
	vec, err := e.embedder.Embed(ctx, text)
	if err != nil {
		return nil
	}
	return vec
}

// EmbedTexts computes one embedding per text without indexing anything,
// using the same length-bucketed batching as AddBatch. The result aligns by
// index with texts; a failed batch leaves the corresponding entries nil.
func (e *Engine) EmbedTexts(ctx context.Context, texts []string) [][]float32 {
	docs := make([]BatchDoc, len(texts))
	for i, t := range texts {
		docs[i] = BatchDoc{Text: t}
	}
	return e.embedDocs(ctx, docs)
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
			if err := e.addInternal(ctx, docs[i].ID, docs[i].Text, docVec, docs[i].Attrs); err != nil {
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

// AddTransaction indexes docs atomically: every document's mutation is
// staged into txn, which is committed once at the end. If embedding or
// staging any document fails, txn is discarded and the whole call returns an
// error with nothing applied to the in-memory index — unlike AddBatch, which
// still applies documents before a later failure.
func (e *Engine) AddTransaction(ctx context.Context, docs []BatchDoc, txn Txn) error {
	// vecs is carried through to the apply loop below without re-embedding.
	// Kept as a local slice rather than writing back into docs[i].Vector —
	// AddBatch's own contract is non-mutating (it copies before sorting),
	// and this must match: a caller is not expected to see its input slice
	// altered after the call.
	vecs := e.embedDocs(ctx, docs)

	e.mu.Lock()
	defer e.mu.Unlock()

	for i, d := range docs {
		if vecs[i] == nil {
			if v, err := e.embedder.Embed(ctx, d.Text); err == nil {
				vecs[i] = v
			} else {
				slog.With("doc_id", d.ID).Warn("Embedding failed, indexing purely lexically", "error", err)
			}
		}
		if err := txn.Put([]byte(d.ID), encodeJournalValue(d.Text, vecs[i], d.Attrs)); err != nil {
			_ = txn.Discard()
			return fmt.Errorf("index: txn stage put %q: %w", d.ID, err)
		}
	}

	if err := txn.Commit(ctx); err != nil {
		return fmt.Errorf("index: txn commit: %w", err)
	}

	for i, d := range docs {
		if err := e.applyInternal(ctx, d.ID, d.Text, vecs[i], d.Attrs); err != nil {
			return fmt.Errorf("index: apply after committed txn: %w", err)
		}
	}
	return e.rebuildFSTLocked()
}

// RemoveBatch deletes all index entries for every id in ids atomically:
// every deletion is staged into txn, committed once, then applied to the
// in-memory delta. If Commit fails, nothing is removed.
func (e *Engine) RemoveBatch(ctx context.Context, ids []string, txn Txn) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, id := range ids {
		if err := txn.Delete([]byte(id)); err != nil {
			_ = txn.Discard()
			return fmt.Errorf("index: txn stage delete %q: %w", id, err)
		}
	}
	if err := txn.Commit(ctx); err != nil {
		return fmt.Errorf("index: txn commit: %w", err)
	}
	for _, id := range ids {
		if err := e.removeInternal(ctx, id); err != nil {
			return fmt.Errorf("index: apply remove after committed txn: %w", err)
		}
	}
	return nil
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
			if !e.hasWordVector(t.Term) {
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
	return e.removeInternal(ctx, originalID)
}

// removeInternal applies one document's removal to the in-memory delta
// without touching the journal — used directly by RemoveBatch, whose caller
// has already durably committed the deletion via its own Txn. Caller must
// hold Engine.mu.
func (e *Engine) removeInternal(ctx context.Context, originalID string) error {
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
	docVecStore := e.vectors.GetVectors()
	glob := e.inverted.GetGlobalSeen()
	docToks := e.inverted.GetDocTokens()

	if _, exists := e.idMapping[internalID]; !exists {
		// Not in the delta: it may live in a segment, where removal is a
		// tombstone (the row is marked dead; postings are never edited).
		e.killBase(internalID)
		e.writeGen++
		return nil
	}

	e.unlinkPostings(internalID, docToks[internalID], idxData, idxPhon)
	delete(docVecStore, internalID)
	delete(e.idMapping, internalID)
	delete(e.docText, internalID)
	e.dropAttrsLocked(internalID)
	e.annDeleteLocked(internalID)

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

	e.writeGen++
	return nil
}

// docFragments returns every posting key a document's raw tokens contribute
// to: each token's edge n-grams and its Soundex code. Fragments are a pure
// function of the tokens, so they are recomputed on demand instead of being
// stored per document (the stored copy was the single largest heap consumer
// after the postings themselves).
func docFragments(tokens []string) []string {
	seen := make(map[string]struct{}, len(tokens)*6)
	var out []string
	add := func(f string) {
		if _, dup := seen[f]; !dup {
			seen[f] = struct{}{}
			out = append(out, f)
		}
	}
	for _, token := range tokens {
		for _, frag := range generateEdgeNgrams(token) {
			add(frag)
		}
		if phon := analysis.Soundex(token); phon != "" {
			add(phon)
		}
	}
	return out
}

// unlinkPostings removes id from every lexical posting list its tokens fed.
func (e *Engine) unlinkPostings(id uint64, tokens []string, idxData, idxPhon map[string][]uint64) {
	for _, frag := range docFragments(tokens) {
		if idList, ok := idxData[frag]; ok {
			if idList = removeID(idList, id); len(idList) == 0 {
				delete(idxData, frag)
			} else {
				idxData[frag] = idList
			}
		}
		if idList, ok := idxPhon[frag]; ok {
			if idList = removeID(idList, id); len(idList) == 0 {
				delete(idxPhon, frag)
			} else {
				idxPhon[frag] = idList
			}
		}
	}
}

// GetText returns the original full text last indexed under originalID, and
// whether a document with that ID currently exists in the index.
func (e *Engine) GetText(originalID string) (string, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	h := fnv.New64a()
	h.Write([]byte(originalID))
	internalID := h.Sum64()

	return e.textOf(internalID)
}

// Count returns the number of documents currently held in the index.
func (e *Engine) Count() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.docCountLocked()
}

// Config returns a copy of the engine's configuration, for callers (e.g.
// pkg/zenith's tests) that need to confirm an Open-time Option actually
// reached the engine. A copy, not a pointer, so a caller can't mutate the
// engine's live config through it.
func (e *Engine) Config() config.Config {
	return *e.config
}

// CacheEnabled reports whether the query-result cache is active
// (Config.QueryCacheSize > 0 at construction time).
func (e *Engine) CacheEnabled() bool {
	return e.cache != nil
}

// addInternal journals then applies one document. Used by Add,
// AddWithVectorAttrs, AddBatch.
func (e *Engine) addInternal(ctx context.Context, originalID string, fullText string, preVec []float32, attrs Attrs) error {
	if e.journal != nil {
		if err := e.journal.Put(ctx, []byte(originalID), encodeJournalValue(fullText, preVec, attrs)); err != nil {
			return fmt.Errorf("index: journal write: %w", err)
		}
	}
	return e.applyInternal(ctx, originalID, fullText, preVec, attrs)
}

// applyInternal applies one document to the in-memory delta without
// touching the journal — used directly by AddTransaction, whose caller has
// already durably committed the document via its own Txn.
func (e *Engine) applyInternal(ctx context.Context, originalID string, fullText string, preVec []float32, attrs Attrs) error {
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
			if _, exists := tempWordVectors[t]; !exists && !e.hasWordVector(t) {
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
	wordVecs := e.vectors.GetWordVectors()
	docVecStore := e.vectors.GetVectors()
	glob := e.inverted.GetGlobalSeen()
	docToks := e.inverted.GetDocTokens()

	// Detect a 64-bit doc-ID hash collision: internalID already maps to a
	// *different* originalID. Without this check the second document would
	// silently overwrite the first's postings, vector and BM25 state below.
	if existing := e.origID(internalID); existing != "" && existing != originalID {
		return fmt.Errorf("index: id hash collision: %q and %q both hash to %d", existing, originalID, internalID)
	}

	// Idempotency: remove previous postings for this document.
	if _, exists := e.idMapping[internalID]; exists {
		e.unlinkPostings(internalID, docToks[internalID], idxData, idxPhon)
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
	} else {
		// A replacement of a document that lives in a segment: tombstone the
		// old copy; the new version goes into the delta below.
		e.killBase(internalID)
	}

	e.idMapping[internalID] = originalID
	e.docText[internalID] = fullText
	e.setAttrsLocked(internalID, copyAttrs(attrs))

	if docVec != nil {
		docVecStore[internalID] = VectorEntry{
			Vector:    FloatsToFloat16(docVec),
			Magnitude: ranking.Magnitude(docVec),
		}
		e.annInsertLocked(internalID, docVec, docVecStore[internalID].Vector)
	} else {
		e.annDeleteLocked(internalID)
		// Re-indexing with no usable vector (embedder down, or the caller
		// didn't supply one) must drop any vector left over from a previous
		// version of this document — otherwise semantic search keeps
		// matching content the document no longer has.
		delete(docVecStore, internalID)
	}
	maps.Copy(wordVecs, tempWordVectors)

	seenInDoc := make(map[string]bool)

	for _, token := range rawTokens {
		for _, frag := range generateEdgeNgrams(token) {
			if seenInDoc[frag] {
				continue
			}
			seenInDoc[frag] = true
			idxData[frag] = append(idxData[frag], internalID)
		}

		if phon := analysis.Soundex(token); phon != "" && !seenInDoc[phon] {
			idxPhon[phon] = append(idxPhon[phon], internalID)
			seenInDoc[phon] = true
		}

		// Increment globalSeen reference count; a term new to the whole index
		// (delta and segments) dirties the FST and joins the BK-tree if built.
		if glob[token] == 0 && e.termRefs(token) == 0 {
			if e.bk != nil {
				e.bk.Add(token)
			}
			e.fstDirty = true
		}
		glob[token]++
	}

	docToks[internalID] = append([]string(nil), rawTokens...) // snapshot

	e.bm25.Index(internalID, rawTokens)

	e.writeGen++
	return nil
}

// Search executes a hybrid query.
//
// The query embedding is computed before Engine.mu is taken, for the same
// reason as in Add: Embed is a network/inference call, and running it while
// holding even RLock previously stalled Add/Remove (which need the write
// lock) and, transitively, every other Search queued behind them.
func (e *Engine) Search(ctx context.Context, query string) ([]SearchResponse, error) {
	return e.SearchWithFilter(ctx, query, nil)
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

// neighborCandidate pairs a word with its similarity score for sorting.
type neighborCandidate struct {
	word  string
	score float32
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

var ErrIncompatibleVersion = fmt.Errorf("index: incompatible file version — rebuild the index with the current binary")

// ErrEmbedderMismatch is returned by Load when the saved file's embedder
// identity (model name and/or vector dimension) doesn't match the embedder
// the Engine was constructed with. Loading anyway would silently mix vectors
// from two different embedding spaces into the same VectorStore, producing
// meaningless dot products — see the P0-5 discussion in ROADMAP.md.
var ErrEmbedderMismatch = fmt.Errorf("index: saved file was written with a different embedder — rebuild the index with the current embedder")

// embedderIdentity returns the (name, dimensions) pair recorded in a saved
// file's header for e's current embedder. An embedder that doesn't implement
// embedding.Named (e.g. a caller's custom WithEmbedder type) is recorded as
// "unknown" and is never checked for mismatch on Load — we can't compare
// what we can't identify.
func (e *Engine) embedderIdentity() (name string, dims int) {
	if e.embedder == nil {
		return "none", 0
	}
	dims = e.embedder.Dimensions()
	if n, ok := e.embedder.(embedding.Named); ok {
		return n.Name(), dims
	}
	return "unknown", dims
}

func writeHeaderString(w io.Writer, s string) error {
	var lbuf [4]byte
	binary.BigEndian.PutUint32(lbuf[:], uint32(len(s)))
	if _, err := w.Write(lbuf[:]); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

func readHeaderString(r io.Reader) (string, error) {
	var lbuf [4]byte
	if _, err := io.ReadFull(r, lbuf[:]); err != nil {
		return "", err
	}
	n := binary.BigEndian.Uint32(lbuf[:])
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
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
// per-request namespaces (hundreds of documents), not the persistent index. ExplainIDs computes the
// same signals for a given (bounded) set of documents without the scan.
func (e *Engine) Explain(ctx context.Context, query string) ([]string, []ExplainHit, error) {
	return e.ExplainFiltered(ctx, query, nil)
}

// explainCand is one way a base query term can be satisfied by a document term.
type explainCand struct {
	tok  string
	dist int
	syn  bool
}

// explainPlan is the query-side half of Explain: everything that depends only
// on the query, computed once and then applied per document by hit.
type explainPlan struct {
	base     []string                 // analysed base query terms, deduplicated
	cands    map[string][]explainCand // per base term, its candidates in preference order
	lexTerms []string                 // base terms plus their synonyms, deduplicated (BM25 query)
	queryVec []float32                // nil = no semantic signal
}

// queryEmbeddingForExplain embeds query for the semantic signal; a failing
// embedder degrades to no semantic signal rather than an error.
func (e *Engine) queryEmbeddingForExplain(ctx context.Context, query string) []float32 {
	if e.embedder == nil {
		return nil
	}
	queryVec, err := embedding.EmbedQuery(ctx, e.embedder, query)
	if err != nil {
		slog.Warn("explain: semantic signal unavailable — embedder failed", "error", err)
		return nil
	}
	return queryVec
}

// planExplain analyses query and resolves each base term's exact, synonym and
// fuzzy candidates. Engine.mu held for reading.
func (e *Engine) planExplain(query string, queryVec []float32) *explainPlan {
	var base []string
	if et, ok := e.analyzer.(exactTokenizer); ok {
		base = et.TokenizeExact(query)
	} else {
		for _, t := range e.analyzer.Analyze(query) {
			base = append(base, t.Term)
		}
	}
	base = dedupe(base)

	rank := func(c explainCand) int {
		switch {
		case c.dist == 0 && !c.syn:
			return 0
		case c.syn:
			return 1
		default:
			return 1 + c.dist
		}
	}
	cands := make(map[string][]explainCand, len(base))
	all := append([]string(nil), base...)
	for _, t := range base {
		cs := []explainCand{{tok: t}}
		for _, s := range analysis.Synonyms(t) {
			cs = append(cs, explainCand{tok: s, syn: true})
			all = append(all, s)
		}
		if len([]rune(t)) >= 2 {
			for _, m := range e.fuzzyMatches(t) {
				if m.Distance > 0 {
					cs = append(cs, explainCand{tok: m.Word, dist: m.Distance})
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
	return &explainPlan{base: base, cands: cands, lexTerms: dedupe(all), queryVec: queryVec}
}

// hit computes one document's signals from a membership test over its terms
// and its BM25 score; ok is false when the document has neither a term hit nor
// a positive semantic score. Engine.mu, inverted and vectors held for reading.
func (p *explainPlan) hit(e *Engine, id uint64, has func(string) bool, lexical float64) (ExplainHit, bool) {
	var terms []TermHit
	for _, t := range p.base {
		for _, c := range p.cands[t] {
			if has(c.tok) {
				terms = append(terms, TermHit{Term: t, Matched: c.tok, Dist: c.dist, Synonym: c.syn})
				break
			}
		}
	}
	sem := 0.0
	if len(p.queryVec) > 0 {
		if v := e.vecOf(id); v != nil {
			if s := ranking.DotProduct(p.queryVec, Float16ToFloats(v)); s > 0 {
				sem = s
			}
		}
	}
	if len(terms) == 0 && sem <= 0 {
		return ExplainHit{}, false
	}
	return ExplainHit{ID: e.origID(id), Lexical: lexical, Semantic: sem, Terms: terms}, true
}

// ExplainFiltered is Explain restricted to documents whose attributes satisfy
// f (nil = no restriction).
func (e *Engine) ExplainFiltered(ctx context.Context, query string, f *Filter) ([]string, []ExplainHit, error) {
	pred := f.pred()
	e.mu.RLock()
	defer e.mu.RUnlock()

	p := e.planExplain(query, nil)
	lex := make(map[uint64]float64)
	for _, r := range e.bm25.Query(p.lexTerms) {
		lex[r.DocID] = r.Score
	}
	p.queryVec = e.queryEmbeddingForExplain(ctx, query)

	e.inverted.RLock()
	e.vectors.RLock()
	defer e.vectors.RUnlock()
	defer e.inverted.RUnlock()

	hits := make([]ExplainHit, 0)
	e.eachDocTerms(func(id uint64, has func(string) bool) {
		if pred != nil && !pred(e.attrs[id]) {
			return
		}
		if h, ok := p.hit(e, id, has, lex[id]); ok {
			hits = append(hits, h)
		}
	})
	sortExplainHits(hits)
	return p.base, hits, nil
}

// ExplainIDs computes exactly the signals ExplainFiltered would report, but
// only for the listed documents (caller-supplied IDs, as SearchResponse.ID
// carries them) instead of scanning every document: the result equals
// ExplainFiltered's restricted to ids, in the same order. Its cost is
// proportional to len(ids), not to the size of the index, so it is the way to
// explain a ranked search's already-bounded result list on a large index.
// Unknown or deleted IDs, duplicates, and documents failing f (nil = no
// restriction) are skipped.
func (e *Engine) ExplainIDs(ctx context.Context, query string, ids []string, f *Filter) ([]string, []ExplainHit, error) {
	pred := f.pred()
	// The embedder is fixed at construction, so the query can be embedded
	// before taking the engine lock (as searchUncached does).
	queryVec := e.queryEmbeddingForExplain(ctx, query)

	e.mu.RLock()
	defer e.mu.RUnlock()

	p := e.planExplain(query, queryVec)

	internal := make([]uint64, 0, len(ids))
	seen := make(map[uint64]struct{}, len(ids))
	for _, s := range ids {
		h := fnv.New64a()
		h.Write([]byte(s))
		id := h.Sum64()
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		internal = append(internal, id)
	}
	// Same BM25 as ExplainFiltered's corpus-wide Query, evaluated for these
	// documents only (identical per-document arithmetic and term order).
	lex := e.bm25.ScoreDocsDefault(internal, p.lexTerms)

	e.inverted.RLock()
	e.vectors.RLock()
	defer e.vectors.RUnlock()
	defer e.inverted.RUnlock()

	hits := make([]ExplainHit, 0, len(internal))
	for _, id := range internal {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		has, ok := e.docTermsOf(id)
		if !ok {
			continue
		}
		if pred != nil && !pred(e.attrs[id]) {
			continue
		}
		if h, ok := p.hit(e, id, has, lex[id]); ok {
			hits = append(hits, h)
		}
	}
	sortExplainHits(hits)
	return p.base, hits, nil
}

// sortExplainHits orders hits by number of matched terms, then BM25, then
// semantic score, then ID.
func sortExplainHits(hits []ExplainHit) {
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
}
