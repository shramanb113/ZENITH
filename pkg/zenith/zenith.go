// Package zenith provides an embeddable hybrid search index for Go applications.
//
// ZENITH is to search what SQLite is to databases — zero dependencies, embeds
// in your app, ships in your binary.
//
// Usage:
//
//	db, err := zenith.Open("search.db")   // or ":memory:" for ephemeral
//	defer db.Close()
//	db.Add(ctx, "doc1", "full text content here")
//	results, _ := db.Search(ctx, "content")
//
// All methods are safe for concurrent use by multiple goroutines.
package zenith

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
	"github.com/shramanb113/ZENITH/internal/reranker"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

const maxIDBytes = 512

// estimatedBytesPerDoc approximates per-document heap cost (postings,
// BK-tree, BM25 state, vectors) for WithMemoryLimit. Derived from TestScale's
// 1M-doc run (internal/index/scale_test.go, see bench/BENCHMARK.md): Go heap
// in use 1,575MB after ingesting 1,000,000 docs ≈ 1.6KB/doc. Superseded the
// older ~11KB/doc figure (1,127MB / 100,000 docs from the MS MARCO hybrid
// benchmark), which measured a HeapAlloc delta at 10x smaller scale, so fixed
// overhead (ONNX runtime, cached vocabulary vectors) dominated its per-doc
// average far more than it does at 1M docs. This is an approximation, not
// exact accounting.
const estimatedBytesPerDoc int64 = 1664 // ≈1.6 KiB, rounded up from 1,651.5 B/doc

// DB is a handle to an open ZENITH search index.
// All methods are safe for concurrent use by multiple goroutines.
// Use Open to obtain a *DB; never construct one directly.
type DB struct {
	mu        sync.RWMutex
	closed    atomic.Bool
	closeOnce sync.Once

	engine   *index.Engine
	reranker *reranker.Reranker // nil unless WithReranker(true)
	path     string             // absolute path; empty for :memory:
	lock     *fileLock          // nil for :memory:
	opts     *options
	docWAL   *wal.WAL      // nil for :memory:
	ckptStop chan struct{} // closed to stop background checkpoint goroutine; nil if not running

	// Checkpointing (checkpoint.go).
	walPath     string
	walSeq      int            // highest WAL archive number used; guarded by mu
	walMu       sync.Mutex     // guards walArchives
	walArchives []string       // journal archives not yet deleted, oldest first
	ckptBusy    atomic.Bool    // a checkpoint is in flight
	ckptWG      sync.WaitGroup // background checkpoint goroutines
}

// Open opens or creates a ZENITH index at path.
// Pass ":memory:" for an in-process index that does not persist to disk.
// Each call to Open(":memory:") creates a new independent database.
// To share an index between goroutines, pass the same *DB instance.
func Open(path string, opt ...Option) (*DB, error) {
	if path == "" {
		return nil, fmt.Errorf("zenith: path must not be empty")
	}

	o := defaultOptions()
	for _, fn := range opt {
		if err := fn(o); err != nil {
			return nil, err
		}
	}

	emb, err := buildEmbedder(o)
	if err != nil {
		return nil, err
	}

	cfg := config.DefaultConfig()
	cfg.FuzzyMaxDist = o.fuzzyDistance
	cfg.WordVectors = !o.noWordVectors
	if o.queryCacheSize >= 0 {
		cfg.QueryCacheSize = o.queryCacheSize
	}
	if o.queryCacheTTL > 0 {
		cfg.QueryCacheTTL = o.queryCacheTTL
	}
	if o.queryCacheRedisAddr != "" {
		cfg.QueryCacheRedisAddr = o.queryCacheRedisAddr
	}
	if o.queryCacheNamespace != "" {
		cfg.QueryCacheNamespace = o.queryCacheNamespace
	}
	cfg.QueryCacheSemanticThreshold = o.queryCacheSemanticThreshold
	cfg.ANNThresholdBandPct = o.annThresholdBandPct

	tkz := analysis.NewStandardAnalyzer()
	scorer := ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight)
	eng := index.NewEngine(cfg, emb, scorer, tkz)
	if o.annMinDocs >= 0 {
		eng.SetANNThreshold(o.annMinDocs)
	}
	if o.queryCacheObserver != nil {
		eng.SetCacheObserver(o.queryCacheObserver)
	}

	db := &DB{
		engine: eng,
		opts:   o,
	}

	if o.rerank {
		dir := o.modelsDir
		if dir == "" {
			dir = defaultModelsDir()
		}
		rr, err := reranker.New(o.rerankModel, dir)
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		db.reranker = rr
	}

	if path == ":memory:" {
		return db, nil
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return nil, fmt.Errorf("zenith: %w", err)
	}
	db.path = absPath

	fl, err := acquireLock(absPath)
	if err != nil {
		return nil, err
	}
	db.lock = fl

	// Open (or create) the WAL journal alongside the gob file.
	walPath := absPath + ".wal"
	db.walPath = walPath
	// A crash mid-checkpoint leaves archived journals beside the live one; they
	// hold the oldest records, so they replay first.
	archives, archivedRecords, archiveSeq, err := walOpenArchives(walPath)
	if err != nil {
		fl.release()
		return nil, err
	}
	db.walArchives, db.walSeq = archives, archiveSeq
	walCfg := wal.WALConfig{SyncMode: wal.SyncAlways, Dir: filepath.Dir(absPath)}
	docWAL, liveRecords, err := wal.OpenWAL(walPath, walCfg)
	if err != nil {
		fl.release()
		return nil, fmt.Errorf("zenith: open wal: %w", err)
	}
	db.docWAL = docWAL
	walRecords := append(archivedRecords, liveRecords...)

	// Load the gob snapshot. Corruption-resistant: if the gob is unreadable
	// (but not a version mismatch) and the WAL has records, rebuild from WAL
	// only. If both are missing/empty, surface the error so the caller knows.
	gobErr := eng.Load(absPath)
	if gobErr != nil && !os.IsNotExist(gobErr) {
		if errors.Is(gobErr, index.ErrIncompatibleVersion) {
			_ = docWAL.Close()
			fl.release()
			return nil, ErrIncompatibleVersion
		}
		if errors.Is(gobErr, index.ErrEmbedderMismatch) {
			// Unlike a corrupt gob, this file decodes fine — it was just built
			// with a different embedder. Loading it anyway (or silently
			// rebuilding from the WAL) would mix incompatible vector spaces,
			// so refuse outright rather than falling through to WAL recovery.
			_ = docWAL.Close()
			fl.release()
			return nil, fmt.Errorf("%w: %v", ErrEmbedderMismatch, gobErr)
		}
		if len(walRecords) == 0 {
			// Corrupt gob and no WAL data to recover from — user must rebuild.
			_ = docWAL.Close()
			fl.release()
			return nil, fmt.Errorf("zenith: %w", gobErr)
		}
		// Corrupt gob but WAL has records — rebuild from WAL.
		slog.Warn("zenith: gob corrupt, rebuilding from WAL", "error", gobErr)
	}

	// Replay WAL delta on top of the gob baseline (or as full history if gob
	// was corrupt). Put records written by this binary carry their embedding
	// alongside the text (see encodeWALValue), so replay re-indexes without
	// re-embedding every document — the previous behaviour made crash
	// recovery cost scale with ONNX inference time, not just WAL size.
	for _, r := range walRecords {
		switch r.Op {
		case wal.OpTypePut:
			text, vec, attrs := decodeWALValue(r.Value)
			_ = eng.AddWithVectorAttrs(context.Background(), string(r.Key), text, vec, attrs)
		case wal.OpTypeDelete:
			_ = eng.Remove(context.Background(), string(r.Key))
		}
	}

	// Start background checkpointing if the caller requested it.
	if o.checkpointInterval > 0 {
		db.ckptStop = make(chan struct{})
		go db.checkpointLoop(o.checkpointInterval)
	}

	return db, nil
}

// Add indexes a document. Safe to call with the same id to re-index
// (idempotent — old entries are replaced cleanly).
func (db *DB) Add(ctx context.Context, id, text string) error {
	return db.add(ctx, id, text, nil)
}

// AddWithAttrs is Add plus metadata attributes that Search can restrict on
// via the Filter search option. Re-adding an id replaces its attributes; an
// Add without attributes clears any previously stored ones.
func (db *DB) AddWithAttrs(ctx context.Context, id, text string, attrs Attrs) error {
	ia, err := toIndexAttrs(attrs)
	if err != nil {
		return err
	}
	return db.add(ctx, id, text, ia)
}

func (db *DB) add(ctx context.Context, id, text string, attrs index.Attrs) (err error) {
	if db == nil {
		return errors.New("zenith: Add called on nil DB")
	}
	defer func() {
		if r := recover(); r != nil {
			db.closed.Store(true)
			err = fmt.Errorf("zenith: internal error: %v", r)
		}
	}()

	if err = validateID(id); err != nil {
		return err
	}
	if err = validateText(text); err != nil {
		return err
	}
	text = sanitiseText(text)

	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed.Load() {
		return ErrClosed
	}

	if db.opts.memoryLimitBytes > 0 {
		extra := 1
		if _, exists := db.engine.GetText(id); exists {
			extra = 0 // overwriting an existing document, not growing the index
		}
		if db.estimatedBytesLocked(extra) > db.opts.memoryLimitBytes {
			return ErrIndexFull
		}
	}

	// Embed once, outside the WAL write, and reuse the vector both for the
	// WAL record (so a crash-recovery replay doesn't re-embed) and for
	// indexing (via AddWithVector, instead of Add which would embed again).
	vec := db.engine.EmbedText(ctx, text)

	if db.docWAL != nil {
		if _, err = db.docWAL.Append(ctx, &wal.Record{
			Op: wal.OpTypePut, Key: []byte(id), Value: encodeWALValue(text, vec, attrs),
		}); err != nil {
			return fmt.Errorf("zenith: wal: %w", err)
		}
	}

	if err = db.engine.AddWithVectorAttrs(ctx, id, text, vec, attrs); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	db.checkpointIfWALTooLarge()
	return nil
}

// AddBatch indexes all documents in docs in a single FST rebuild pass.
// More efficient than calling Add in a loop for large inputs.
// NOT atomic — if AddBatch returns an error, some documents may already
// be indexed. Documents are processed in sorted ID order for deterministic results.
func (db *DB) AddBatch(ctx context.Context, docs map[string]string) error {
	return db.addBatch(ctx, docs, nil)
}

// AddBatchWithAttrs is AddBatch plus per-document attributes keyed by id.
// Documents without an entry in attrs are indexed with no attributes.
func (db *DB) AddBatchWithAttrs(ctx context.Context, docs map[string]string, attrs map[string]Attrs) error {
	converted := make(map[string]index.Attrs, len(attrs))
	for id, a := range attrs {
		ia, err := toIndexAttrs(a)
		if err != nil {
			return fmt.Errorf("attrs for %q: %w", id, err)
		}
		converted[id] = ia
	}
	return db.addBatch(ctx, docs, converted)
}

func (db *DB) addBatch(ctx context.Context, docs map[string]string, attrs map[string]index.Attrs) (err error) {
	if db == nil {
		return errors.New("zenith: AddBatch called on nil DB")
	}
	defer func() {
		if r := recover(); r != nil {
			db.closed.Store(true)
			err = fmt.Errorf("zenith: internal error: %v", r)
		}
	}()

	if len(docs) == 0 {
		return nil
	}

	batch := make([]index.BatchDoc, 0, len(docs))
	for id, text := range docs {
		if err = validateID(id); err != nil {
			return err
		}
		if err = validateText(text); err != nil {
			return err
		}
		batch = append(batch, index.BatchDoc{ID: id, Text: sanitiseText(text), Attrs: attrs[id]})
	}

	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed.Load() {
		return ErrClosed
	}

	if db.opts.memoryLimitBytes > 0 {
		newDocs := 0
		for _, d := range batch {
			if _, exists := db.engine.GetText(d.ID); !exists {
				newDocs++
			}
		}
		if db.estimatedBytesLocked(newDocs) > db.opts.memoryLimitBytes {
			return ErrIndexFull
		}
	}

	// Embed once, outside the WAL write, using the same length-bucketed
	// batching AddBatch itself would use — batch.Vector is then already
	// populated, so AddBatch's internal embedDocs skips these entirely.
	texts := make([]string, len(batch))
	for i, d := range batch {
		texts[i] = d.Text
	}
	vecs := db.engine.EmbedTexts(ctx, texts)
	for i := range batch {
		batch[i].Vector = vecs[i]
	}

	if db.docWAL != nil {
		walRecs := make([]*wal.Record, len(batch))
		for i, d := range batch {
			walRecs[i] = &wal.Record{Op: wal.OpTypePut, Key: []byte(d.ID), Value: encodeWALValue(d.Text, d.Vector, d.Attrs)}
		}
		if _, err = db.docWAL.AppendBatch(ctx, walRecs); err != nil {
			return fmt.Errorf("zenith: wal: %w", err)
		}
	}

	if err = db.engine.AddBatch(ctx, batch); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	db.checkpointIfWALTooLarge()
	return nil
}

// Search executes a hybrid lexical + fuzzy + semantic query.
// Returns results sorted by score descending. Returns []Result{} (never nil)
// when no documents match.
func (db *DB) Search(ctx context.Context, query string, opts ...SearchOption) (results []Result, err error) {
	if db == nil {
		return nil, errors.New("zenith: Search called on nil DB")
	}
	defer func() {
		if r := recover(); r != nil {
			db.closed.Store(true)
			results = []Result{}
			err = fmt.Errorf("zenith: internal error: %v", r)
		}
	}()

	so := &searchOptions{limit: db.opts.limit}
	for _, fn := range opts {
		fn(so)
	}

	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed.Load() {
		return nil, ErrClosed
	}

	if so.explain {
		// Run the ranked search first and explain only the documents it
		// returned (at most Config.MaxResults), rather than ExplainFiltered's
		// scan of every document: the signals per document are identical, but
		// the cost is bounded by the candidate list instead of the index size.
		raw, err := db.engine.SearchFilteredWeighted(ctx, query, so.indexFilter(), so.weights)
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		ids := make([]string, len(raw))
		for i, r := range raw {
			ids[i] = r.ID
		}
		terms, hits, err := db.engine.ExplainIDs(ctx, query, ids, so.indexFilter())
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		out := buildExplained(terms, hits, raw, 0)
		db.sortResultsByAttribute(out, so.sortField, so.sortDesc)
		return truncate(out, so.limit), nil
	}

	raw, err := db.engine.SearchFilteredWeighted(ctx, query, so.indexFilter(), so.weights)
	if err != nil {
		return nil, fmt.Errorf("zenith: %w", err)
	}

	if db.reranker != nil {
		raw = db.reranker.Rerank(ctx, query, raw, db.engine.GetText)
	}

	out := buildResults(raw, 0)
	db.sortResultsByAttribute(out, so.sortField, so.sortDesc)
	return truncate(out, so.limit), nil
}

// estimatedBytesLocked returns the projected heap usage after adding
// extraDocs new documents. Caller must hold db.mu.
func (db *DB) estimatedBytesLocked(extraDocs int) int64 {
	return int64(db.engine.Count()+extraDocs) * estimatedBytesPerDoc
}

// Get returns the original text last indexed under id, and whether a
// document with that id currently exists.
func (db *DB) Get(id string) (text string, found bool, err error) {
	if db == nil {
		return "", false, errors.New("zenith: Get called on nil DB")
	}
	defer func() {
		if r := recover(); r != nil {
			db.closed.Store(true)
			err = fmt.Errorf("zenith: internal error: %v", r)
		}
	}()

	if id == "" {
		return "", false, ErrInvalidID
	}

	db.mu.RLock()
	defer db.mu.RUnlock()
	if db.closed.Load() {
		return "", false, ErrClosed
	}

	text, found = db.engine.GetText(id)
	return text, found, nil
}

// Delete removes a document from the index. Idempotent — deleting a
// non-existent id returns nil.
func (db *DB) Delete(ctx context.Context, id string) (err error) {
	if db == nil {
		return errors.New("zenith: Delete called on nil DB")
	}
	defer func() {
		if r := recover(); r != nil {
			db.closed.Store(true)
			err = fmt.Errorf("zenith: internal error: %v", r)
		}
	}()

	if id == "" {
		return ErrInvalidID
	}

	db.mu.Lock()
	defer db.mu.Unlock()
	if db.closed.Load() {
		return ErrClosed
	}

	if db.docWAL != nil {
		if _, err = db.docWAL.Append(ctx, &wal.Record{
			Op: wal.OpTypeDelete, Key: []byte(id),
		}); err != nil {
			return fmt.Errorf("zenith: wal: %w", err)
		}
	}

	if err = db.engine.Remove(ctx, id); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	db.checkpointIfWALTooLarge()
	return nil
}

// WAL Put-record Value tags. walValueHasVector (1) is the P0-4 layout
// [tag][4B textLen][text][4B vecLen][vec]; walValueHasAttrs (2) appends
// [4B attrsLen][attrs JSON] for P1 metadata. Both remain decodable; anything
// else is treated as legacy raw text (a pre-P0-4 WAL).
const (
	walValueHasVector byte = 1
	walValueHasAttrs  byte = 2
)

// encodeWALValue packs text, its (possibly nil) embedding and its (possibly
// nil) attributes into a WAL record Value, so a crash-recovery replay can
// re-index without calling the embedder again and without losing metadata.
func encodeWALValue(text string, vec []float32, attrs index.Attrs) []byte {
	textBytes := []byte(text)
	var attrJSON []byte
	if len(attrs) > 0 {
		attrJSON, _ = json.Marshal(attrs) // Attrs is plain data; Marshal cannot fail
	}
	buf := make([]byte, 1+4+len(textBytes)+4+len(vec)*4+4+len(attrJSON))
	buf[0] = walValueHasAttrs
	binary.BigEndian.PutUint32(buf[1:5], uint32(len(textBytes)))
	off := 5
	off += copy(buf[off:], textBytes)
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(vec)))
	off += 4
	for _, f := range vec {
		binary.BigEndian.PutUint32(buf[off:off+4], math.Float32bits(f))
		off += 4
	}
	binary.BigEndian.PutUint32(buf[off:off+4], uint32(len(attrJSON)))
	off += 4
	copy(buf[off:], attrJSON)
	return buf
}

// decodeWALValue is the inverse of encodeWALValue. A value that is not
// well-formed (a pre-P0-4 raw-text WAL, or truncated/corrupt) is treated as
// legacy plain text with no stored vector or attributes, so replay falls
// back to re-embedding instead of failing outright.
func decodeWALValue(raw []byte) (text string, vec []float32, attrs index.Attrs) {
	if len(raw) < 5 || (raw[0] != walValueHasVector && raw[0] != walValueHasAttrs) {
		return string(raw), nil, nil
	}
	textLen := binary.BigEndian.Uint32(raw[1:5])
	off := 5
	if uint64(off)+uint64(textLen)+4 > uint64(len(raw)) {
		return string(raw), nil, nil
	}
	text = string(raw[off : off+int(textLen)])
	off += int(textLen)
	vecLen := binary.BigEndian.Uint32(raw[off : off+4])
	off += 4
	vecEnd := uint64(off) + uint64(vecLen)*4
	if vecEnd > uint64(len(raw)) {
		return text, nil, nil
	}
	if raw[0] == walValueHasVector && vecEnd != uint64(len(raw)) {
		return text, nil, nil
	}
	vec = make([]float32, vecLen)
	for i := range vec {
		vec[i] = math.Float32frombits(binary.BigEndian.Uint32(raw[off : off+4]))
		off += 4
	}
	if raw[0] == walValueHasAttrs {
		if off+4 > len(raw) {
			return text, vec, nil
		}
		attrLen := int(binary.BigEndian.Uint32(raw[off : off+4]))
		off += 4
		if attrLen > 0 && off+attrLen == len(raw) {
			var a index.Attrs
			if json.Unmarshal(raw[off:], &a) == nil {
				attrs = a
			}
		}
	}
	return text, vec, attrs
}

// defaultWALCheckpointThreshold bounds crash-recovery cost by default, even
// when the caller never sets WithCheckpointInterval: once the WAL grows past
// this many bytes, the next Add/AddBatch/Delete triggers a synchronous
// checkpoint. 64MB matches the MemTableMaxSize convention used elsewhere in
// this codebase (see CLAUDE.md) rather than measured WAL-replay cost.
var defaultWALCheckpointThreshold uint64 = 64 * 1024 * 1024 // a var so tests can shrink it

// checkpointIfWALTooLarge starts a checkpoint once the WAL exceeds
// defaultWALCheckpointThreshold. Must be called with db.mu held (write lock).
// The checkpoint runs in the background (checkpoint.go): only its O(1) freeze
// happens here, so the Add that crossed the threshold — and every search — is
// not held up while the segment is written. A failure is logged, not returned:
// the document that triggered the check is already safely durable in the
// (still-growing) WAL, so a bookkeeping checkpoint failing must not fail the
// caller's Add/AddBatch/Delete.
func (db *DB) checkpointIfWALTooLarge() {
	if db.docWAL == nil || db.docWAL.Size() < defaultWALCheckpointThreshold {
		return
	}
	db.startCheckpointAsyncLocked()
}

// checkpointLoop runs as a background goroutine when WithCheckpointInterval is set.
// It periodically checkpoints, bounding WAL growth.
func (db *DB) checkpointLoop(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-db.ckptStop:
			return
		case <-ticker.C:
			db.mu.Lock()
			if !db.closed.Load() {
				db.startCheckpointAsyncLocked()
			}
			db.mu.Unlock()
		}
	}
}

// Close flushes and closes the database. Idempotent — safe to call twice.
// Returns the save error if persistence fails (disk full, permissions, etc.).
func (db *DB) Close() (err error) {
	if db == nil {
		return nil
	}

	db.closeOnce.Do(func() {
		defer func() {
			if r := recover(); r != nil {
				db.closed.Store(true)
				err = fmt.Errorf("zenith: internal error during close: %v", r)
			}
		}()

		// Stop the background checkpoint goroutine before acquiring the lock.
		if db.ckptStop != nil {
			close(db.ckptStop)
			db.ckptStop = nil
		}

		db.mu.Lock()
		defer db.mu.Unlock()

		db.closed.Store(true)

		// A background checkpoint needs neither db.mu nor the WAL handle, so it
		// can be waited for here.
		db.ckptWG.Wait()

		if db.path != "" {
			if saveErr := db.checkpointSyncLocked(); saveErr != nil {
				err = saveErr
			}
		}

		if db.docWAL != nil {
			_ = db.docWAL.Close()
			db.docWAL = nil
		}

		// Persist the ANN graph so the next open does not rebuild it (best
		// effort: without it the open just rebuilds), then unmap the segment
		// files (required on Windows before they can be deleted or replaced, and
		// releases address space everywhere).
		if db.engine != nil && db.path != "" {
			if saveErr := db.engine.SaveANN(); saveErr != nil {
				slog.Warn("zenith: could not save the ANN graph", "error", saveErr)
			}
		}
		if db.engine != nil {
			if closeErr := db.engine.Close(); closeErr != nil && err == nil {
				err = closeErr
			}
		}

		if db.reranker != nil {
			db.reranker.Close()
			db.reranker = nil
		}

		if db.lock != nil {
			db.lock.release()
			db.lock = nil
		}
	})
	return err
}

// --- Input validation ---

func validateID(id string) error {
	if id == "" {
		return ErrInvalidID
	}
	if len(id) > maxIDBytes {
		return ErrIDTooLong
	}
	for _, r := range id {
		if r < 0x20 || r == unicode.ReplacementChar {
			return ErrInvalidID
		}
	}
	if strings.Contains(id, "||") {
		return ErrInvalidID
	}
	return nil
}

func validateText(text string) error {
	if strings.TrimSpace(text) == "" {
		return ErrEmptyDocument
	}
	return nil
}

func sanitiseText(text string) string {
	text = strings.ToValidUTF8(text, "")
	text = strings.Map(func(r rune) rune {
		if r == 0 {
			return -1
		}
		return r
	}, text)
	return text
}

// --- Result construction ---

// truncate slices results down to limit, or returns it unchanged when limit
// is 0 (unlimited) or already satisfied.
func truncate(results []Result, limit int) []Result {
	if limit > 0 && len(results) > limit {
		return results[:limit]
	}
	return results
}

// sortResultsByAttribute replaces results' order in place with the order
// Engine.SortByAttribute gives for field/desc (a no-op when field is empty).
// It round-trips through index.SearchResponse rather than duplicating
// SortByAttribute's comparator here, since results (already deduplicated by
// buildResults/buildExplained) and the round-tripped slice share the same
// unique document IDs.
func (db *DB) sortResultsByAttribute(results []Result, field string, desc bool) {
	if field == "" || len(results) < 2 {
		return
	}
	tmp := make([]index.SearchResponse, len(results))
	byID := make(map[string]Result, len(results))
	for i, r := range results {
		tmp[i] = index.SearchResponse{ID: r.ID, Score: r.Score}
		byID[r.ID] = r
	}
	db.engine.SortByAttribute(tmp, field, desc)
	for i, t := range tmp {
		results[i] = byID[t.ID]
	}
}

func buildResults(raw []index.SearchResponse, limit int) []Result {
	if len(raw) == 0 {
		return []Result{}
	}

	maxScore := raw[0].Score
	for _, r := range raw[1:] {
		if r.Score > maxScore {
			maxScore = r.Score
		}
	}

	type entry struct {
		score  float64
		chunks []Chunk
	}
	seen := make(map[string]*entry, len(raw))

	for _, r := range raw {
		norm := normaliseScore(r.Score, maxScore)
		origID, chunk := parseChunkID(r.ID)

		if e, exists := seen[origID]; exists {
			if norm > e.score {
				e.score = norm
			}
			if chunk != nil {
				e.chunks = append(e.chunks, *chunk)
			}
		} else {
			e := &entry{score: norm}
			if chunk != nil {
				e.chunks = []Chunk{*chunk}
			}
			seen[origID] = e
		}
	}

	results := make([]Result, 0, len(seen))
	for id, e := range seen {
		results = append(results, Result{
			ID:     id,
			Score:  e.score,
			Chunks: e.chunks,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].ID < results[j].ID
	})

	if limit > 0 && len(results) > limit {
		results = results[:limit]
	}
	return results
}

func buildExplained(terms []string, hits []index.ExplainHit, raw []index.SearchResponse, limit int) []Result {
	scores := make(map[string]float64, len(raw))
	for _, r := range buildResults(raw, 0) {
		scores[r.ID] = r.Score
	}
	out := make([]Result, 0, len(hits))
	for _, h := range hits {
		tm := make([]TermMatch, len(h.Terms))
		for i, t := range h.Terms {
			tm[i] = TermMatch{Term: t.Term, Matched: t.Matched, Dist: t.Dist, Synonym: t.Synonym}
		}
		out = append(out, Result{ID: h.ID, Score: scores[h.ID], Signals: &Signals{
			QueryTerms: append([]string(nil), terms...), Lexical: h.Lexical, Semantic: h.Semantic, Terms: tm,
		}})
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out
}

func normaliseScore(score, maxScore float64) float64 {
	if maxScore <= 0 {
		return 0
	}
	n := score / maxScore
	if math.IsNaN(n) || math.IsInf(n, 0) || n < 0 {
		return 0
	}
	if n > 1 {
		return 1
	}
	return n
}

// parseChunkID splits an internal chunk ID into the original document ID
// and optional Chunk metadata. Chunk IDs use "||" as separator:
//
//	"docID||p3||c1||text||10.00,20.00,400.00,15.00"
//
// Documents without chunks return the ID unchanged and a nil Chunk.
func parseChunkID(rawID string) (string, *Chunk) {
	idx := strings.Index(rawID, "||")
	if idx < 0 {
		return rawID, nil
	}
	origID := rawID[:idx]
	rest := strings.Split(rawID[idx+2:], "||")

	c := &Chunk{}
	if len(rest) >= 1 && strings.HasPrefix(rest[0], "p") {
		c.Page, _ = strconv.Atoi(rest[0][1:])
	}
	if len(rest) >= 2 && strings.HasPrefix(rest[1], "c") {
		c.Index, _ = strconv.Atoi(rest[1][1:])
	}
	if len(rest) >= 4 {
		coords := strings.Split(rest[3], ",")
		if len(coords) == 4 {
			v0, _ := strconv.ParseFloat(coords[0], 32)
			v1, _ := strconv.ParseFloat(coords[1], 32)
			v2, _ := strconv.ParseFloat(coords[2], 32)
			v3, _ := strconv.ParseFloat(coords[3], 32)
			c.BboxX, c.BboxY = float32(v0), float32(v1)
			c.BboxW, c.BboxH = float32(v2), float32(v3)
		}
	}
	return origID, c
}

// buildEmbedder constructs the embedder based on options.
// Falls back to deterministic (hash-based) embeddings if ONNX is unavailable.
func buildEmbedder(o *options) (embedding.Embedder, error) {
	if o.embedder != nil {
		return o.embedder, nil
	}
	if o.model != "" && !o.bm25Only {
		dir := o.modelsDir
		if dir == "" {
			dir = defaultModelsDir()
		}
		m, err := localembedder.NewByID(o.model, dir)
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		if o.cacheSize > 0 {
			if cached, err := embedding.NewCachingEmbedder(m, o.cacheSize); err == nil {
				return cached, nil
			}
		}
		return m, nil
	}
	if o.bm25Only {
		// Return nil vectors so the engine skips vector storage and vectorPass
		// entirely — pure lexical (BM25 + n-gram + fuzzy) search only.
		return &nullEmbedder{}, nil
	}
	if localEmb, err := localembedder.New(); err == nil {
		if o.cacheSize > 0 {
			if cached, err := embedding.NewCachingEmbedder(localEmb, o.cacheSize); err == nil {
				return cached, nil
			}
		}
		return localEmb, nil
	}
	return embedding.NewDeterministicEmbedder(384), nil
}

// defaultModelsDir is where `zenith models pull` installs non-bundled models.
func defaultModelsDir() string {
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".zenith", "models")
	}
	return "models"
}

// nullEmbedder returns nil vectors, causing the engine to skip vector indexing
// and vector search. Used by WithBM25Only() to guarantee pure lexical mode.
type nullEmbedder struct{}

func (n *nullEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, nil
}

func (n *nullEmbedder) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}

func (n *nullEmbedder) Dimensions() int { return 0 }

// Name identifies this embedder for index-file compatibility checks (see
// embedding.Named). BM25-only mode never touches vectors, so it's recorded
// distinctly rather than as an absent/unknown identity.
func (n *nullEmbedder) Name() string { return "none:bm25-only" }
