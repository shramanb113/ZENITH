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
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

const maxIDBytes = 512

// estimatedBytesPerDoc approximates per-document heap cost (postings,
// BK-tree, BM25 state, vectors) for WithMemoryLimit. Derived from the
// measured hybrid-mode heap delta in bench/BENCHMARK.md: 1,127MB / 100,000
// docs ≈ 11KB/doc. This is an approximation, not exact accounting.
const estimatedBytesPerDoc int64 = 11 * 1024

// DB is a handle to an open ZENITH search index.
// All methods are safe for concurrent use by multiple goroutines.
// Use Open to obtain a *DB; never construct one directly.
type DB struct {
	mu        sync.RWMutex
	closed    atomic.Bool
	closeOnce sync.Once

	engine   *index.Engine
	path     string    // absolute path; empty for :memory:
	lock     *fileLock // nil for :memory:
	opts     *options
	docWAL   *wal.WAL      // nil for :memory:
	ckptStop chan struct{} // closed to stop background checkpoint goroutine; nil if not running
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

	emb := buildEmbedder(o)

	cfg := config.DefaultConfig()
	cfg.FuzzyMaxDist = o.fuzzyDistance
	cfg.WordVectors = !o.noWordVectors

	tkz := analysis.NewStandardAnalyzer()
	scorer := ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight)
	eng := index.NewEngine(cfg, emb, scorer, tkz)

	db := &DB{
		engine: eng,
		opts:   o,
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
	walCfg := wal.WALConfig{SyncMode: wal.SyncAlways, Dir: filepath.Dir(absPath)}
	docWAL, walRecords, err := wal.OpenWAL(walPath, walCfg)
	if err != nil {
		fl.release()
		return nil, fmt.Errorf("zenith: open wal: %w", err)
	}
	db.docWAL = docWAL

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
		if len(walRecords) == 0 {
			// Corrupt gob and no WAL data to recover from — user must rebuild.
			_ = docWAL.Close()
			fl.release()
			return nil, fmt.Errorf("zenith: %w", gobErr)
		}
		// Corrupt gob but WAL has records — rebuild from WAL.
		slog.Warn("zenith: gob corrupt, rebuilding from WAL", "error", gobErr)
	}

	// Replay WAL delta on top of the gob baseline (or as full history if gob was corrupt).
	for _, r := range walRecords {
		switch r.Op {
		case wal.OpTypePut:
			_ = eng.Add(context.Background(), string(r.Key), string(r.Value))
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
func (db *DB) Add(ctx context.Context, id, text string) (err error) {
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

	if db.docWAL != nil {
		if _, err = db.docWAL.Append(ctx, &wal.Record{
			Op: wal.OpTypePut, Key: []byte(id), Value: []byte(text),
		}); err != nil {
			return fmt.Errorf("zenith: wal: %w", err)
		}
	}

	if err = db.engine.Add(ctx, id, text); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	return nil
}

// AddBatch indexes all documents in docs in a single FST rebuild pass.
// More efficient than calling Add in a loop for large inputs.
// NOT atomic — if AddBatch returns an error, some documents may already
// be indexed. Documents are processed in sorted ID order for deterministic results.
func (db *DB) AddBatch(ctx context.Context, docs map[string]string) (err error) {
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
		batch = append(batch, index.BatchDoc{ID: id, Text: sanitiseText(text)})
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

	if db.docWAL != nil {
		walRecs := make([]*wal.Record, len(batch))
		for i, d := range batch {
			walRecs[i] = &wal.Record{Op: wal.OpTypePut, Key: []byte(d.ID), Value: []byte(d.Text)}
		}
		if _, err = db.docWAL.AppendBatch(ctx, walRecs); err != nil {
			return fmt.Errorf("zenith: wal: %w", err)
		}
	}

	if err = db.engine.AddBatch(ctx, batch); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
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
		terms, hits, err := db.engine.Explain(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		raw, err := db.engine.Search(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("zenith: %w", err)
		}
		return buildExplained(terms, hits, raw, so.limit), nil
	}

	raw, err := db.engine.Search(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("zenith: %w", err)
	}

	return buildResults(raw, so.limit), nil
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
	return nil
}

// checkpoint saves the gob snapshot and resets the WAL.
// MUST be called with db.mu held (write lock).
func (db *DB) checkpoint() error {
	if db.path == "" || db.docWAL == nil {
		return nil
	}
	if err := db.engine.Save(db.path); err != nil {
		return fmt.Errorf("zenith: %w", err)
	}
	if err := db.docWAL.Reset(); err != nil {
		slog.Warn("zenith: WAL reset failed after checkpoint", "error", err)
	}
	return nil
}

// checkpointLoop runs as a background goroutine when WithCheckpointInterval is set.
// It periodically saves the gob and resets the WAL, bounding WAL growth.
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
				if err := db.checkpoint(); err != nil {
					slog.Warn("zenith: background checkpoint failed", "error", err)
				}
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

		if db.path != "" {
			if saveErr := db.checkpoint(); saveErr != nil {
				err = saveErr
			}
		}

		if db.docWAL != nil {
			_ = db.docWAL.Close()
			db.docWAL = nil
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
func buildEmbedder(o *options) embedding.Embedder {
	if o.embedder != nil {
		return o.embedder
	}
	if o.bm25Only {
		// Return nil vectors so the engine skips vector storage and vectorPass
		// entirely — pure lexical (BM25 + n-gram + fuzzy) search only.
		return &nullEmbedder{}
	}
	if localEmb, err := localembedder.New(); err == nil {
		if o.cacheSize > 0 {
			if cached, err := embedding.NewCachingEmbedder(localEmb, o.cacheSize); err == nil {
				return cached
			}
		}
		return localEmb
	}
	return embedding.NewDeterministicEmbedder(384)
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
