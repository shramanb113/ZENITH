// Engine orchestrator
package storage

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/storage/compaction"
	"github.com/shramanb113/ZENITH/internal/storage/memtable"
	"github.com/shramanb113/ZENITH/internal/storage/sstable"
	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

// Engine orchestrates the full LSM storage pipeline:
//
//	Write path:  WAL → active MemTable → (freeze) → SSTable flush
//	Read path:   active MemTable → immutable MemTables → SSTables (via Bloom + index)
//	FST:         rebuilt from global term vocabulary after every SSTable flush
//	             and on explicit Save() — gives O(log n) prefix search
//	Compaction:  background leveled compaction via Compactor
//
// The Engine is safe for concurrent use. Multiple goroutines may call Put,
// Delete, and Get simultaneously. Flush is serialised via the GroupCommitter.
type Engine struct {
	cfg EngineConfig

	// WAL — written before every MemTable mutation for crash safety.
	walFile *wal.WAL

	// Active MemTable — receives all current writes.
	mu     sync.RWMutex
	active *memtable.MemTable

	// Immutable MemTables waiting to be flushed. In practice this slice
	// rarely holds more than one entry; it grows only if flushing falls
	// behind write rate.
	immutable []*memtable.MemTable

	// SSTable flush pipeline — group committer batches concurrent flushes.
	committer  *sstable.GroupCommitter
	sstCounter atomic.Uint64

	// Leveled compactor — runs in the background, triggered on each flush.
	compactor *compaction.Compactor

	// manifestMu serialises persistManifest so concurrent flush/compaction
	// goroutines can't interleave writes to the on-disk manifest.
	manifestMu sync.Mutex

	// FST dictionary — rebuilt from the term vocabulary after every flush.
	// Provides O(log n) exact lookup and prefix search over indexed terms.
	fst *analysis.FSTDictionary

	// Global term vocabulary — accumulated across all indexed documents.
	// Mirrors what the index/engine.go's InvertedIndex.globalSeen holds,
	// but owned here for the storage-layer FST rebuild.
	vocabMu sync.RWMutex
	vocab   map[string]struct{}

	// journalRecords holds WAL records recovered on Open.
	// The server reads these once to replay document mutations into the index
	// engine after loading a gob snapshot. Cleared by Checkpoint.
	journalRecords []wal.Record

	// Lifecycle
	closeOnce sync.Once
	closed    chan struct{}
	flushWg   sync.WaitGroup
}

// EngineConfig holds storage engine parameters.
type EngineConfig struct {
	// WALPath is the path to the WAL file (e.g. "./data/wal/zenith.wal").
	WALPath string

	// SSTDir is the directory where SSTable files are written.
	SSTDir string

	// MemTableMaxSize is the byte threshold at which a MemTable is frozen
	// and queued for flushing. Default: 64MB.
	MemTableMaxSize int64

	// CommitWindow is the group-committer batch window. Default: 4ms.
	CommitWindow time.Duration

	// WALConfig is passed directly to wal.OpenWAL.
	WALConfig wal.WALConfig

	// CompactorConfig tunes the background leveled compactor.
	// Dir is overridden to match SSTDir at Open time.
	CompactorConfig compaction.CompactorConfig

	// FSTPath is the on-disk path for the storage-layer FST file.
	// When set, RebuildFST writes the FST atomically to this path and
	// reopens it via memory mapping so the vocabulary dict stays off-heap.
	// On Open, if the file exists it is loaded directly (fast startup).
	FSTPath string
}

// DefaultEngineConfig returns a production-ready config.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		WALPath:         "./data/wal/zenith.wal",
		SSTDir:          "./data/sst",
		MemTableMaxSize: 64 * 1024 * 1024, // 64 MB
		CommitWindow:    4 * time.Millisecond,
		WALConfig: wal.WALConfig{
			SyncMode: wal.SyncAlways,
			Dir:      "./data/wal",
		},
		CompactorConfig: compaction.CompactorConfig{
			L0Threshold:        4,
			LevelSizeBase:      10 * 1024 * 1024, // 10 MB
			LevelSizeMult:      10,
			MaxLevels:          7,
			CompactionInterval: 30 * time.Second,
		},
		FSTPath: "./data/terms.fst",
	}
}

// Open creates or recovers a storage Engine.
//
// On first open with no existing WAL, a fresh MemTable and WAL are created.
// On recovery, the WAL is replayed into a new MemTable before Open returns.
// The caller must call Close() when done to flush buffers and sync the WAL.
func Open(cfg EngineConfig) (*Engine, error) {
	if err := os.MkdirAll(filepath.Dir(cfg.WALPath), 0755); err != nil {
		return nil, fmt.Errorf("storage: create wal dir: %w", err)
	}
	if err := os.MkdirAll(cfg.SSTDir, 0755); err != nil {
		return nil, fmt.Errorf("storage: create sst dir: %w", err)
	}

	walFile, records, err := wal.OpenWAL(cfg.WALPath, cfg.WALConfig)
	if err != nil {
		return nil, fmt.Errorf("storage: open wal: %w", err)
	}

	e := &Engine{
		cfg:            cfg,
		walFile:        walFile,
		active:         memtable.NewMemTable(cfg.MemTableMaxSize),
		fst:            analysis.NewFSTDictionary(),
		vocab:          make(map[string]struct{}),
		closed:         make(chan struct{}),
		journalRecords: records,
	}

	e.committer = sstable.NewGroupCommitter(cfg.CommitWindow, e.nextSSTPath)

	// Start the leveled compactor. Dir is pinned to SSTDir so compacted files
	// land in the same directory as flushed SSTables.
	cfg.CompactorConfig.Dir = cfg.SSTDir
	e.compactor = compaction.NewCompactor(cfg.CompactorConfig)
	e.compactor.SetOnChange(e.persistManifest)

	// Rediscover SSTables left behind by a previous run BEFORE replaying the
	// WAL or starting the compactor: without this, sstCounter restarts at 0
	// and the very next flush overwrites the previous run's first SSTable
	// file, and the compactor has no idea those files even exist.
	if err := e.rediscoverSSTables(); err != nil {
		slog.Warn("storage: SSTable rediscovery failed, starting with an empty set", "error", err)
	}

	// Replay WAL records. Replay must be able to rotate the MemTable exactly
	// like a normal write does: if a large WAL simply exceeds MemTableMaxSize
	// partway through, the naive approach of replaying into one MemTable
	// leaves it frozen when Open returns, and Put/Delete on a table that
	// starts out frozen fail forever (see rotateLocked's doc comment).
	for _, r := range records {
		if err := e.applyReplayRecord(r); err != nil {
			slog.Warn("WAL replay: skipping bad record", "seq", r.Seq, "error", err)
		}
	}
	slog.Info("WAL replayed", "records", len(records))

	e.compactor.Run()

	// Load FST from disk for instant startup — no vocab rebuild needed.
	// The file was written by the last RebuildFST call and reflects the
	// vocabulary accumulated before the previous shutdown.
	if cfg.FSTPath != "" {
		if err := e.fst.OpenFromFile(cfg.FSTPath); err == nil {
			slog.Info("Storage FST loaded from disk", "path", cfg.FSTPath, "terms", e.fst.Size())
		}
	}

	return e, nil
}

// applyReplayRecord applies one recovered WAL record directly to the
// MemTable stack during Open, rotating the active table first if it is
// already frozen (see rotateLocked).
func (e *Engine) applyReplayRecord(r wal.Record) error {
	e.mu.Lock()
	if e.active.IsFrozen() {
		e.rotateLocked()
	}
	err := memtable.ApplyRecord(e.active, r)
	e.mu.Unlock()
	return err
}

// ── Write path ────────────────────────────────────────────────────────────────

// Put writes key→value to the WAL and then the active MemTable.
// If the MemTable becomes frozen after this write, a background flush
// is triggered automatically.
func (e *Engine) Put(ctx context.Context, key, value []byte) error {
	if e.isClosed() {
		return errors.New("storage: engine is closed")
	}
	if len(key) == 0 {
		// Validate before the WAL append, not after: once a record is
		// durable in the WAL it MUST be applicable to the MemTable, or a
		// crash-replay will keep hitting the same unusable record forever.
		return memtable.ErrKeyEmpty
	}

	seq, err := e.walFile.Append(ctx, &wal.Record{
		Op:    wal.OpTypePut,
		Key:   key,
		Value: value,
	})
	if err != nil {
		return fmt.Errorf("storage: wal append: %w", err)
	}
	_ = seq

	e.mu.Lock()
	// The active MemTable may already be frozen (e.g. it started that way
	// after a WAL replay that hit the size cap) — rotate before writing
	// instead of failing forever. See rotateLocked.
	if e.active.IsFrozen() {
		e.rotateLocked()
	}
	err = e.active.Put(key, value)
	frozen := e.active.IsFrozen()
	e.mu.Unlock()

	if err != nil {
		return fmt.Errorf("storage: memtable put: %w", err)
	}

	if frozen {
		e.rotateAndFlush()
	}

	return nil
}

// Delete writes a tombstone to the WAL and MemTable.
func (e *Engine) Delete(ctx context.Context, key []byte) error {
	if e.isClosed() {
		return errors.New("storage: engine is closed")
	}
	if len(key) == 0 {
		return memtable.ErrKeyEmpty
	}

	_, err := e.walFile.Append(ctx, &wal.Record{
		Op:  wal.OpTypeDelete,
		Key: key,
	})
	if err != nil {
		return fmt.Errorf("storage: wal append: %w", err)
	}

	e.mu.Lock()
	if e.active.IsFrozen() {
		e.rotateLocked()
	}
	err = e.active.Delete(key)
	frozen := e.active.IsFrozen()
	e.mu.Unlock()

	if err != nil {
		return fmt.Errorf("storage: memtable delete: %w", err)
	}

	if frozen {
		e.rotateAndFlush()
	}

	return nil
}

// AddTerms adds stemmed terms to the vocabulary and marks the FST as stale.
// Called by index/engine.go after each document is indexed.
// The FST is rebuilt lazily on the next flush or explicit RebuildFST() call.
func (e *Engine) AddTerms(terms []string) {
	e.vocabMu.Lock()
	for _, t := range terms {
		e.vocab[t] = struct{}{}
	}
	e.vocabMu.Unlock()
}

// ── Read path ─────────────────────────────────────────────────────────────────

// Get returns the value for key, searching:
//  1. active MemTable
//  2. immutable MemTables (newest first)
//  3. SSTables — NOT implemented here yet; returns (nil, false) if not in memory.
//     The index/engine.go layer handles SSTable reads via Reader directly.
func (e *Engine) Get(key []byte) ([]byte, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// 1. Active MemTable. A tombstone here must stop the search outright —
	// it shadows whatever value an older immutable MemTable might still hold
	// for the same key (a plain MemTable.Get can't distinguish "absent, keep
	// looking" from "tombstoned, stop", so GetRaw is used instead).
	if val, exists, deleted := e.active.GetRaw(key); exists {
		if deleted {
			return nil, false
		}
		return val, true
	}

	// 2. Immutable MemTables (newest first) — same tombstone-stops-the-search rule.
	for i := len(e.immutable) - 1; i >= 0; i-- {
		if val, exists, deleted := e.immutable[i].GetRaw(key); exists {
			if deleted {
				return nil, false
			}
			return val, true
		}
	}

	return nil, false
}

// ── FST ───────────────────────────────────────────────────────────────────────

// Contains returns true if term exists in the FST dictionary.
// Returns false if the FST has not been built yet — use AddTerms + RebuildFST.
func (e *Engine) Contains(term string) bool {
	return e.fst.Contains(term)
}

// PrefixSearch returns up to maxResults terms beginning with prefix.
// Returns nil if the FST is not built or no terms match.
func (e *Engine) PrefixSearch(prefix string, maxResults int) ([]string, error) {
	return e.fst.PrefixSearch(prefix, maxResults)
}

// RebuildFST rebuilds the FST from the current vocabulary snapshot.
// If FSTPath is set, the FST is written atomically to disk and reopened via
// memory mapping. Otherwise, the FST is kept in-memory.
// Called automatically after every SSTable flush and on Close.
func (e *Engine) RebuildFST() error {
	e.vocabMu.RLock()
	terms := make([]string, 0, len(e.vocab))
	for t := range e.vocab {
		terms = append(terms, t)
	}
	e.vocabMu.RUnlock()

	var err error
	if e.cfg.FSTPath != "" {
		err = e.fst.BuildToFile(terms, e.cfg.FSTPath)
	} else {
		err = e.fst.Build(terms)
	}
	if err != nil {
		return fmt.Errorf("storage: rebuild fst: %w", err)
	}

	slog.Info("FST rebuilt", "terms", len(terms))
	return nil
}

// ── Flush pipeline ────────────────────────────────────────────────────────────

// rotateAndFlush freezes the active MemTable, creates a new one, and
// triggers a background flush of the frozen table.
func (e *Engine) rotateAndFlush() {
	e.mu.Lock()
	e.rotateLocked()
	e.mu.Unlock()
}

// rotateLocked swaps out the current active MemTable for a fresh one and
// queues the old one for a background flush. Caller must hold e.mu.
//
// This is also called defensively from Put/Delete/applyReplay BEFORE writing
// when the active table is already frozen (rather than only after a write
// freezes it) — without this, a MemTable that starts out frozen (e.g. WAL
// replay hit the size cap mid-recovery) would reject every future write
// forever, since MemTable.Put/Delete short-circuit once frozen.
func (e *Engine) rotateLocked() {
	frozen := e.active
	e.active = memtable.NewMemTable(e.cfg.MemTableMaxSize)
	e.immutable = append(e.immutable, frozen)

	e.flushWg.Add(1)
	go func() {
		defer e.flushWg.Done()
		if err := e.flushImmutableWithRetry(frozen); err != nil {
			slog.Error("SSTable flush failed after retries — table stays queued for the next flush/restart", "error", err)
		}
	}()
}

// flushImmutableWithRetry retries a transient flush failure (e.g. a momentary
// disk error) a bounded number of times with backoff before giving up. On
// total failure the MemTable is left in e.immutable (never dropped) so a
// later ForceFlush, or a WAL replay on restart, can still recover the data.
func (e *Engine) flushImmutableWithRetry(mt *memtable.MemTable) error {
	const maxAttempts = 4
	backoff := 200 * time.Millisecond
	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err = e.flushImmutable(mt); err == nil {
			return nil
		}
		if attempt < maxAttempts {
			slog.Warn("SSTable flush failed, retrying", "attempt", attempt, "error", err)
			time.Sleep(backoff)
			backoff *= 2
		}
	}
	return err
}

// flushImmutable flushes a frozen MemTable to an SSTable via the group
// committer, then removes it from the immutable list and rebuilds the FST.
func (e *Engine) flushImmutable(mt *memtable.MemTable) error {
	entries := mt.Iterator()
	if len(entries) == 0 {
		e.removeImmutable(mt)
		return nil
	}

	path, err := e.committer.Submit(entries)
	if err != nil {
		return fmt.Errorf("storage: flush submit: %w", err)
	}

	slog.Info("SSTable written", "path", path, "entries", len(entries))

	// Register with the compactor so it can track L0 file count and trigger
	// compaction when the threshold is reached. Min/max keys are read directly
	// from the sorted entries slice (first and last entries).
	if e.compactor != nil {
		var size int64
		if info, err2 := os.Stat(path); err2 == nil {
			size = info.Size()
		}
		minKey := make([]byte, len(entries[0].Key))
		copy(minKey, entries[0].Key)
		maxKey := make([]byte, len(entries[len(entries)-1].Key))
		copy(maxKey, entries[len(entries)-1].Key)

		var seq int64
		if m := flushFileRe.FindStringSubmatch(path); m != nil {
			if n, perr := strconv.ParseInt(m[1], 10, 64); perr == nil {
				seq = n
			}
		}

		e.compactor.AddSSTable(&compaction.SSTableMeta{
			Path:   path,
			MinKey: minKey,
			MaxKey: maxKey,
			Size:   size,
			Level:  0,
			Seq:    seq,
		})
	}

	e.removeImmutable(mt)

	// Rebuild FST after every flush so prefix search reflects new terms.
	if err := e.RebuildFST(); err != nil {
		slog.Warn("FST rebuild after flush failed", "error", err)
	}

	return nil
}

// removeImmutable removes mt from the immutable list once it has been flushed.
func (e *Engine) removeImmutable(mt *memtable.MemTable) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i, m := range e.immutable {
		if m == mt {
			e.immutable = append(e.immutable[:i], e.immutable[i+1:]...)
			return
		}
	}
}

// ForceFlush flushes the current active MemTable to an SSTable immediately,
// regardless of whether it has reached maxSize. Used during graceful shutdown
// and explicit Save() calls.
func (e *Engine) ForceFlush() error {
	e.mu.Lock()
	if e.active.Size() == 0 {
		e.mu.Unlock()
		return nil
	}
	frozen := e.active
	e.active = memtable.NewMemTable(e.cfg.MemTableMaxSize)
	e.immutable = append(e.immutable, frozen)
	e.mu.Unlock()

	return e.flushImmutable(frozen)
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

// Close gracefully shuts down the engine:
//  1. Flushes the active MemTable to an SSTable
//  2. Waits for all background flushes to complete
//  3. Stops the background compactor
//  4. Rebuilds the FST one final time
//  5. Closes the WAL
func (e *Engine) Close() error {
	var closeErr error
	e.closeOnce.Do(func() {
		close(e.closed)

		if err := e.ForceFlush(); err != nil {
			slog.Error("ForceFlush on close failed", "error", err)
			closeErr = err
		}

		e.flushWg.Wait()

		if e.compactor != nil {
			e.compactor.Stop()
		}

		if err := e.RebuildFST(); err != nil {
			slog.Warn("Final FST rebuild failed", "error", err)
		}

		if err := e.walFile.Close(); err != nil {
			slog.Error("WAL close failed", "error", err)
			if closeErr == nil {
				closeErr = err
			}
		}
	})
	return closeErr
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func (e *Engine) isClosed() bool {
	select {
	case <-e.closed:
		return true
	default:
		return false
	}
}

// nextSSTPath generates a unique SSTable file path.
// Called by the GroupCommitter's pathGen function.
func (e *Engine) nextSSTPath() string {
	n := e.sstCounter.Add(1)
	return filepath.Join(e.cfg.SSTDir, fmt.Sprintf("%010d.sst", n))
}

// ── Journal recovery ──────────────────────────────────────────────────────────

// Records returns WAL records recovered during Open. The server reads these
// once to replay document mutations into the index engine after loading a gob
// snapshot. Cleared by Checkpoint.
func (e *Engine) Records() []wal.Record { return e.journalRecords }

// Checkpoint is called after a successful gob save. It resets the WAL so the
// next startup has an empty journal — nothing to replay beyond the gob.
func (e *Engine) Checkpoint() error {
	e.journalRecords = nil
	return e.walFile.Reset()
}
