package compaction

import (
	"bytes"
	"container/heap"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shramanb113/ZENITH/internal/storage/memtable"
	"github.com/shramanb113/ZENITH/internal/storage/sstable"
)

// CompactorConfig holds all tuning knobs for the leveled compactor.
type CompactorConfig struct {
	// L0Threshold is the number of L0 SSTables that triggers an L0→L1 compaction.
	L0Threshold int

	// LevelSizeBase is the byte budget for L1. Each subsequent level is
	// LevelSizeBase * LevelSizeMult^(level-1).
	LevelSizeBase int64

	// LevelSizeMult is the size multiplier between adjacent levels (typically 10).
	LevelSizeMult int

	// MaxLevels is the total number of levels (L0 … L(MaxLevels-1)).
	MaxLevels int

	// CompactionInterval is how often the background goroutine checks for work.
	CompactionInterval time.Duration

	// Dir is the directory where compacted SSTable files are written.
	Dir string
}

// SSTableMeta describes an SSTable for the compactor's bookkeeping.
type SSTableMeta struct {
	Path   string
	MinKey []byte
	MaxKey []byte
	Size   int64
	Level  int

	// Seq orders SSTables by true write recency, independent of which level
	// they currently live in. It is the flush counter value the file was
	// created with (or, for a compaction output, the max Seq among its
	// inputs). Duplicate keys across files are resolved by Seq, not by
	// level or by file-slice position — L0 can hold several files with
	// overlapping key ranges, and slice/level order alone doesn't tell you
	// which one was written last.
	Seq int64
}

// ManifestEntry is the on-disk representation of one live SSTable, persisted
// so a restart can rediscover the full level structure instead of starting
// from an empty set (which would make the next flush reuse and overwrite a
// stale file number).
type ManifestEntry struct {
	Path  string `json:"path"`
	Level int    `json:"level"`
	Seq   int64  `json:"seq"`
}

// Snapshot returns the current set of live SSTables across all levels, for persistence.
func (c *Compactor) Snapshot() []ManifestEntry {
	c.mu.Lock()
	defer c.mu.Unlock()
	var out []ManifestEntry
	for lvl, metas := range c.levels {
		for _, m := range metas {
			out = append(out, ManifestEntry{Path: m.Path, Level: lvl, Seq: m.Seq})
		}
	}
	return out
}

// LoadSnapshot repopulates level state from a previously persisted manifest.
// Must be called before Run(). opener reads back the min/max key and size of
// each file (Reader already exposes these; nothing needs to be duplicated
// into the manifest itself). Entries that fail to open are skipped with a
// warning rather than aborting startup — a single missing/corrupt SSTable
// file must not block the engine from opening.
func (c *Compactor) LoadSnapshot(entries []ManifestEntry, opener func(path string) (minKey, maxKey []byte, size int64, err error)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, e := range entries {
		if e.Level < 0 || e.Level >= len(c.levels) {
			continue
		}
		minKey, maxKey, size, err := opener(e.Path)
		if err != nil {
			slog.Warn("compaction: skipping SSTable from manifest", "path", e.Path, "error", err)
			continue
		}
		c.levels[e.Level] = append(c.levels[e.Level], &SSTableMeta{
			Path:   e.Path,
			MinKey: minKey,
			MaxKey: maxKey,
			Size:   size,
			Level:  e.Level,
			Seq:    e.Seq,
		})
	}
}

// SetOnChange registers a callback invoked (without c.mu held) whenever the
// live SSTable set changes — used by the storage Engine to persist the
// manifest after every flush registration and every compaction pass.
func (c *Compactor) SetOnChange(fn func()) {
	c.mu.Lock()
	c.onChange = fn
	c.mu.Unlock()
}

func (c *Compactor) notifyChanged() {
	c.mu.Lock()
	fn := c.onChange
	c.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// ─── Compactor ────────────────────────────────────────────────────────────────

// Compactor performs leveled compaction in the background.
//
// Level layout:
//   - L0: directly receives flushed MemTable SSTables; files may have overlapping key ranges.
//   - L1…Ln: non-overlapping key ranges within a level (maintained by compaction).
//
// Trigger rules:
//   - L0 compaction: when len(L0) >= L0Threshold, merge ALL L0 files + overlapping L1 files → L1.
//   - Ln compaction: when totalSize(Ln) > LevelSizeBase * LevelSizeMult^(n-1), pick the
//     largest Ln file + all overlapping L(n+1) files → L(n+1).
//
// Tombstone pruning: tombstones are dropped only when merging into the last level.
type Compactor struct {
	mu          sync.Mutex
	levels      [][]*SSTableMeta
	cfg         CompactorConfig
	stopCh      chan struct{}
	stopOnce    sync.Once
	wg          sync.WaitGroup
	pathCounter atomic.Uint64
	onChange    func()
}

// ─── Lifecycle ────────────────────────────────────────────────────────────────

// NewCompactor creates a Compactor. Zero-value cfg fields are replaced with
// production-safe defaults.
func NewCompactor(cfg CompactorConfig) *Compactor {
	if cfg.L0Threshold == 0 {
		cfg.L0Threshold = 4
	}
	if cfg.LevelSizeBase == 0 {
		cfg.LevelSizeBase = 10 * 1024 * 1024 // 10 MB
	}
	if cfg.LevelSizeMult == 0 {
		cfg.LevelSizeMult = 10
	}
	if cfg.MaxLevels == 0 {
		cfg.MaxLevels = 7
	}
	if cfg.CompactionInterval == 0 {
		cfg.CompactionInterval = 30 * time.Second
	}

	levels := make([][]*SSTableMeta, cfg.MaxLevels)
	return &Compactor{
		levels: levels,
		cfg:    cfg,
		stopCh: make(chan struct{}),
	}
}

// Run starts the background compaction goroutine. Call once after the engine
// is open. Safe to call even if the Compactor will never be stopped (e.g. in
// tests that just want to register SSTables without running compaction).
func (c *Compactor) Run() {
	c.wg.Add(1)
	go func() {
		defer c.wg.Done()
		ticker := time.NewTicker(c.cfg.CompactionInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := c.maybeCompact(); err != nil {
					slog.Warn("compaction cycle failed", "error", err)
				}
			case <-c.stopCh:
				return
			}
		}
	}()
}

// Stop signals the background goroutine to exit and blocks until it has.
// Safe to call multiple times.
func (c *Compactor) Stop() {
	c.stopOnce.Do(func() { close(c.stopCh) })
	c.wg.Wait()
}

// ─── Registration ─────────────────────────────────────────────────────────────

// AddSSTable registers a newly flushed SSTable at L0.
// Called by the storage engine after every MemTable flush.
func (c *Compactor) AddSSTable(meta *SSTableMeta) {
	c.mu.Lock()
	c.levels[0] = append(c.levels[0], meta)
	c.mu.Unlock()
	c.notifyChanged()
}

// ─── Trigger Checks ───────────────────────────────────────────────────────────

// maybeCompact checks all trigger conditions and runs at most one compaction pass.
func (c *Compactor) maybeCompact() error {
	if c.needsL0Compaction() {
		if err := c.compactL0(); err != nil {
			return fmt.Errorf("L0 compaction: %w", err)
		}
	}
	if level := c.needsLevelCompaction(); level >= 0 {
		if err := c.compactLevel(level); err != nil {
			return fmt.Errorf("level %d compaction: %w", level, err)
		}
	}
	return nil
}

// needsL0Compaction returns true when the L0 file count has reached the threshold.
func (c *Compactor) needsL0Compaction() bool {
	c.mu.Lock()
	n := len(c.levels[0])
	c.mu.Unlock()
	return n >= c.cfg.L0Threshold
}

// needsLevelCompaction returns the first level (1…MaxLevels-2) that exceeds its
// size target, or -1 if all are within budget.
func (c *Compactor) needsLevelCompaction() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Check L1 through second-to-last level (last level is the sink — no level below it).
	for i := 1; i < c.cfg.MaxLevels-1; i++ {
		if c.totalLevelSize(i) > c.levelSizeTarget(i) {
			return i
		}
	}
	return -1
}

// ─── Compaction Execution ─────────────────────────────────────────────────────

// compactL0 merges all L0 SSTables plus any overlapping L1 SSTables into L1.
// L0 is special — files can have overlapping key ranges, so all must be merged
// together in a single pass.
func (c *Compactor) compactL0() error {
	// Snapshot L0 under lock.
	c.mu.Lock()
	if len(c.levels[0]) == 0 {
		c.mu.Unlock()
		return nil
	}
	l0 := make([]*SSTableMeta, len(c.levels[0]))
	copy(l0, c.levels[0])
	c.mu.Unlock()

	// Compute the union key range of all L0 files.
	minKey := l0[0].MinKey
	maxKey := l0[0].MaxKey
	for _, m := range l0[1:] {
		if bytes.Compare(m.MinKey, minKey) < 0 {
			minKey = m.MinKey
		}
		if bytes.Compare(m.MaxKey, maxKey) > 0 {
			maxKey = m.MaxKey
		}
	}

	// Find overlapping L1 files (acquires its own lock).
	l1Overlap := c.findOverlapping(1, minKey, maxKey)

	allInputs := append(l0, l1Overlap...)

	outputs, err := c.mergeSSTableS(allInputs, 1)
	if err != nil {
		return err
	}

	c.replaceSSTableS(allInputs, outputs, 1)
	return c.deleteSSTableFiles(allInputs)
}

// compactLevel picks one SSTable from level n, finds all overlapping SSTables
// in level n+1, merges them, writes new SSTables into n+1, and removes the inputs.
func (c *Compactor) compactLevel(level int) error {
	// Pick target under lock then release.
	c.mu.Lock()
	target := c.pickCompactionTarget(level)
	c.mu.Unlock()

	if target == nil {
		return nil
	}

	// Find overlapping files in the next level (acquires its own lock).
	overlapping := c.findOverlapping(level+1, target.MinKey, target.MaxKey)

	inputs := make([]*SSTableMeta, 0, 1+len(overlapping))
	inputs = append(inputs, target)
	inputs = append(inputs, overlapping...)

	outputs, err := c.mergeSSTableS(inputs, level+1)
	if err != nil {
		return err
	}

	c.replaceSSTableS(inputs, outputs, level+1)
	return c.deleteSSTableFiles(inputs)
}

// pickCompactionTarget picks the largest SSTable at level n.
// Caller must hold c.mu.
func (c *Compactor) pickCompactionTarget(level int) *SSTableMeta {
	if level >= len(c.levels) || len(c.levels[level]) == 0 {
		return nil
	}
	best := c.levels[level][0]
	for _, m := range c.levels[level][1:] {
		if m.Size > best.Size {
			best = m
		}
	}
	return best
}

// findOverlapping returns all SSTables at the given level whose key range
// overlaps [minKey, maxKey]. Acquires and releases c.mu internally.
func (c *Compactor) findOverlapping(level int, minKey, maxKey []byte) []*SSTableMeta {
	c.mu.Lock()
	defer c.mu.Unlock()
	if level >= len(c.levels) {
		return nil
	}
	var result []*SSTableMeta
	for _, m := range c.levels[level] {
		// Ranges overlap when m.MaxKey >= minKey AND m.MinKey <= maxKey.
		if bytes.Compare(m.MaxKey, minKey) >= 0 && bytes.Compare(m.MinKey, maxKey) <= 0 {
			result = append(result, m)
		}
	}
	return result
}

// ─── Merge Core ───────────────────────────────────────────────────────────────

// mergeHeapItem is one input SSTable's current head entry in the k-way merge.
type mergeHeapItem struct {
	entry memtable.Entry
	seq   int64
	it    *sstable.Iterator
}

// mergeHeap orders by key ASC; ties broken by Seq DESC (newest write wins).
// Seq — not level, and not slice/append order — is the source of truth for
// recency: L0 alone can hold several files with overlapping key ranges (one
// per MemTable flush), so two entries at the "same level" are not equally
// new, and relying on the order they happened to be appended in silently
// picks the OLDEST one instead of the newest.
type mergeHeap []*mergeHeapItem

func (h mergeHeap) Len() int { return len(h) }
func (h mergeHeap) Less(i, j int) bool {
	cmp := bytes.Compare(h[i].entry.Key, h[j].entry.Key)
	if cmp != 0 {
		return cmp < 0
	}
	return h[i].seq > h[j].seq
}
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(x any)   { *h = append(*h, x.(*mergeHeapItem)) }
func (h *mergeHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return item
}

// advance pulls the next entry from item's iterator and pushes it back onto
// the heap, or drops it silently once its file is exhausted. item.seq is a
// per-file constant (each item wraps one input file's iterator) and is left
// untouched — only item.entry changes as the file is walked.
func advance(h *mergeHeap, item *mergeHeapItem) error {
	e, ok, err := item.it.Next()
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}
	item.entry = e
	heap.Push(h, item)
	return nil
}

// mergeSSTableS merges the entries from all input SSTables into one new SSTable
// at targetLevel, using a k-way streaming merge: only one block per input
// file is held in memory at a time (via sstable.Iterator), instead of
// loading every input file's full contents into memory up front.
//
// Merge semantics:
//   - For duplicate keys, the entry with the highest Seq wins (newest write) — see mergeHeap.
//   - Tombstones are preserved in all levels except the last (where they are dropped).
//
// Returns the metadata of the newly written SSTables.
func (c *Compactor) mergeSSTableS(inputs []*SSTableMeta, targetLevel int) ([]*SSTableMeta, error) {
	if len(inputs) == 0 {
		return nil, nil
	}

	readers := make([]*sstable.Reader, 0, len(inputs))
	defer func() {
		for _, r := range readers {
			r.Close()
		}
	}()

	h := &mergeHeap{}
	heap.Init(h)
	for _, meta := range inputs {
		r, err := sstable.OpenReader(meta.Path)
		if err != nil {
			return nil, fmt.Errorf("compaction: open %s: %w", meta.Path, err)
		}
		readers = append(readers, r)

		it := r.NewIterator()
		item := &mergeHeapItem{it: it, seq: meta.Seq}
		if err := advance(h, item); err != nil {
			return nil, fmt.Errorf("compaction: read %s: %w", meta.Path, err)
		}
	}

	isLastLevel := targetLevel == c.cfg.MaxLevels-1
	var merged []memtable.Entry
	var maxSeq int64

	for h.Len() > 0 {
		top := heap.Pop(h).(*mergeHeapItem)
		key := top.entry.Key
		best := top

		// Among every heap item currently sharing this key, keep only the
		// one with the highest Seq; advance every consumed iterator.
		for h.Len() > 0 && bytes.Equal((*h)[0].entry.Key, key) {
			cand := heap.Pop(h).(*mergeHeapItem)
			winner, loser := best, cand
			if cand.seq > best.seq {
				winner, loser = cand, best
			}
			if err := advance(h, loser); err != nil {
				return nil, fmt.Errorf("compaction: read next entry: %w", err)
			}
			best = winner
		}

		if best.seq > maxSeq {
			maxSeq = best.seq
		}
		if !(best.entry.Deleted && isLastLevel) {
			merged = append(merged, best.entry)
		}
		if err := advance(h, best); err != nil {
			return nil, fmt.Errorf("compaction: read next entry: %w", err)
		}
	}

	if len(merged) == 0 {
		return nil, nil
	}

	// Write the merged entries to a new SSTable.
	outPath := c.newSSTablePath()
	w, err := sstable.NewWriter(outPath)
	if err != nil {
		return nil, fmt.Errorf("compaction: create output SSTable %s: %w", outPath, err)
	}
	if err := w.WriteAll(merged); err != nil {
		w.Close()
		os.Remove(outPath)
		return nil, fmt.Errorf("compaction: write merged SSTable: %w", err)
	}
	if err := w.Close(); err != nil {
		os.Remove(outPath)
		return nil, fmt.Errorf("compaction: close merged SSTable: %w", err)
	}

	// Read back size and key bounds.
	var size int64
	if info, err := os.Stat(outPath); err == nil {
		size = info.Size()
	}

	minKey := make([]byte, len(merged[0].Key))
	copy(minKey, merged[0].Key)
	maxKey := make([]byte, len(merged[len(merged)-1].Key))
	copy(maxKey, merged[len(merged)-1].Key)

	slog.Info("compaction: wrote merged SSTable",
		"path", outPath,
		"entries", len(merged),
		"target_level", targetLevel,
	)

	return []*SSTableMeta{{
		Path:   outPath,
		MinKey: minKey,
		MaxKey: maxKey,
		Size:   size,
		Level:  targetLevel,
		Seq:    maxSeq,
	}}, nil
}

// ─── Manifest / Bookkeeping ───────────────────────────────────────────────────

// replaceSSTableS atomically removes input SSTables from their levels and
// inserts the output SSTables into targetLevel.
func (c *Compactor) replaceSSTableS(inputs []*SSTableMeta, outputs []*SSTableMeta, targetLevel int) {
	inputSet := make(map[string]bool, len(inputs))
	for _, m := range inputs {
		inputSet[m.Path] = true
	}

	c.mu.Lock()

	// Remove every input SSTable from whatever level it lives in.
	for lvl := range c.levels {
		kept := c.levels[lvl][:0]
		for _, m := range c.levels[lvl] {
			if !inputSet[m.Path] {
				kept = append(kept, m)
			}
		}
		c.levels[lvl] = kept
	}

	// Insert outputs at the target level.
	c.levels[targetLevel] = append(c.levels[targetLevel], outputs...)

	c.mu.Unlock()
	c.notifyChanged()
}

// deleteSSTableFiles removes the on-disk files for compacted-away SSTables.
// Best-effort: logs and continues on partial failure, returns the first error.
func (c *Compactor) deleteSSTableFiles(metas []*SSTableMeta) error {
	var firstErr error
	for _, m := range metas {
		if err := os.Remove(m.Path); err != nil && !os.IsNotExist(err) {
			slog.Warn("compaction: failed to delete input SSTable", "path", m.Path, "error", err)
			if firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// levelSizeTarget returns the byte budget for level n.
// Caller must hold c.mu (or level content must not change during the call).
func (c *Compactor) levelSizeTarget(level int) int64 {
	target := c.cfg.LevelSizeBase
	for i := 1; i < level; i++ {
		target *= int64(c.cfg.LevelSizeMult)
	}
	return target
}

// totalLevelSize returns the sum of all SSTable sizes at level n.
// Caller must hold c.mu.
func (c *Compactor) totalLevelSize(level int) int64 {
	var total int64
	for _, m := range c.levels[level] {
		total += m.Size
	}
	return total
}

// newSSTablePath generates a unique file path for a compacted SSTable.
func (c *Compactor) newSSTablePath() string {
	n := c.pathCounter.Add(1)
	return filepath.Join(c.cfg.Dir, fmt.Sprintf("compacted_%010d.sst", n))
}
