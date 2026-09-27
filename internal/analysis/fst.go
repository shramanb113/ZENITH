package analysis

import (
	"bytes"
	"fmt"
	"os"
	"runtime"
	"sort"
	"sync"

	"github.com/blevesearch/vellum"
)

// FSTDictionary is a read-optimised, memory-efficient term dictionary built
// on top of blevesearch/vellum (finite state transducer).
//
// # Two operating modes
//
//  1. In-memory (Build): for tests and one-off use. The FST lives in a
//     bytes.Buffer and is held entirely in RAM.
//
//  2. On-disk (BuildToFile + OpenFromFile): for production. The FST is written
//     atomically to a file and reopened via memory-mapping. The OS pages in
//     only the portions accessed — 2–4 bytes/term on disk vs 50–100 bytes/term
//     in a Go map. Startup is instant (open mmap, no rebuild).
//
// # Memory safety
//
// Contains, PrefixSearch, and Size hold a read lock; Build, BuildToFile,
// OpenFromFile, and Close hold a write lock. Concurrent queries are safe while
// a rebuild is in flight.
type FSTDictionary struct {
	mu    sync.RWMutex
	fst   *vellum.FST
	built bool
}

// NewFSTDictionary creates an empty FSTDictionary.
// Call Build or OpenFromFile before using Contains or PrefixSearch.
func NewFSTDictionary() *FSTDictionary {
	return &FSTDictionary{}
}

// ─── In-memory build ──────────────────────────────────────────────────────────

// Build constructs the FST entirely in RAM from terms.
// terms need not be sorted — Build sorts and deduplicates internally.
// Prefer BuildToFile for production; use Build in tests.
func (d *FSTDictionary) Build(terms []string) error {
	deduped := sortAndDedup(terms)

	var buf bytes.Buffer
	fst, err := buildFSTInto(&buf, deduped)
	if err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	closeFST(d.fst)
	d.fst = fst
	d.built = len(deduped) > 0
	return nil
}

// ─── On-disk build ────────────────────────────────────────────────────────────

// BuildToFile builds the FST and writes it atomically to path, then reloads it.
//
// On Linux/macOS the file is memory-mapped (vellum.Open) for efficient paging.
// On Windows, mmap is avoided entirely: the FST is read into memory with
// vellum.Load so no file handle is kept open. This is required because Windows
// refuses to rename over a file that has any open handle — whether from our own
// mmap, OneDrive sync, or Windows Defender — producing "Access is denied".
func (d *FSTDictionary) BuildToFile(terms []string, path string) error {
	deduped := sortAndDedup(terms)
	tmpPath := path + ".tmp"

	// Write the new FST to the temp file outside the lock — this is the slow part.
	f, err := os.Create(tmpPath)
	if err != nil {
		return fmt.Errorf("fst: create %s: %w", tmpPath, err)
	}
	if _, err2 := buildFSTInto(f, deduped); err2 != nil {
		f.Close()
		os.Remove(tmpPath)
		return err2
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("fst: sync %s: %w", tmpPath, err)
	}
	f.Close()

	// Hold the write lock for the entire rename → reload sequence, but keep the
	// OLD fst/built state intact until the NEW one has loaded successfully. A
	// rename or load failure (e.g. Windows AV holding a handle) must leave
	// query resolution exactly as usable as before this call, not silently
	// disabled until the next successful rebuild.
	d.mu.Lock()
	defer d.mu.Unlock()

	// On Windows: remove the destination before rename so that no external
	// handle (cloud sync, antivirus) can block the operation. This only
	// touches the on-disk file; the in-memory d.fst (already fully loaded via
	// vellum.Load on Windows, not mmap'd) is unaffected until the swap below.
	if runtime.GOOS == "windows" {
		_ = os.Remove(path)
	}

	if err := os.Rename(tmpPath, path); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("fst: rename to %s: %w", path, err)
	}

	newFST, err := loadFST(path)
	if err != nil {
		return fmt.Errorf("fst: load %s: %w", path, err)
	}

	closeFST(d.fst)
	d.fst = newFST
	d.built = true
	return nil
}

// OpenFromFile loads the FST at path, replacing any existing FST.
// The old FST (if any) is closed first to release any resources it holds.
func (d *FSTDictionary) OpenFromFile(path string) error {
	newFST, err := loadFST(path)
	if err != nil {
		return fmt.Errorf("fst: open %s: %w", path, err)
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	closeFST(d.fst)
	d.fst = newFST
	d.built = true
	return nil
}

// loadFST opens the FST at path.
// On Windows the file is read fully into memory (vellum.Load) so no file handle
// remains open after this call. On other platforms it is memory-mapped
// (vellum.Open) so the OS can page in only the portions that are accessed.
func loadFST(path string) (*vellum.FST, error) {
	if runtime.GOOS == "windows" {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return vellum.Load(data)
	}
	return vellum.Open(path)
}

// Close releases any memory mapping held by the FST.
// After Close, Contains and PrefixSearch return false/nil until rebuilt.
func (d *FSTDictionary) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if err := closeFST(d.fst); err != nil {
		return err
	}
	d.fst = nil
	d.built = false
	return nil
}

// ─── Query API ────────────────────────────────────────────────────────────────

// Contains returns true if term exists exactly in the dictionary.
func (d *FSTDictionary) Contains(term string) bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.built || d.fst == nil {
		return false
	}
	_, exists, err := d.fst.Get([]byte(term))
	return err == nil && exists
}

// PrefixSearch returns up to maxResults terms that start with prefix,
// in lexicographic order. Returns nil if the FST is not built or no match.
func (d *FSTDictionary) PrefixSearch(prefix string, maxResults int) ([]string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.built || d.fst == nil {
		return nil, nil
	}
	if maxResults <= 0 {
		maxResults = 20
	}

	startKey := []byte(prefix)
	endKey := prefixUpperBound(prefix)

	itr, err := d.fst.Iterator(startKey, endKey)
	if err != nil {
		return nil, nil
	}

	var results []string
	for err == nil && len(results) < maxResults {
		key, _ := itr.Current()
		results = append(results, string(key))
		err = itr.Next()
	}
	return results, nil
}

// Size returns the number of terms in the FST. Returns 0 if not built.
func (d *FSTDictionary) Size() int {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.built || d.fst == nil {
		return 0
	}
	return int(d.fst.Len())
}

// IsBuilt returns true if Build or OpenFromFile has been called successfully.
func (d *FSTDictionary) IsBuilt() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.built
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

// buildFSTInto writes a vellum FST for deduped (already sorted+deduped) terms
// to w. Returns the loaded FST when w is a *bytes.Buffer (in-memory path),
// or nil when w is a file (caller will re-open via mmap).
func buildFSTInto(w interface {
	Write([]byte) (int, error)
}, deduped []string) (*vellum.FST, error) {
	if len(deduped) == 0 {
		// Return an empty in-memory FST so Contains always returns false cleanly.
		var buf bytes.Buffer
		b, err := vellum.New(&buf, nil)
		if err != nil {
			return nil, fmt.Errorf("fst: create builder: %w", err)
		}
		if err := b.Close(); err != nil {
			return nil, fmt.Errorf("fst: close empty builder: %w", err)
		}
		// Write serialized bytes to w so BuildToFile gets a valid (non-empty) file.
		if _, err := w.Write(buf.Bytes()); err != nil {
			return nil, fmt.Errorf("fst: write empty: %w", err)
		}
		fst, err := vellum.Load(buf.Bytes())
		if err != nil {
			return nil, fmt.Errorf("fst: load empty: %w", err)
		}
		return fst, nil
	}

	builder, err := vellum.New(w, nil)
	if err != nil {
		return nil, fmt.Errorf("fst: create builder: %w", err)
	}

	for i, term := range deduped {
		if err := builder.Insert([]byte(term), uint64(i+1)); err != nil {
			return nil, fmt.Errorf("fst: insert %q: %w", term, err)
		}
	}
	if err := builder.Close(); err != nil {
		return nil, fmt.Errorf("fst: builder close: %w", err)
	}

	// For in-memory path (bytes.Buffer), load and return the FST.
	if buf, ok := w.(*bytes.Buffer); ok {
		fst, err := vellum.Load(buf.Bytes())
		if err != nil {
			return nil, fmt.Errorf("fst: load: %w", err)
		}
		return fst, nil
	}

	// For file path, caller re-opens via vellum.Open (mmap).
	return nil, nil
}

// sortAndDedup returns a sorted, deduplicated copy of terms.
func sortAndDedup(terms []string) []string {
	if len(terms) == 0 {
		return nil
	}
	cp := make([]string, len(terms))
	copy(cp, terms)
	sort.Strings(cp)
	out := cp[:0]
	for i, t := range cp {
		if i == 0 || t != cp[i-1] {
			out = append(out, t)
		}
	}
	return out
}

// closeFST closes fst if non-nil, ignoring nil. Used for mmap cleanup.
func closeFST(fst *vellum.FST) error {
	if fst != nil {
		return fst.Close()
	}
	return nil
}

// prefixUpperBound returns the smallest byte string strictly greater than all
// strings starting with prefix. Used as the exclusive end key for vellum's
// prefix iterator.
func prefixUpperBound(prefix string) []byte {
	b := []byte(prefix)
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] < 0xff {
			upper := make([]byte, i+1)
			copy(upper, b)
			upper[i]++
			return upper
		}
	}
	return nil
}
