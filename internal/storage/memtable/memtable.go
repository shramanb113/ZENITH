package memtable

import (
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/shramanb113/ZENITH/internal/storage/wal"
)

var (
	ErrFrozen   = errors.New("memtable: is frozen and cannot accept writes")
	ErrKeyEmpty = errors.New("memtable: key cannot be empty")
)

const DefaultMaxSize = 64 * 1024 * 1024 // 64 MB

// MemTable is a sorted, concurrent in-memory write buffer backed by a skip-list.
// It accepts writes until it is frozen (size >= maxSize), at which point the
// storage engine rotates it into an immutable and flushes it to an SSTable.
//
// Concurrency: the skip-list owns its own RWMutex for data access; the frozen
// and size fields are updated atomically. Multiple goroutines may call Put,
// Delete, and Get simultaneously.
type MemTable struct {
	list    *skipList
	size    atomic.Int64
	maxSize int64
	frozen  atomic.Bool
}

// Entry is a key-value pair emitted by Iterator for the SSTable writer.
type Entry struct {
	Key     []byte
	Value   []byte // nil when Deleted is true
	Deleted bool   // true = tombstone
	Seq     uint64 // WAL sequence number — reserved for future compaction use
}

func NewMemTable(maxSize int64) *MemTable {
	if maxSize <= 0 {
		maxSize = DefaultMaxSize
	}
	return &MemTable{
		list:    newSkipList(),
		maxSize: maxSize,
	}
}

// Put inserts or updates key→value. Returns ErrFrozen if the table is frozen,
// ErrKeyEmpty if key is nil/empty.
func (m *MemTable) Put(key, value []byte) error {
	if len(key) == 0 {
		return ErrKeyEmpty
	}
	if m.frozen.Load() {
		return ErrFrozen
	}

	k := make([]byte, len(key))
	copy(k, key)
	v := make([]byte, len(value))
	copy(v, value)

	delta := m.list.put(k, v, false)
	if m.size.Add(delta) >= m.maxSize {
		m.frozen.Store(true)
	}
	return nil
}

// Delete writes a tombstone for key. Returns ErrFrozen / ErrKeyEmpty on error.
func (m *MemTable) Delete(key []byte) error {
	if len(key) == 0 {
		return ErrKeyEmpty
	}
	if m.frozen.Load() {
		return ErrFrozen
	}

	k := make([]byte, len(key))
	copy(k, key)

	delta := m.list.put(k, nil, true)
	if m.size.Add(delta) >= m.maxSize {
		m.frozen.Store(true)
	}
	return nil
}

// Get returns (value, true) for a live key, or (nil, false) if the key is
// absent or a tombstone.
func (m *MemTable) Get(key []byte) ([]byte, bool) {
	val, exists, deleted := m.GetRaw(key)
	if !exists || deleted {
		return nil, false
	}
	return val, true
}

// GetRaw returns the raw tri-state lookup result: (value, exists, deleted).
// exists=false means the key was never written to this MemTable at all —
// callers falling through a stack of tables (active -> immutable -> ...)
// should keep searching older layers. exists=true, deleted=true means a
// tombstone was found — callers MUST stop and treat the key as absent
// without checking older layers, since a tombstone shadows any older value.
func (m *MemTable) GetRaw(key []byte) (value []byte, exists bool, deleted bool) {
	if len(key) == 0 {
		return nil, false, false
	}
	return m.list.get(key)
}

// Iterator returns all entries in lexicographic key order. Because the
// skip-list keeps keys sorted at all times, no post-sort is needed.
// Safe to call concurrently — the skip-list takes a read snapshot.
func (m *MemTable) Iterator() []Entry {
	return m.list.iterate()
}

func (m *MemTable) IsFrozen() bool { return m.frozen.Load() }
func (m *MemTable) Size() int64    { return m.size.Load() }

// ApplyRecord replays a WAL record into the MemTable during recovery.
func ApplyRecord(m *MemTable, r wal.Record) error {
	switch r.Op {
	case wal.OpTypePut:
		return m.Put(r.Key, r.Value)
	case wal.OpTypeDelete:
		return m.Delete(r.Key)
	default:
		return fmt.Errorf("memtable: unknown op type %d at seq %d", r.Op, r.Seq)
	}
}

