// Engine orchestrator
package storage

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/cockroachdb/pebble"
)

// journalOpPut/journalOpDelete prefix every journaled value with one byte so
// a full-keyspace replay (Replay) can still see deletes. Pebble's public
// iterators silently skip real tombstones (db.Delete) during iteration —
// relying on "absent from iteration means deleted" would silently lose
// delete-since-last-segment-save information across a restart. A logical
// delete is therefore written as a live Set carrying only the sentinel byte,
// not a real Pebble Delete; real Pebble Deletes are reserved for Prune,
// which really does want the entry gone for good.
const (
	journalOpPut    byte = 1
	journalOpDelete byte = 2
)

// Engine durably journals document mutations ahead of the in-memory index
// (the DocumentJournal contract) and supports atomic multi-document
// transactions (Txn). It is backed by a single Pebble LSM store — Pebble
// owns its own WAL, MemTable, SSTables, compaction, and crash recovery; none
// of that is reimplemented here.
type Engine struct {
	db *pebble.DB

	closeOnce sync.Once
	closed    chan struct{}
}

// EngineConfig holds storage engine parameters.
type EngineConfig struct {
	// Dir is the directory Pebble stores its data in. Created if missing.
	Dir string
}

// DefaultEngineConfig returns a production-ready config.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{Dir: "./data/pebble"}
}

// Open creates or recovers a storage Engine at cfg.Dir. Recovery (replaying
// Pebble's own internal WAL for anything not yet flushed to an SSTable) is
// handled entirely inside pebble.Open — no hand-rolled recovery code runs
// here. The caller must call Close() when done.
func Open(cfg EngineConfig) (*Engine, error) {
	if cfg.Dir == "" {
		return nil, errors.New("storage: EngineConfig.Dir must not be empty")
	}
	if err := os.MkdirAll(cfg.Dir, 0755); err != nil {
		return nil, fmt.Errorf("storage: create dir: %w", err)
	}

	db, err := pebble.Open(cfg.Dir, &pebble.Options{})
	if err != nil {
		return nil, fmt.Errorf("storage: pebble open: %w", err)
	}

	return &Engine{
		db:     db,
		closed: make(chan struct{}),
	}, nil
}

// Close flushes and closes the underlying Pebble store.
func (e *Engine) Close() error {
	var closeErr error
	e.closeOnce.Do(func() {
		close(e.closed)
		closeErr = e.db.Close()
	})
	return closeErr
}

func (e *Engine) isClosed() bool {
	select {
	case <-e.closed:
		return true
	default:
		return false
	}
}

// Put durably records key→value before the caller applies it to the
// in-memory index (the DocumentJournal contract). pebble.Sync forces an
// fsync before returning, matching the old engine's SyncAlways guarantee.
func (e *Engine) Put(ctx context.Context, key, value []byte) error {
	if e.isClosed() {
		return errors.New("storage: engine is closed")
	}
	if len(key) == 0 {
		return errors.New("storage: key must not be empty")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	envelope := make([]byte, 1+len(value))
	envelope[0] = journalOpPut
	copy(envelope[1:], value)
	if err := e.db.Set(key, envelope, pebble.Sync); err != nil {
		return fmt.Errorf("storage: pebble set: %w", err)
	}
	return nil
}

// Delete durably records a logical delete for key (see the journalOp* doc
// comment above for why this is a live Set, not a real Pebble Delete).
func (e *Engine) Delete(ctx context.Context, key []byte) error {
	if e.isClosed() {
		return errors.New("storage: engine is closed")
	}
	if len(key) == 0 {
		return errors.New("storage: key must not be empty")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	if err := e.db.Set(key, []byte{journalOpDelete}, pebble.Sync); err != nil {
		return fmt.Errorf("storage: pebble tombstone set: %w", err)
	}
	return nil
}

// Get returns the decoded value for key if it is currently journaled as a
// live Put. Not called by the product today (nothing in cmd/server or
// cmd/zenith reads the journal back directly) — kept because it is a
// correct, essentially free building block for tests and any future caller,
// unlike the old engine's Get, which never read SSTables back at all.
func (e *Engine) Get(key []byte) ([]byte, bool) {
	raw, closer, err := e.db.Get(key)
	if err != nil {
		return nil, false
	}
	defer closer.Close()
	if len(raw) == 0 || raw[0] != journalOpPut {
		return nil, false
	}
	value := make([]byte, len(raw)-1)
	copy(value, raw[1:])
	return value, true
}

// Replay iterates every currently-journaled entry in key order and invokes
// fn once per entry, reporting whether it was a logical delete. Unlike the
// old engine's Records() (a WAL-record buffer populated once at Open and
// drained by the caller), Replay reflects live state at call time — correct
// because Pebble has already recovered everything into queryable state by
// the time Open returns, with no separate "recovered records" buffer to
// track.
func (e *Engine) Replay(fn func(key, value []byte, isDelete bool) error) error {
	iter, err := e.db.NewIter(nil)
	if err != nil {
		return fmt.Errorf("storage: new iterator: %w", err)
	}
	defer iter.Close()

	for valid := iter.First(); valid; valid = iter.Next() {
		raw := iter.Value()
		if len(raw) == 0 {
			continue
		}
		key := append([]byte(nil), iter.Key()...)
		switch raw[0] {
		case journalOpDelete:
			if err := fn(key, nil, true); err != nil {
				return err
			}
		case journalOpPut:
			value := append([]byte(nil), raw[1:]...)
			if err := fn(key, value, false); err != nil {
				return err
			}
		}
	}
	return iter.Error()
}

// Snapshot is a point-in-time view of the journal, used to coordinate a safe
// Prune: capture it right before calling the index's segment Save(), then
// Prune exactly these keys after Save() succeeds. Pruning based on wall-clock
// time instead would race against writes arriving while Save() runs.
type Snapshot struct {
	snap *pebble.Snapshot
}

// Snapshot returns a new point-in-time view of the journal.
func (e *Engine) Snapshot() *Snapshot {
	return &Snapshot{snap: e.db.NewSnapshot()}
}

// Close releases the snapshot's resources. Safe to call after Prune.
func (s *Snapshot) Close() error {
	return s.snap.Close()
}

// Prune permanently deletes every key s saw, in one atomic batch. Call only
// after the segment save s was taken for has succeeded — those keys are now
// redundant with that segment. Unlike Put/Delete's logical envelope deletes,
// this uses a real Pebble Delete: these entries are gone for good.
func (e *Engine) Prune(s *Snapshot) error {
	iter, err := s.snap.NewIter(nil)
	if err != nil {
		return fmt.Errorf("storage: snapshot iterator: %w", err)
	}
	defer iter.Close()

	batch := e.db.NewBatch()
	defer batch.Close()
	for valid := iter.First(); valid; valid = iter.Next() {
		if err := batch.Delete(iter.Key(), nil); err != nil {
			return fmt.Errorf("storage: stage prune delete: %w", err)
		}
	}
	if err := iter.Error(); err != nil {
		return err
	}
	if batch.Empty() {
		return nil
	}
	if err := batch.Commit(pebble.Sync); err != nil {
		return fmt.Errorf("storage: commit prune batch: %w", err)
	}
	return nil
}

// Txn stages a batch of Put/Delete operations for one atomic commit: every
// key becomes durable and visible together (one fsync), or Discard releases
// it with nothing applied. Used for multi-document transactions — see
// internal/index.Engine.AddTransaction.
type Txn struct {
	batch *pebble.Batch
}

// NewTxn starts a new transaction against e.
func (e *Engine) NewTxn() *Txn {
	return &Txn{batch: e.db.NewBatch()}
}

// Put stages a logical put. Not durable until Commit succeeds.
func (t *Txn) Put(key, value []byte) error {
	envelope := make([]byte, 1+len(value))
	envelope[0] = journalOpPut
	copy(envelope[1:], value)
	return t.batch.Set(key, envelope, nil)
}

// Delete stages a logical delete. Not durable until Commit succeeds.
func (t *Txn) Delete(key []byte) error {
	return t.batch.Set(key, []byte{journalOpDelete}, nil)
}

// Commit durably applies every staged operation atomically: one fsync, every
// key visible together or none are.
func (t *Txn) Commit(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		_ = t.batch.Close()
		return err
	}
	if err := t.batch.Commit(pebble.Sync); err != nil {
		return fmt.Errorf("storage: txn commit: %w", err)
	}
	return nil
}

// Discard releases the transaction's resources without applying anything.
func (t *Txn) Discard() error {
	return t.batch.Close()
}
