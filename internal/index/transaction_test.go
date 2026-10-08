package index

import (
	"context"
	"errors"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// fakeTxn is a minimal in-memory stand-in for *storage.Txn, used so this
// package's tests don't need to import internal/storage (keeping the
// decoupling the production code relies on honest in the tests too).
type fakeTxn struct {
	puts      map[string][]byte
	deletes   map[string]bool
	committed bool
	discarded bool
	commitErr error
}

func newFakeTxn() *fakeTxn {
	return &fakeTxn{puts: map[string][]byte{}, deletes: map[string]bool{}}
}
func (f *fakeTxn) Put(key, value []byte) error {
	f.puts[string(key)] = append([]byte(nil), value...)
	return nil
}
func (f *fakeTxn) Delete(key []byte) error {
	f.deletes[string(key)] = true
	return nil
}
func (f *fakeTxn) Commit(ctx context.Context) error {
	if f.commitErr != nil {
		return f.commitErr
	}
	f.committed = true
	return nil
}
func (f *fakeTxn) Discard() error {
	f.discarded = true
	return nil
}

func newTestEngineForTxn() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	return e
}

func TestAddTransaction_AllDocsSearchableAfterCommit(t *testing.T) {
	e := newTestEngineForTxn()
	defer e.Close()
	ctx := context.Background()
	txn := newFakeTxn()

	docs := []BatchDoc{
		{ID: "doc1", Text: "hello world"},
		{ID: "doc2", Text: "goodbye world"},
	}
	if err := e.AddTransaction(ctx, docs, txn); err != nil {
		t.Fatalf("AddTransaction: %v", err)
	}
	if !txn.committed {
		t.Fatal("txn was never committed")
	}
	if len(txn.puts) != 2 {
		t.Fatalf("txn.puts has %d entries, want 2", len(txn.puts))
	}

	results, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("doc1 not searchable after AddTransaction committed")
	}
}

func TestAddTransaction_CommitFailureAppliesNothing(t *testing.T) {
	e := newTestEngineForTxn()
	defer e.Close()
	ctx := context.Background()
	txn := newFakeTxn()
	txn.commitErr = errors.New("simulated commit failure")

	docs := []BatchDoc{
		{ID: "doc1", Text: "hello world"},
		{ID: "doc2", Text: "goodbye world"},
	}
	if err := e.AddTransaction(ctx, docs, txn); err == nil {
		t.Fatal("AddTransaction: want error on commit failure, got nil")
	}

	results, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) != 0 {
		t.Fatal("a document became searchable despite the transaction's commit failing — in-memory apply must be all-or-nothing")
	}
}

func TestRemoveBatch_AllRemovedAfterCommit(t *testing.T) {
	e := newTestEngineForTxn()
	defer e.Close()
	ctx := context.Background()

	if err := e.AddBatch(ctx, []BatchDoc{{ID: "doc1", Text: "hello world"}, {ID: "doc2", Text: "goodbye world"}}); err != nil {
		t.Fatalf("AddBatch: %v", err)
	}

	txn := newFakeTxn()
	if err := e.RemoveBatch(ctx, []string{"doc1", "doc2"}, txn); err != nil {
		t.Fatalf("RemoveBatch: %v", err)
	}
	if !txn.committed {
		t.Fatal("txn was never committed")
	}
	if len(txn.deletes) != 2 {
		t.Fatalf("txn.deletes has %d entries, want 2", len(txn.deletes))
	}

	results, err := e.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) != 0 {
		t.Fatal("doc1 still searchable after RemoveBatch committed")
	}
}
