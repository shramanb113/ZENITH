package main

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/ranking"
	storage "github.com/shramanb113/ZENITH/internal/storage"
)

// TestRealStorageTxnSatisfiesIndexTxn proves *storage.Txn really satisfies
// index.Txn by duck typing alone — no shared import between the two
// packages makes this true.
func TestRealStorageTxnSatisfiesIndexTxn(t *testing.T) {
	storageEng, err := storage.Open(storage.EngineConfig{Dir: t.TempDir()})
	if err != nil {
		t.Fatalf("storage.Open: %v", err)
	}
	defer storageEng.Close()

	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	emb := embedding.NewDeterministicEmbedder(384)
	scorer := ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight)
	engine := index.NewEngine(cfg, emb, scorer, analysis.NewStandardAnalyzer())
	defer engine.Close()

	ctx := context.Background()
	txn := storageEng.NewTxn() // *storage.Txn, passed where index.Txn is expected
	docs := []index.BatchDoc{{ID: "doc1", Text: "hello world"}}
	if err := engine.AddTransaction(ctx, docs, txn); err != nil {
		t.Fatalf("AddTransaction with a real *storage.Txn: %v", err)
	}

	results, err := engine.SearchFiltered(ctx, "hello", nil)
	if err != nil {
		t.Fatalf("SearchFiltered: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("doc1 not searchable after a real-storage-backed AddTransaction")
	}

	val, ok := storageEng.Get([]byte("doc1"))
	if !ok {
		t.Fatal("doc1 not durably journaled by the real Txn commit")
	}
	_ = val
}
