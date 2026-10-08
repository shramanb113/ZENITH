package index

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

func writeGenTestEngine() *Engine {
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	e := NewEngine(cfg, noVecEmbedder{}, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	return e
}

func TestWriteGen_BumpsOnAdd(t *testing.T) {
	ctx := context.Background()
	e := writeGenTestEngine()
	defer e.Close()

	before := e.writeGen
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", nil, nil); err != nil {
		t.Fatal(err)
	}
	if e.writeGen == before {
		t.Fatalf("writeGen unchanged after Add: before=%d after=%d", before, e.writeGen)
	}
}

func TestWriteGen_BumpsOnAddBatch(t *testing.T) {
	ctx := context.Background()
	e := writeGenTestEngine()
	defer e.Close()

	before := e.writeGen
	if err := e.AddBatch(ctx, []BatchDoc{{ID: "doc1", Text: "hello world"}}); err != nil {
		t.Fatal(err)
	}
	if e.writeGen == before {
		t.Fatalf("writeGen unchanged after AddBatch: before=%d after=%d", before, e.writeGen)
	}
}

func TestWriteGen_BumpsOnRemove(t *testing.T) {
	ctx := context.Background()
	e := writeGenTestEngine()
	defer e.Close()

	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", nil, nil); err != nil {
		t.Fatal(err)
	}
	before := e.writeGen
	if err := e.Remove(ctx, "doc1"); err != nil {
		t.Fatal(err)
	}
	if e.writeGen == before {
		t.Fatalf("writeGen unchanged after Remove: before=%d after=%d", before, e.writeGen)
	}
}

func TestWriteGen_BumpsOnRemoveOfSegmentOnlyDoc(t *testing.T) {
	// Remove's early-return path (killBase, for a doc that lives in a
	// flushed segment rather than the live delta) must also bump writeGen —
	// it is a real mutation (the doc becomes a tombstone) even though it
	// never touches e.idMapping.
	ctx := context.Background()
	e := writeGenTestEngine()
	defer e.Close()

	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	before := e.writeGen
	if err := e.Remove(ctx, "doc1"); err != nil {
		t.Fatal(err)
	}
	if e.writeGen == before {
		t.Fatalf("writeGen unchanged after Remove of a segment-only doc: before=%d after=%d", before, e.writeGen)
	}
}
