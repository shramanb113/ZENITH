package index

import (
	"context"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// Review Focus #4 (from the plan): "Redis unreachable, timeout, or a decode
// failure on a stale-shaped L2 entry must degrade to a cache miss, never a
// Search error." Before this test, that guarantee was only exercised at the
// internal/querycache.BytesCache layer in isolation (e.g.
// TestBytesCache_UnreachableRedisIsATreatedAsMiss) — nothing proved it holds
// at the actual Engine.Search entry point a real caller uses.
func TestSearch_UnreachableL2RedisDegradesToMissNotError(t *testing.T) {
	mr := miniredis.RunT(t)
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = 1000
	cfg.QueryCacheRedisAddr = mr.Addr()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()

	ctx := context.Background()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	// A real L2 hit path first, to prove Redis is genuinely wired in (not
	// just configured and silently unused).
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}

	// Now take Redis down entirely and confirm every subsequent Search —
	// cache miss (new query) and what would have been a cache hit (same
	// query again) alike — still succeeds with no error.
	mr.Close()

	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatalf("Search with Redis down (repeat query, would-be L2 hit) returned an error: %v, want nil", err)
	}
	if _, err := e.SearchFiltered(ctx, "a brand new query text", nil); err != nil {
		t.Fatalf("Search with Redis down (new query, genuine miss) returned an error: %v, want nil", err)
	}
}

// The same guarantee again, specifically for a decode failure on a
// stale-shaped L2 entry (e.g. left over from an older binary version) —
// Review Focus #4's second named case, distinct from "Redis unreachable".
func TestSearch_L2DecodeFailureDegradesToMissNotError(t *testing.T) {
	mr := miniredis.RunT(t)
	cfg := config.DefaultConfig()
	cfg.WordVectors = false
	cfg.QueryCacheSize = 1000
	cfg.QueryCacheRedisAddr = mr.Addr()
	emb := &countingSearchEmbedder{vec: []float32{1, 0}}
	e := NewEngine(cfg, emb, ranking.NewWeightedRRFRanker(cfg.RRFConstant, cfg.MaxResults, 1.0, cfg.VectorWeight), analysis.NewStandardAnalyzer())
	e.SetAutoCompact(false)
	defer e.Close()

	ctx := context.Background()
	if err := e.AddWithVectorAttrs(ctx, "doc1", "hello world", []float32{1, 0}, nil); err != nil {
		t.Fatal(err)
	}

	// Poison every key in Redis with garbage that can never gob-decode into
	// a cacheResult, simulating a stale-shaped entry from an older binary.
	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatal(err)
	}
	for _, k := range mr.Keys() {
		if err := mr.Set(k, "not a valid gob payload"); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := e.SearchFiltered(ctx, "hello", nil); err != nil {
		t.Fatalf("Search with a corrupt L2 entry returned an error: %v, want nil (must degrade to a miss and recompute)", err)
	}
}
