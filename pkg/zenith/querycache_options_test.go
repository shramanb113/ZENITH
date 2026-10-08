package zenith

import (
	"context"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/index"
)

type fakeDBCacheObserver struct {
	hits   map[string]int
	misses int
}

func (f *fakeDBCacheObserver) ObserveQueryCacheHit(tier string) {
	if f.hits == nil {
		f.hits = map[string]int{}
	}
	f.hits[tier]++
}
func (f *fakeDBCacheObserver) ObserveQueryCacheMiss() { f.misses++ }

func TestWithQueryCacheObserver_ReceivesHitsAndMisses(t *testing.T) {
	ctx := context.Background()
	obs := &fakeDBCacheObserver{}
	db, err := Open(":memory:", WithBM25Only(), WithQueryCacheObserver(obs))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	if err := db.Add(ctx, "doc1", "hello world"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Search(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	if obs.misses != 1 {
		t.Fatalf("misses = %d after first (uncached) search, want 1", obs.misses)
	}
	if _, err := db.Search(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	if obs.hits["l1"] != 1 {
		t.Fatalf("hits[l1] = %d after second (cached) search, want 1; hits=%v", obs.hits["l1"], obs.hits)
	}
}

var _ index.CacheObserver = (*fakeDBCacheObserver)(nil)

func TestWithQueryCacheSize_ZeroDisablesCache(t *testing.T) {
	ctx := context.Background()
	db, err := Open(":memory:", WithBM25Only(), WithQueryCacheSize(0))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.Add(ctx, "doc1", "hello world"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Search(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	if db.engine.CacheEnabled() {
		t.Fatal("engine cache is enabled after WithQueryCacheSize(0)")
	}
}

func TestWithQueryCacheSize_RejectsNegative(t *testing.T) {
	if _, err := Open(":memory:", WithBM25Only(), WithQueryCacheSize(-1)); err == nil {
		t.Fatal("WithQueryCacheSize(-1) accepted, want ErrInvalidOption")
	}
}

func TestWithQueryCacheSize_DefaultLeavesConfigDefaultInPlace(t *testing.T) {
	db, err := Open(":memory:", WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if !db.engine.CacheEnabled() {
		t.Fatal("engine cache is disabled with no WithQueryCacheSize override — the Config default (1000) should apply")
	}
}

func TestWithQueryCacheSemanticThreshold_Applied(t *testing.T) {
	db, err := Open(":memory:", WithBM25Only(), WithQueryCacheSemanticThreshold(0.95))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := db.engine.Config().QueryCacheSemanticThreshold; got != 0.95 {
		t.Fatalf("QueryCacheSemanticThreshold = %v, want 0.95", got)
	}
}

func TestWithANNThresholdBand_Applied(t *testing.T) {
	db, err := Open(":memory:", WithBM25Only(), WithANNThresholdBand(0.2))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := db.engine.Config().ANNThresholdBandPct; got != 0.2 {
		t.Fatalf("ANNThresholdBandPct = %v, want 0.2", got)
	}
}

func TestWithQueryCacheRedisAddrAndNamespace_Applied(t *testing.T) {
	db, err := Open(":memory:", WithBM25Only(), WithQueryCacheNamespace("tenant-1"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := db.engine.Config().QueryCacheNamespace; got != "tenant-1" {
		t.Fatalf("QueryCacheNamespace = %q, want %q", got, "tenant-1")
	}
	// Redis addr itself is exercised against a real/fake Redis in
	// internal/querycache's own tests (Task 2) and internal/collections'
	// (Task 13) — this only confirms the Option reaches Config.
}

func TestWithQueryCacheTTL_Applied(t *testing.T) {
	db, err := Open(":memory:", WithBM25Only(), WithQueryCacheTTL(90*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if got := db.engine.Config().QueryCacheTTL; got != 90*time.Second {
		t.Fatalf("QueryCacheTTL = %v, want 90s", got)
	}
}
