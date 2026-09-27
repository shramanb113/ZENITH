package zenith_test

import (
	"errors"
	"fmt"
	"testing"

	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// Each option is tested by passing it to Open and checking whether an
// error is returned — that's the only observable effect from outside the package.

func TestWithLimit_Positive(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithLimit(5))
	if err != nil {
		t.Fatalf("WithLimit(5) should be valid: %v", err)
	}
	db.Close()
}

func TestWithLimit_Zero(t *testing.T) {
	_, err := zenith.Open(":memory:", zenith.WithLimit(0))
	if err == nil {
		t.Fatal("WithLimit(0) should return an error")
	}
}

func TestWithLimit_Negative(t *testing.T) {
	_, err := zenith.Open(":memory:", zenith.WithLimit(-1))
	if err == nil {
		t.Fatal("WithLimit(-1) should return an error")
	}
}

func TestWithCacheSize_Positive(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithCacheSize(100))
	if err != nil {
		t.Fatalf("WithCacheSize(100) should be valid: %v", err)
	}
	db.Close()
}

func TestWithCacheSize_Zero(t *testing.T) {
	// 0 disables the cache — valid
	db, err := zenith.Open(":memory:", zenith.WithCacheSize(0))
	if err != nil {
		t.Fatalf("WithCacheSize(0) should be valid: %v", err)
	}
	db.Close()
}

func TestWithCacheSize_Negative(t *testing.T) {
	_, err := zenith.Open(":memory:", zenith.WithCacheSize(-1))
	if err == nil {
		t.Fatal("WithCacheSize(-1) should return an error")
	}
}

func TestWithFuzzyDistance_Valid(t *testing.T) {
	for _, d := range []int{0, 1, 2, 3, 4, 5} {
		db, err := zenith.Open(":memory:", zenith.WithFuzzyDistance(d))
		if err != nil {
			t.Fatalf("WithFuzzyDistance(%d) should be valid: %v", d, err)
		}
		db.Close()
	}
}

func TestWithFuzzyDistance_Negative(t *testing.T) {
	_, err := zenith.Open(":memory:", zenith.WithFuzzyDistance(-1))
	if err == nil {
		t.Fatal("WithFuzzyDistance(-1) should return an error")
	}
}

func TestWithFuzzyDistance_AboveMax_Clamped(t *testing.T) {
	// Values above 5 are clamped, not rejected
	db, err := zenith.Open(":memory:", zenith.WithFuzzyDistance(100))
	if err != nil {
		t.Fatalf("WithFuzzyDistance(100) should be clamped, not error: %v", err)
	}
	db.Close()
}

func TestWithBM25Only(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("WithBM25Only() should be valid: %v", err)
	}
	db.Close()
}

func TestWithEmbedder_Nil(t *testing.T) {
	_, err := zenith.Open(":memory:", zenith.WithEmbedder(nil))
	if err == nil {
		t.Fatal("WithEmbedder(nil) should return an error")
	}
}

func TestWithEmbedder_Custom(t *testing.T) {
	emb := embedding.NewDeterministicEmbedder(384)
	db, err := zenith.Open(":memory:", zenith.WithEmbedder(emb))
	if err != nil {
		t.Fatalf("WithEmbedder(valid) should be valid: %v", err)
	}
	db.Close()
}

func TestMultipleOptions(t *testing.T) {
	emb := embedding.NewDeterministicEmbedder(384)
	db, err := zenith.Open(":memory:",
		zenith.WithEmbedder(emb),
		zenith.WithLimit(20),
		zenith.WithFuzzyDistance(1),
		zenith.WithCacheSize(0),
	)
	if err != nil {
		t.Fatalf("multiple valid options should not error: %v", err)
	}
	db.Close()
}

func TestLimitSearchOption_Positive(t *testing.T) {
	// Limit as a SearchOption — just verify it doesn't break Search
	db, err := zenith.Open(":memory:", zenith.WithBM25Only())
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := bgCtx()
	db.Add(ctx, "a", "hello world")
	db.Add(ctx, "b", "hello world again")

	results, err := db.Search(ctx, "hello", zenith.Limit(1))
	if err != nil {
		t.Fatalf("Search with Limit option: %v", err)
	}
	if len(results) > 1 {
		t.Fatalf("Limit(1) should cap at 1 result, got %d", len(results))
	}
}

func TestWithMemoryLimit_ZeroOrNegative(t *testing.T) {
	if _, err := zenith.Open(":memory:", zenith.WithMemoryLimit(0)); err == nil {
		t.Fatal("WithMemoryLimit(0) should return an error")
	}
	if _, err := zenith.Open(":memory:", zenith.WithMemoryLimit(-1)); err == nil {
		t.Fatal("WithMemoryLimit(-1) should return an error")
	}
}

func TestWithMemoryLimit_RejectsGrowthPastLimit(t *testing.T) {
	// ~11KB/doc estimate; 30KB fits ~2 docs before Add starts returning ErrIndexFull.
	db, err := zenith.Open(":memory:", zenith.WithBM25Only(), zenith.WithMemoryLimit(30*1024))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := bgCtx()
	var lastErr error
	added := 0
	for i := 0; i < 10; i++ {
		lastErr = db.Add(ctx, fmt.Sprintf("doc%d", i), "some content")
		if lastErr != nil {
			break
		}
		added++
	}
	if added == 0 || added >= 10 {
		t.Fatalf("expected the limit to be hit partway through, added %d docs before error", added)
	}
	if !errors.Is(lastErr, zenith.ErrIndexFull) {
		t.Fatalf("expected ErrIndexFull once the memory limit is exceeded, got %v", lastErr)
	}
}

func TestWithMemoryLimit_AllowsUpdatingExistingDocAtLimit(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithBM25Only(), zenith.WithMemoryLimit(30*1024))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer db.Close()

	ctx := bgCtx()
	for i := 0; i < 10; i++ {
		if db.Add(ctx, fmt.Sprintf("doc%d", i), "some content") != nil {
			break // limit reached; fine, we just need at least one doc indexed
		}
	}

	if err := db.Add(ctx, "doc0", "updated content"); err != nil {
		t.Fatalf("updating an existing document at the limit should not fail: %v", err)
	}
}
