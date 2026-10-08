package collections

import (
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

func TestOpenOpts_IncludesNamespaceMatchingCollectionID(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseAll()

	snap, err := zenith.SnapshotOptions(m.openOpts("my-collection-id")...)
	if err != nil {
		t.Fatal(err)
	}
	if snap.QueryCacheNamespace != "my-collection-id" {
		t.Fatalf("QueryCacheNamespace = %q, want %q", snap.QueryCacheNamespace, "my-collection-id")
	}
}

func TestOpenOpts_PassesConfiguredRedisAddr(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{Root: dir, QueryCacheRedisAddr: "localhost:6379"})
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseAll()

	snap, err := zenith.SnapshotOptions(m.openOpts("id-a")...)
	if err != nil {
		t.Fatal(err)
	}
	if snap.QueryCacheRedisAddr != "localhost:6379" {
		t.Fatalf("QueryCacheRedisAddr = %q, want %q", snap.QueryCacheRedisAddr, "localhost:6379")
	}
}

func TestOpenOpts_NoRedisAddrConfiguredLeavesItUnset(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseAll()

	snap, err := zenith.SnapshotOptions(m.openOpts("id-a")...)
	if err != nil {
		t.Fatal(err)
	}
	if snap.QueryCacheRedisAddr != "" {
		t.Fatalf("QueryCacheRedisAddr = %q, want \"\" (no Redis configured for this Manager)", snap.QueryCacheRedisAddr)
	}
}

// Review Finding (code review, 2026-10-08): addEngineFlags registers
// --query-cache-size/--query-cache-ttl/--query-cache-semantic-threshold/
// --ann-threshold-band-pct on `serve`, but runHTTP only ever passed
// --collection-query-cache-redis-addr through to collections.Config -- the
// other four silently did nothing in --http mode despite being documented
// as applying to it.
func TestOpenOpts_PassesConfiguredQueryCacheSizeTTLAndAdaptiveSettings(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{
		Root:                        dir,
		QueryCacheSize:              250,
		QueryCacheTTL:               90 * time.Second,
		QueryCacheSemanticThreshold: 0.9,
		ANNThresholdBandPct:         0.15,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseAll()

	snap, err := zenith.SnapshotOptions(m.openOpts("id-a")...)
	if err != nil {
		t.Fatal(err)
	}
	if snap.QueryCacheSize != 250 {
		t.Fatalf("QueryCacheSize = %d, want 250", snap.QueryCacheSize)
	}
	if snap.QueryCacheTTL != 90*time.Second {
		t.Fatalf("QueryCacheTTL = %v, want 90s", snap.QueryCacheTTL)
	}
	if snap.QueryCacheSemanticThreshold != 0.9 {
		t.Fatalf("QueryCacheSemanticThreshold = %v, want 0.9", snap.QueryCacheSemanticThreshold)
	}
	if snap.ANNThresholdBandPct != 0.15 {
		t.Fatalf("ANNThresholdBandPct = %v, want 0.15", snap.ANNThresholdBandPct)
	}
}

func TestOpenOpts_UnconfiguredQueryCacheSettingsLeaveEngineDefaults(t *testing.T) {
	dir := t.TempDir()
	m, err := New(Config{Root: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseAll()

	snap, err := zenith.SnapshotOptions(m.openOpts("id-a")...)
	if err != nil {
		t.Fatal(err)
	}
	// -1 is zenith's own "unconfigured, engine/Config default applies"
	// sentinel (see zenith.WithQueryCacheSize): openOpts must only call
	// WithQueryCacheSize when the Manager was actually configured with a
	// size, never pass a forced-off 0 by default.
	if snap.QueryCacheSize != -1 {
		t.Fatalf("QueryCacheSize = %d, want -1 (unconfigured, engine default applies)", snap.QueryCacheSize)
	}
}
