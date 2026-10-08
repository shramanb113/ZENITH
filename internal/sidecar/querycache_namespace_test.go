package sidecar

import (
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// Review Finding (code review, 2026-10-08): addEngineFlags registers
// --query-cache-size/--query-cache-ttl/--query-cache-semantic-threshold/
// --ann-threshold-band-pct on `serve`, but runHTTP never passed any of them
// into sidecar.Config either — ephemeral namespaces silently ignored all
// four, the same gap internal/collections had.
func TestNamespaceOpts_PassesConfiguredQueryCacheSettings(t *testing.T) {
	s := New(Config{
		QueryCacheSize:              250,
		QueryCacheTTL:               90 * time.Second,
		QueryCacheSemanticThreshold: 0.9,
		ANNThresholdBandPct:         0.15,
	})

	snap, err := zenith.SnapshotOptions(s.namespaceOpts()...)
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

func TestNamespaceOpts_UnconfiguredQueryCacheSettingsLeaveEngineDefaults(t *testing.T) {
	s := New(Config{})

	snap, err := zenith.SnapshotOptions(s.namespaceOpts()...)
	if err != nil {
		t.Fatal(err)
	}
	if snap.QueryCacheSize != -1 {
		t.Fatalf("QueryCacheSize = %d, want -1 (unconfigured, engine default applies)", snap.QueryCacheSize)
	}
}
