package config

import "testing"

func TestDefaultConfig_QueryCacheDefaults(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.QueryCacheSize != 1000 {
		t.Errorf("QueryCacheSize = %d, want 1000", cfg.QueryCacheSize)
	}
	if cfg.QueryCacheTTL.Minutes() != 5 {
		t.Errorf("QueryCacheTTL = %v, want 5m", cfg.QueryCacheTTL)
	}
	if cfg.QueryCacheRedisAddr != "" {
		t.Errorf("QueryCacheRedisAddr = %q, want \"\" (L2 disabled by default)", cfg.QueryCacheRedisAddr)
	}
	if cfg.QueryCacheNamespace != "" {
		t.Errorf("QueryCacheNamespace = %q, want \"\"", cfg.QueryCacheNamespace)
	}
	if cfg.QueryCacheSemanticThreshold != 0 {
		t.Errorf("QueryCacheSemanticThreshold = %v, want 0 (disabled by default)", cfg.QueryCacheSemanticThreshold)
	}
	if cfg.ANNThresholdBandPct != 0 {
		t.Errorf("ANNThresholdBandPct = %v, want 0 (disabled by default)", cfg.ANNThresholdBandPct)
	}
}
