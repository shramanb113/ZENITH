package collections

import (
	"testing"

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
