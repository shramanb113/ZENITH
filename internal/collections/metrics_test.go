package collections

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/shramanb113/ZENITH/internal/metrics"
)

// After N searches on a collection, zenith_collection_queries_total carries
// exactly that many "ok" observations; after the collection is deleted, its
// series is gone (metrics.ForgetCollection ran) rather than lingering with
// stale numbers.
func TestManager_SearchAndDeleteMetrics(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()

	const id = "metrics-search-delete"
	if _, _, err := m.Create(id, CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if _, err := m.Upsert(ctx, id, map[string]string{"d1": "hello world"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}

	for i := 0; i < 3; i++ {
		if _, err := m.Search(ctx, id, "hello"); err != nil {
			t.Fatalf("Search #%d: %v", i, err)
		}
	}
	if got := testutil.ToFloat64(metrics.CollectionQueriesTotal.WithLabelValues(id, "ok")); got != 3 {
		t.Fatalf("queries_total{%s,ok} = %v, want 3", id, got)
	}

	before := testutil.CollectAndCount(metrics.CollectionQueriesTotal)
	if err := m.Delete(id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	after := testutil.CollectAndCount(metrics.CollectionQueriesTotal)
	if after >= before {
		t.Fatalf("queries_total series count went from %d to %d after Delete, want fewer", before, after)
	}
}

// Create/Upsert/DeleteDoc keep zenith_collection_documents in sync with the
// collection's real document count.
func TestManager_DocumentsGaugeTracksCount(t *testing.T) {
	m, _ := newTestManager(t, nil)
	ctx := context.Background()

	const id = "metrics-doc-gauge"
	if _, _, err := m.Create(id, CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if got := testutil.ToFloat64(metrics.CollectionDocuments.WithLabelValues(id)); got != 0 {
		t.Fatalf("documents{%s} after Create = %v, want 0", id, got)
	}

	if _, err := m.Upsert(ctx, id, map[string]string{"d1": "x", "d2": "y"}, nil); err != nil {
		t.Fatalf("Upsert: %v", err)
	}
	if got := testutil.ToFloat64(metrics.CollectionDocuments.WithLabelValues(id)); got != 2 {
		t.Fatalf("documents{%s} after Upsert(2) = %v, want 2", id, got)
	}

	if err := m.DeleteDoc(ctx, id, "d1"); err != nil {
		t.Fatalf("DeleteDoc: %v", err)
	}
	if got := testutil.ToFloat64(metrics.CollectionDocuments.WithLabelValues(id)); got != 1 {
		t.Fatalf("documents{%s} after DeleteDoc = %v, want 1", id, got)
	}
}

// Lifecycle events are counted at creation and at an explicit Delete.
func TestManager_LifecycleEventsCounted(t *testing.T) {
	m, _ := newTestManager(t, nil)

	const id = "metrics-lifecycle"
	createdBefore := testutil.ToFloat64(metrics.CollectionLifecycleTotal.WithLabelValues("created"))
	deletedBefore := testutil.ToFloat64(metrics.CollectionLifecycleTotal.WithLabelValues("deleted"))

	if _, _, err := m.Create(id, CreateOptions{}); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if got := testutil.ToFloat64(metrics.CollectionLifecycleTotal.WithLabelValues("created")); got != createdBefore+1 {
		t.Fatalf("lifecycle_total{created} = %v, want %v", got, createdBefore+1)
	}

	if err := m.Delete(id); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if got := testutil.ToFloat64(metrics.CollectionLifecycleTotal.WithLabelValues("deleted")); got != deletedBefore+1 {
		t.Fatalf("lifecycle_total{deleted} = %v, want %v", got, deletedBefore+1)
	}
}
