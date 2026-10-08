// Package metrics exposes ZENITH's Prometheus metrics: a dedicated registry,
// the metric collectors themselves, and small wrappers (Route, gRPC
// interceptors, InstrumentEmbedder) that observe them at the few places
// requests and embeddings actually happen. See CLAUDE.md and README.md
// "Observability" for the full metric table and what each one means.
package metrics

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
)

// latencyBuckets spans 1ms to ~8.2s (prometheus.ExponentialBuckets(0.001, 2, 14)).
var latencyBuckets = prometheus.ExponentialBuckets(0.001, 2, 14)

// Registry is a dedicated registry (not prometheus.DefaultRegisterer) so
// ZENITH's /metrics endpoint carries exactly its own series plus the Go/
// process collectors, not whatever else a host process might register.
var Registry = prometheus.NewRegistry()

var (
	BuildInfo = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "zenith_build_info",
		Help: "Always 1; labels carry the running version and embedding model.",
	}, []string{"version", "model"})

	HTTPRequestsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_http_requests_total",
		Help: "HTTP requests by route and status code.",
	}, []string{"route", "code"})
	HTTPRequestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "zenith_http_request_duration_seconds",
		Help:    "HTTP request latency by route.",
		Buckets: latencyBuckets,
	}, []string{"route"})

	GRPCRequestsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_grpc_requests_total",
		Help: "gRPC requests by full method and status code name.",
	}, []string{"method", "code"})
	GRPCRequestDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "zenith_grpc_request_duration_seconds",
		Help:    "gRPC request latency by full method.",
		Buckets: latencyBuckets,
	}, []string{"method"})

	CollectionQueriesTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_collection_queries_total",
		Help: "Collection search queries by collection and outcome (ok/error).",
	}, []string{"collection", "outcome"})
	CollectionQueryDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "zenith_collection_query_duration_seconds",
		Help:    "Collection search query latency by collection.",
		Buckets: latencyBuckets,
	}, []string{"collection"})
	CollectionDocuments = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "zenith_collection_documents",
		Help: "Current document count by collection, from collection metadata.",
	}, []string{"collection"})
	CollectionDocsUpsertedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_collection_docs_upserted_total",
		Help: "Documents upserted by collection.",
	}, []string{"collection"})
	CollectionDocsDeletedTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_collection_docs_deleted_total",
		Help: "Documents deleted by collection.",
	}, []string{"collection"})
	CollectionDiskBytes = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "zenith_collection_disk_bytes",
		Help: "On-disk size of a collection's directory, by collection. Updated on Sweep.",
	}, []string{"collection"})
	CollectionWALBytes = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "zenith_collection_wal_bytes",
		Help: "Combined size of a collection's live and archived WAL files, by collection. A proxy for checkpoint activity: pkg/zenith exposes no checkpoint counter. Updated on Sweep.",
	}, []string{"collection"})
	Collections = prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "zenith_collections",
		Help: "Number of collections by state (open/closed).",
	}, []string{"state"})
	CollectionLifecycleTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_collection_lifecycle_total",
		Help: "Collection lifecycle events (created, deleted, opened, closed_idle, closed_lru, closed_shutdown).",
	}, []string{"event"})

	NamespaceQueriesTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "zenith_namespace_queries_total",
		Help: "Ephemeral namespace search queries.",
	})
	NamespaceQueryDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "zenith_namespace_query_duration_seconds",
		Help:    "Ephemeral namespace search query latency.",
		Buckets: latencyBuckets,
	})
	NamespacesActive = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "zenith_namespaces_active",
		Help: "Currently active ephemeral namespaces.",
	})
	NamespaceEvictionsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_namespace_evictions_total",
		Help: "Ephemeral namespace evictions by reason (ttl, lru, replaced, deleted, shutdown).",
	}, []string{"reason"})

	EmbeddingDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "zenith_embedding_duration_seconds",
		Help:    "Embedding call latency by op (document, batch, query).",
		Buckets: latencyBuckets,
	}, []string{"op"})
	EmbeddingTextsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_embedding_texts_total",
		Help: "Texts embedded by op (document, batch, query).",
	}, []string{"op"})

	QueryCacheHitsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_query_cache_hits_total",
		Help: "Query-result cache hits, by tier (l1, l2, semantic).",
	}, []string{"tier"})
	QueryCacheMissesTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "zenith_query_cache_misses_total",
		Help: "Query-result cache misses.",
	})

	ErrorsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "zenith_errors_total",
		Help: "Errors by surface (http/grpc/embedder) and code.",
	}, []string{"surface", "code"})
)

func init() {
	Registry.MustRegister(
		collectors.NewGoCollector(),
		collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}),
		BuildInfo,
		HTTPRequestsTotal, HTTPRequestDuration,
		GRPCRequestsTotal, GRPCRequestDuration,
		CollectionQueriesTotal, CollectionQueryDuration, CollectionDocuments,
		CollectionDocsUpsertedTotal, CollectionDocsDeletedTotal,
		CollectionDiskBytes, CollectionWALBytes, Collections, CollectionLifecycleTotal,
		NamespaceQueriesTotal, NamespaceQueryDuration, NamespacesActive, NamespaceEvictionsTotal,
		EmbeddingDuration, EmbeddingTextsTotal,
		QueryCacheHitsTotal, QueryCacheMissesTotal,
		ErrorsTotal,
	)
}

// RegisterIndexDocuments registers a GaugeFunc reporting the raw gRPC
// engine's live document count. gRPC-mode only: cmd/server/main.go and
// cmd/zenith/serve.go's --db path call this once with engine.Count(); the
// collections/HTTP path has no single raw engine, hence no series here.
func RegisterIndexDocuments(count func() float64) {
	Registry.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
		Name: "zenith_index_documents",
		Help: "Live document count of the raw --db engine (gRPC mode only).",
	}, count))
}

// ForgetCollection removes every series for a deleted collection's id label
// so /metrics does not accumulate unbounded cardinality from churn, and a
// deleted collection's stale numbers don't linger. Call it from
// collections.Manager.Delete after the entry is removed.
func ForgetCollection(id string) {
	labels := prometheus.Labels{"collection": id}
	CollectionQueriesTotal.DeletePartialMatch(labels)
	CollectionQueryDuration.DeletePartialMatch(labels)
	CollectionDocuments.DeletePartialMatch(labels)
	CollectionDocsUpsertedTotal.DeletePartialMatch(labels)
	CollectionDocsDeletedTotal.DeletePartialMatch(labels)
	CollectionDiskBytes.DeletePartialMatch(labels)
	CollectionWALBytes.DeletePartialMatch(labels)
}

// observeDuration is a small helper shared by Route/the embedder wrapper/the
// gRPC interceptors: record how long fn took against hist, return what fn
// returned.
func observeDuration(hist prometheus.Observer, fn func()) time.Duration {
	start := time.Now()
	fn()
	d := time.Since(start)
	hist.Observe(d.Seconds())
	return d
}
