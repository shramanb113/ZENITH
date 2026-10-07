package sidecar

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/shramanb113/ZENITH/internal/collections"
	"github.com/shramanb113/ZENITH/internal/embedding"
	"github.com/shramanb113/ZENITH/internal/localembedder"
)

// TestConcurrentQPS drives N concurrent goroutines against the real HTTP
// sidecar's persistent collections, each repeatedly searching, and reports
// aggregate QPS plus p50/p95/p99 latency under load.
//
// Nothing like this existed before this test: every other benchmark in this
// repo (TestMSMARCOHybrid, TestHybridEngineLatency, BenchmarkEmbed*, ...)
// measures single-goroutine sequential latency, never concurrent-client
// throughput. This is the baseline for the embedding-cache de-dup work
// (see ROADMAP.md): run once before and once after that change for a real
// before/after number, and once with ZENITH_QPS_EMBEDDER=local to see the
// actual process-wide bottleneck — one onnxModel mutex shared by every
// tenant (internal/localembedder/model.go), since collections.Manager and
// sidecar.Server both hold the same *Embedder instance — in real numbers
// instead of by inspection.
//
// Deliberately scoped to the HTTP collections path, not gRPC: persistent
// collections (the actual multi-tenant surface) are HTTP-only; the gRPC
// --db path is a single engine, single tenant, so "concurrent-tenant QPS"
// doesn't apply to it the same way. A gRPC throughput harness would measure
// something different (single-engine concurrent-request handling) and is
// left for a separate pass if that's ever the question.
//
//	ZENITH_QPS=1 go test ./internal/sidecar -run TestConcurrentQPS -v -timeout 5m
//	ZENITH_QPS=1 ZENITH_QPS_EMBEDDER=local go test ./internal/sidecar -run TestConcurrentQPS -v -timeout 5m
//
// Tunables (env, all optional): ZENITH_QPS_CLIENTS (default 32),
// ZENITH_QPS_COLLECTIONS (default 8 — also exercises concurrent-tenant
// capacity, not just raw QPS), ZENITH_QPS_DURATION (default 5s, Go duration
// syntax), ZENITH_QPS_EMBEDDER=local (default: deterministic, CGo-free,
// stable, isolates the HTTP/locking path from the ONNX bottleneck).
func TestConcurrentQPS(t *testing.T) {
	if os.Getenv("ZENITH_QPS") == "" {
		t.Skip("set ZENITH_QPS=1 to run")
	}

	clients := envInt(t, "ZENITH_QPS_CLIENTS", 32)
	nCollections := envInt(t, "ZENITH_QPS_COLLECTIONS", 8)
	duration := envDuration(t, "ZENITH_QPS_DURATION", 5*time.Second)

	var emb embedding.Embedder
	embLabel := "deterministic"
	if os.Getenv("ZENITH_QPS_EMBEDDER") == "local" {
		embLabel = "local"
		le, err := localembedder.NewByID("", "")
		if err != nil {
			t.Skipf("local embedder unavailable: %v (build with CGO_ENABLED=1)", err)
		}
		cached, cerr := embedding.NewCachingEmbedder(le, 10_000)
		if cerr != nil {
			t.Fatalf("caching embedder: %v", cerr)
		}
		emb = cached
	} else {
		emb = embedding.NewDeterministicEmbedder(384)
	}

	dir := t.TempDir()
	mgr, err := collections.New(collections.Config{Root: dir, Embedder: emb, MaxOpen: nCollections})
	if err != nil {
		t.Fatalf("collections.New: %v", err)
	}
	t.Cleanup(func() { _ = mgr.CloseAll() })

	_, ts, _, _ := newTest(t, func(c *Config) { c.Collections = mgr })

	// Pre-create and seed every tenant up front: this measures query
	// throughput under load, not ingest/index-build cost. A collection key
	// is required on every data-plane call even when the admin key (unset
	// here) is empty, so each tenant's key is captured at creation.
	ids := make([]string, nCollections)
	keys := make([]string, nCollections)
	for i := range ids {
		id := fmt.Sprintf("tenant-%d", i)
		ids[i] = id
		resp, body := do(t, ts, "POST", "/v1/collections", map[string]any{"id": id}, nil)
		if resp.StatusCode != 201 {
			t.Fatalf("create %s: %d %v", id, resp.StatusCode, body)
		}
		key, _ := body["key"].(string)
		keys[i] = key
		hdr := map[string]string{"X-Zenith-Key": key}
		docs := make([]map[string]string, 20)
		for j := range docs {
			docs[j] = map[string]string{"id": fmt.Sprintf("d%d", j), "text": fmt.Sprintf("tenant %d document %d about search and retrieval systems", i, j)}
		}
		if resp, body := do(t, ts, "PUT", "/v1/collections/"+id+"/docs", map[string]any{"docs": docs}, hdr); resp.StatusCode != 200 {
			t.Fatalf("seed %s: %d %v", id, resp.StatusCode, body)
		}
	}

	// A higher per-host connection cap than the net/http default (2) so the
	// measurement isn't an artifact of client-side connection-pool
	// contention instead of the server-side path under test.
	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: clients}}
	unique := os.Getenv("ZENITH_QPS_UNIQUE_QUERIES") != ""
	fixedBody, _ := json.Marshal(map[string]any{
		"queries": []map[string]string{{"id": "q", "text": "search and retrieval"}},
		"limit":   10,
	})
	// A never-repeated query text can never be a cache hit, however good
	// the cache is — this isolates the ONNX mutex itself (the ceiling Q3
	// documents) from the repeated-query case the default mode measures.
	bodyFor := func(worker int, n int64) []byte {
		if !unique {
			return fixedBody
		}
		b, _ := json.Marshal(map[string]any{
			"queries": []map[string]string{{"id": "q", "text": fmt.Sprintf("unique query worker %d seq %d", worker, n)}},
			"limit":   10,
		})
		return b
	}

	var (
		wg    sync.WaitGroup
		total atomic.Int64
		errs  atomic.Int64
		latMu sync.Mutex
	)
	// Preallocate: clients*expected-iterations is unknown up front, but a
	// generous estimate avoids repeated slice growth from every worker
	// appending under the same lock during the hot loop.
	latencies := make([]time.Duration, 0, clients*1024)
	stop := make(chan struct{})
	timer := time.AfterFunc(duration, func() { close(stop) })
	defer timer.Stop()

	start := time.Now()
	for c := 0; c < clients; c++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			idx := worker % nCollections
			url := ts.URL + "/v1/collections/" + ids[idx] + "/search"
			key := keys[idx]
			var n int64
			for {
				select {
				case <-stop:
					return
				default:
				}
				n++
				t0 := time.Now()
				req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(bodyFor(worker, n)))
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("X-Zenith-Key", key)
				resp, err := client.Do(req)
				d := time.Since(t0)
				if err != nil {
					errs.Add(1)
					continue
				}
				_, _ = io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode != 200 {
					errs.Add(1)
					continue
				}
				total.Add(1)
				latMu.Lock()
				latencies = append(latencies, d)
				latMu.Unlock()
			}
		}(c)
	}
	wg.Wait()
	elapsed := time.Since(start)

	sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
	pct := func(p float64) time.Duration {
		if len(latencies) == 0 {
			return 0
		}
		idx := int(p * float64(len(latencies)))
		if idx >= len(latencies) {
			idx = len(latencies) - 1
		}
		return latencies[idx]
	}
	qps := float64(total.Load()) / elapsed.Seconds()
	queryMode := "repeated"
	if unique {
		queryMode = "unique"
	}
	line := fmt.Sprintf("clients=%d collections=%d embedder=%-13s queries=%-8s %7.1f QPS (n=%d, %d errors)  p50=%-9s p95=%-9s p99=%s",
		clients, nCollections, embLabel, queryMode, qps, total.Load(), errs.Load(),
		pct(0.50).Round(100*time.Microsecond), pct(0.95).Round(100*time.Microsecond), pct(0.99).Round(100*time.Microsecond))
	t.Log(line)
	fmt.Fprintln(os.Stderr, "QPS "+line)
	if errs.Load() > 0 {
		t.Errorf("%d request errors during the run", errs.Load())
	}
}

func envInt(t *testing.T, key string, def int) int {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		t.Fatalf("%s: invalid int %q: %v", key, v, err)
	}
	return n
}

func envDuration(t *testing.T, key string, def time.Duration) time.Duration {
	t.Helper()
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		t.Fatalf("%s: invalid duration %q: %v", key, v, err)
	}
	return d
}
