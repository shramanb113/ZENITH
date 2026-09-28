// Command benchmark runs the ZENITH embeddable search benchmark against
// SQLite FTS5 and Bleve and prints a comparison table.
//
// Usage:
//
//	go run ./cmd/benchmark --scale=100k
//	go run ./cmd/benchmark --scale=1m --skip=sqlite
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/bench/internal/corpus"
	"github.com/shramanb113/ZENITH/bench/internal/engines"
	"github.com/shramanb113/ZENITH/bench/internal/metrics"
	"github.com/shramanb113/ZENITH/bench/internal/queryset"
	"github.com/shramanb113/ZENITH/bench/internal/report"
	"github.com/shramanb113/ZENITH/bench/internal/sysinfo"
)

func main() {
	scaleFlag := flag.String("scale", "100k", "corpus size: 100k or 1m")
	skipFlag := flag.String("skip", "", "comma-separated engines to skip: zenith-hybrid,zenith-bm25,sqlite,bleve")
	cacheDir := flag.String("cache", ".cache", "directory for cached dataset files")
	resultsDir := flag.String("results", "results", "directory for markdown output files")
	querySetFlag := flag.String("queryset", "clean",
		"query set: clean | typo (one edit per word, seeded) | codemixed (synthetic romanised-Hindi noise; a proxy, not real Hinglish data)")
	seedFlag := flag.Int64("seed", 1, "seed for typo/codemixed query generation (same seed = same queries)")
	flag.Parse()

	kind := queryset.Kind(strings.ToLower(*querySetFlag))
	if !kind.Valid() {
		fmt.Fprintf(os.Stderr, "unknown --queryset %q (use clean, typo or codemixed)\n", *querySetFlag)
		os.Exit(1)
	}

	limit := scaleToLimit(*scaleFlag)
	if limit == 0 {
		fmt.Fprintf(os.Stderr, "unknown --scale value %q (use 100k or 1m)\n", *scaleFlag)
		os.Exit(1)
	}

	skip := parseSkip(*skipFlag)
	ctx := context.Background()
	sys := sysinfo.Collect()

	fmt.Printf("Loading MS MARCO dataset (scale=%s)...\n", *scaleFlag)
	passages, queries, qrels, err := corpus.Load(*cacheDir, limit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "corpus: %v\n", err)
		os.Exit(1)
	}
	subset := corpus.SubsetIDs(passages)

	// Keep only queries that have a ground-truth entry in qrels.
	// The full queries file has 101K entries; only 6,980 have qrels.
	// Running all 101K multiplies query time by 14x with no recall benefit.
	filtered := queries[:0]
	for _, q := range queries {
		if _, ok := qrels[q.ID]; ok {
			filtered = append(filtered, q)
		}
	}
	queries = filtered

	// Perturb query text only; qrels (ground truth) still key on the original
	// query ID, so recall is graded against the same relevant passages.
	if kind != queryset.Clean {
		rng := rand.New(rand.NewSource(*seedFlag))
		for i := range queries {
			queries[i].Text = queryset.Apply(kind, queries[i].Text, rng)
		}
		fmt.Printf("Query set: %s (seed %d) — e.g. %q\n", kind, *seedFlag, queries[0].Text)
	}

	fmt.Printf("Loaded %d passages, %d queries (qrel-filtered), %d qrels\n", len(passages), len(queries), len(qrels))

	docs := make(map[string]string, len(passages))
	for _, p := range passages {
		docs[p.ID] = p.Text
	}

	var results []report.EngineResult

	results = append(results, runEngine(ctx, "zenith-hybrid", skip, func() (engines.Engine, error) {
		return engines.NewZenithHybrid()
	}, docs, queries, qrels, subset))

	results = append(results, runEngine(ctx, "zenith-bm25", skip, func() (engines.Engine, error) {
		return engines.NewZenithBM25Only()
	}, docs, queries, qrels, subset))

	results = append(results, runEngine(ctx, "sqlite", skip, func() (engines.Engine, error) {
		return engines.NewSQLiteEngine()
	}, docs, queries, qrels, subset))

	results = append(results, runEngine(ctx, "bleve", skip, func() (engines.Engine, error) {
		return engines.NewBleveEngine()
	}, docs, queries, qrels, subset))

	label := *scaleFlag
	if kind != queryset.Clean {
		label += "-" + string(kind)
	}
	report.Print(results, label, sys.String(), *resultsDir)
}

func runEngine(
	ctx context.Context,
	key string,
	skip map[string]bool,
	newFn func() (engines.Engine, error),
	docs map[string]string,
	queries []corpus.Query,
	qrels map[string]string,
	subset map[string]struct{},
) report.EngineResult {
	if skip[key] {
		return report.EngineResult{
			Name:       engineDisplayName(key),
			Skipped:    true,
			SkipReason: "excluded via --skip flag",
		}
	}

	fmt.Printf("\n[%s] initialising...\n", engineDisplayName(key))
	eng, err := newFn()
	if err != nil {
		return report.EngineResult{
			Name:       engineDisplayName(key),
			Skipped:    true,
			SkipReason: fmt.Sprintf("init failed: %v", err),
		}
	}
	defer eng.Close()

	// ── Indexing ──────────────────────────────────────────────────────────────
	fmt.Printf("[%s] indexing %d documents...\n", eng.Name(), len(docs))
	var memBefore, memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	indexStart := time.Now()
	if err := eng.IndexBatch(ctx, docs); err != nil {
		return report.EngineResult{
			Name:       eng.Name(),
			Skipped:    true,
			SkipReason: fmt.Sprintf("indexing failed: %v", err),
		}
	}
	indexTime := time.Since(indexStart)

	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	var heapDelta float64
	if memAfter.HeapAlloc >= memBefore.HeapAlloc {
		heapDelta = float64(memAfter.HeapAlloc-memBefore.HeapAlloc) / 1e6
	} else {
		heapDelta = -float64(memBefore.HeapAlloc-memAfter.HeapAlloc) / 1e6
	}
	fmt.Printf("[%s] indexed in %s, heap delta %.0f MB\n", eng.Name(), fmtDur(indexTime), heapDelta)

	// ── Querying + Recall ─────────────────────────────────────────────────────
	fmt.Printf("[%s] running %d queries...\n", eng.Name(), len(queries))
	latencies := make([]time.Duration, 0, len(queries))
	queryResults := make(map[string][]string, len(queries))

	for _, q := range queries {
		start := time.Now()
		ids, err := eng.Search(ctx, q.Text, 10)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Fprintf(os.Stderr, "[%s] query %q failed: %v\n", eng.Name(), q.ID, err)
			continue
		}
		latencies = append(latencies, elapsed)
		queryResults[q.ID] = ids
	}

	p50, p95, p99 := metrics.Percentiles(latencies)
	recall := metrics.RecallAt10(queryResults, qrels, subset)
	fmt.Printf("[%s] Recall@10=%.3f  p50=%s p95=%s p99=%s\n",
		eng.Name(), recall, fmtDur(p50), fmtDur(p95), fmtDur(p99))

	return report.EngineResult{
		Name:        eng.Name(),
		Recall10:    recall,
		P50:         p50,
		P95:         p95,
		P99:         p99,
		IndexTime:   indexTime,
		HeapDeltaMB: heapDelta,
	}
}

func scaleToLimit(s string) int {
	switch strings.ToLower(s) {
	case "100k":
		return 100_000
	case "1m":
		return 1_000_000
	default:
		return 0
	}
}

func parseSkip(s string) map[string]bool {
	m := make(map[string]bool)
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(strings.ToLower(part))
		if part != "" {
			m[part] = true
		}
	}
	return m
}

func engineDisplayName(key string) string {
	switch key {
	case "zenith-hybrid":
		return "ZENITH hybrid"
	case "zenith-bm25":
		return "ZENITH BM25"
	case "sqlite":
		return "SQLite FTS5"
	case "bleve":
		return "Bleve"
	default:
		return key
	}
}

func fmtDur(d time.Duration) string {
	switch {
	case d < time.Millisecond:
		return fmt.Sprintf("%.1fµs", float64(d.Microseconds()))
	case d < time.Second:
		return fmt.Sprintf("%.1fms", float64(d.Milliseconds()))
	default:
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
}
