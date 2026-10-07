package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// ── colour helpers ────────────────────────────────────────────────────────────

const (
	green  = "\033[32m"
	yellow = "\033[33m"
	red    = "\033[31m"
	cyan   = "\033[36m"
	bold   = "\033[1m"
	reset  = "\033[0m"
)

func pass(s string) string    { return green + s + reset }
func warn(s string) string    { return yellow + s + reset }
func fail(s string) string    { return red + s + reset }
func info(s string) string    { return cyan + s + reset }
func hilight(s string) string { return bold + s + reset }

// ── document corpus ───────────────────────────────────────────────────────────

var corpus = []struct {
	id   string
	text string
}{
	{"TECH-01", "The PageRank algorithm uses backlink structures to determine the perceived importance of web pages."},
	{"DATA-08", "Modern ranking systems prioritize various signals to ensure high-quality results."},
	{"LEGAL-03", "The relational database was revolutionary for its time, despite many relational anomalies."},
	{"AI-04", "Transformer ensembles often over-rely on lexical overlap instead of capturing deep semantic similarity."},
	{"ENV-06", "Global warming requires environmental solutions and atmospheric carbon capture."},
}

// ── test suite ────────────────────────────────────────────────────────────────

type trial struct {
	query    string
	expected string // expected top result ID
	category string // EXACT / FUZZY / PHONETIC / NEURAL
	reason   string
}

var suite = []trial{
	// ── EXACT / LEXICAL ──────────────────────────────────────────────────────
	{
		query:    "PageRank algorithm",
		expected: "TECH-01",
		category: "EXACT",
		reason:   "Verbatim key terms present in TECH-01",
	},
	{
		query:    "relational database",
		expected: "LEGAL-03",
		category: "EXACT",
		reason:   "Verbatim match in LEGAL-03",
	},
	{
		query:    "ranking signals",
		expected: "DATA-08",
		category: "EXACT",
		reason:   "Both stems present in DATA-08",
	},

	// ── FUZZY ────────────────────────────────────────────────────────────────
	{
		query:    "Transfomer", // 1 char missing
		expected: "AI-04",
		category: "FUZZY",
		reason:   "Levenshtein 1 — 'Transformer'",
	},
	{
		query:    "Pge Rank", // space typo
		expected: "TECH-01",
		category: "FUZZY",
		reason:   "Levenshtein 1 on 'Page'",
	},
	{
		query:    "warmng", // 2 char typo
		expected: "ENV-06",
		category: "FUZZY",
		reason:   "Levenshtein 2 — 'warming'",
	},

	// ── PHONETIC ─────────────────────────────────────────────────────────────
	{
		query:    "PageRanc", // c→k Soundex match
		expected: "TECH-01",
		category: "PHONETIC",
		reason:   "Soundex: PageRanc ≈ PageRank",
	},
	{
		query:    "Transfourmer",
		expected: "AI-04",
		category: "PHONETIC",
		reason:   "Soundex: Transfourmer ≈ Transformer",
	},

	// ── NEURAL / SEMANTIC ────────────────────────────────────────────────────
	{
		query:    "machine learning",
		expected: "AI-04",
		category: "NEURAL",
		reason:   "Semantic: 'machine learning' ≈ 'Transformer ensembles'",
	},
	{
		query:    "climate change",
		expected: "ENV-06",
		category: "NEURAL",
		reason:   "Semantic: 'climate' ≈ 'global warming'",
	},
	{
		query:    "graph search web",
		expected: "TECH-01",
		category: "NEURAL",
		reason:   "Semantic: 'graph' ≈ 'backlink', 'web pages'",
	},
	{
		query:    "SQL tables schema",
		expected: "LEGAL-03",
		category: "NEURAL",
		reason:   "Semantic: SQL domain ≈ 'relational database'",
	},
}

// ── result tracking ───────────────────────────────────────────────────────────

type result struct {
	trial
	topID    string
	topScore float64
	rank     int // 1-indexed rank of expected ID; -1 = not found
	latency  time.Duration
	err      error
}

func (r result) verdict() string {
	switch {
	case r.err != nil:
		return fail("ERROR")
	case r.rank == 1:
		return pass("✅ PASS")
	case r.rank > 1:
		return warn(fmt.Sprintf("⚠  RANK %d", r.rank))
	default:
		return fail("❌ MISS")
	}
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	addr := "localhost:8080"
	if len(os.Args) > 1 {
		addr = os.Args[1]
	}

	dialOpts := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	if key := os.Getenv("ZENITH_KEY"); key != "" {
		dialOpts = append(dialOpts, grpc.WithUnaryInterceptor(func(
			ctx context.Context, method string, req, reply any, cc *grpc.ClientConn,
			invoker grpc.UnaryInvoker, opts ...grpc.CallOption,
		) error {
			return invoker(metadata.AppendToOutgoingContext(ctx, "x-zenith-key", key), method, req, reply, cc, opts...)
		}))
	}

	conn, err := grpc.NewClient(addr, dialOpts...)
	if err != nil {
		log.Fatalf("gRPC dial failed: %v", err)
	}
	defer conn.Close()

	client := zenithproto.NewSearchServiceClient(conn)

	fmt.Println(hilight("\n══════════════════════════════════════════════"))
	fmt.Println(hilight("  ZENITH RANKING DIAGNOSTIC"))
	fmt.Println(hilight("══════════════════════════════════════════════"))

	// ── 1. INDEX ─────────────────────────────────────────────────────────────
	indexCorpus(client)

	// ── 2. RUN SUITE ─────────────────────────────────────────────────────────
	results := runSuite(client)

	// ── 3. REPORT ────────────────────────────────────────────────────────────
	printReport(results)
}

// ── indexing ──────────────────────────────────────────────────────────────────

func indexCorpus(client zenithproto.SearchServiceClient) {
	fmt.Printf("\n%s Indexing %d documents...\n", info("►"), len(corpus))

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "  ID\tStatus\tMessage")
	fmt.Fprintln(w, "  ──\t──────\t───────")

	for _, d := range corpus {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		resp, err := client.IndexDocuments(ctx, &zenithproto.IndexRequest{
			Id:   d.id,
			Data: d.text,
		})
		cancel()

		if err != nil {
			fmt.Fprintf(w, "  %s\t%s\t%v\n", d.id, fail("FAIL"), err)
			log.Fatalf("fatal: indexing %s failed: %v", d.id, err)
		}
		statusStr := pass("OK")
		if !resp.GetStatus() {
			statusStr = fail("FAIL")
		}
		fmt.Fprintf(w, "  %s\t%s\t%s\n", d.id, statusStr, resp.GetMessage())
	}
	w.Flush()
	fmt.Println()
}

// ── query suite ───────────────────────────────────────────────────────────────

func runSuite(client zenithproto.SearchServiceClient) []result {
	fmt.Printf("%s Running %d queries...\n\n", info("►"), len(suite))

	results := make([]result, 0, len(suite))

	for _, t := range suite {
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
		start := time.Now()
		resp, err := client.Search(ctx, &zenithproto.SearchRequest{Query: t.query})
		latency := time.Since(start)
		cancel()

		r := result{trial: t, latency: latency, err: err, rank: -1}

		if err == nil && len(resp.GetResults()) > 0 {
			r.topID = resp.Results[0].Id
			r.topScore = resp.Results[0].Score
			for i, res := range resp.Results {
				if res.Id == t.expected {
					r.rank = i + 1
					break
				}
			}
		}

		// Print inline as we go so you can watch live
		printInline(r, resp)
		results = append(results, r)
	}

	return results
}

func printInline(r result, resp *zenithproto.SearchResponse) {
	fmt.Printf("%s [%-8s] %s\n",
		r.verdict(),
		r.category,
		hilight(fmt.Sprintf("%q", r.query)),
	)
	fmt.Printf("         Expected: %s | Got: %s | Score: %.5f | Latency: %v\n",
		r.expected, orNA(r.topID), r.topScore, r.latency.Round(time.Millisecond),
	)
	fmt.Printf("         Reason: %s\n", r.reason)

	if resp != nil && len(resp.Results) > 0 {
		fmt.Printf("         Full ranking: ")
		parts := make([]string, 0, len(resp.Results))
		for i, res := range resp.Results {
			marker := ""
			if res.Id == r.expected {
				marker = "◀"
			}
			parts = append(parts, fmt.Sprintf("%d.%s(%.4f)%s", i+1, res.Id, res.Score, marker))
		}
		fmt.Println(strings.Join(parts, "  "))
	}
	fmt.Println()
}

// ── summary report ────────────────────────────────────────────────────────────

func printReport(results []result) {
	pass1, warned, missed, errored := 0, 0, 0, 0
	var totalLatency time.Duration

	byCategory := map[string][3]int{} // [pass, warn/miss, error]

	for _, r := range results {
		totalLatency += r.latency
		cat := r.category
		entry := byCategory[cat]
		switch {
		case r.err != nil:
			errored++
			entry[2]++
		case r.rank == 1:
			pass1++
			entry[0]++
		case r.rank > 1:
			warned++
			entry[1]++
		default:
			missed++
			entry[1]++
		}
		byCategory[cat] = entry
	}

	total := len(results)
	avg := totalLatency / time.Duration(total)

	fmt.Println(hilight("══════════════════════════════════════════════"))
	fmt.Println(hilight("  SUMMARY"))
	fmt.Println(hilight("══════════════════════════════════════════════"))

	w := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintf(w, "  Total queries:\t%d\n", total)
	fmt.Fprintf(w, "  %s:\t%d / %d  (%.0f%%)\n",
		pass("Rank-1 hits"), pass1, total, pct(pass1, total))
	fmt.Fprintf(w, "  %s:\t%d / %d\n",
		warn("Found (not rank-1)"), warned, total)
	fmt.Fprintf(w, "  %s:\t%d / %d\n",
		fail("Not found"), missed, total)
	if errored > 0 {
		fmt.Fprintf(w, "  %s:\t%d\n", fail("Errors"), errored)
	}
	fmt.Fprintf(w, "  Avg latency:\t%v\n", avg.Round(time.Millisecond))
	w.Flush()

	fmt.Println()
	fmt.Println(hilight("  BY CATEGORY"))

	w2 := tabwriter.NewWriter(os.Stdout, 0, 0, 3, ' ', 0)
	fmt.Fprintln(w2, "  Category\tPass\tWarn/Miss\tError")
	fmt.Fprintln(w2, "  ────────\t────\t─────────\t─────")
	for _, cat := range []string{"EXACT", "FUZZY", "PHONETIC", "NEURAL"} {
		e := byCategory[cat]
		fmt.Fprintf(w2, "  %s\t%s\t%s\t%s\n",
			cat,
			pass(fmt.Sprintf("%d", e[0])),
			warnOrFail(e[1]),
			failOrDash(e[2]),
		)
	}
	w2.Flush()

	fmt.Println()

	// Final verdict
	if pass1 == total {
		fmt.Println(pass("  ✅ ALL TESTS PASS — engine is clean"))
	} else if float64(pass1)/float64(total) >= 0.75 {
		fmt.Println(warn(fmt.Sprintf("  ⚠  %d/%d PASS — check WARN/MISS above", pass1, total)))
	} else {
		fmt.Println(fail(fmt.Sprintf("  ❌ POOR — only %d/%d rank-1 hits. Check pipeline.", pass1, total)))
	}

	fmt.Println(hilight("══════════════════════════════════════════════\n"))
}

// ── helpers ───────────────────────────────────────────────────────────────────

func orNA(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(n) / float64(total) * 100
}

func warnOrFail(n int) string {
	if n == 0 {
		return pass("0")
	}
	return warn(fmt.Sprintf("%d", n))
}

func failOrDash(n int) string {
	if n == 0 {
		return "-"
	}
	return fail(fmt.Sprintf("%d", n))
}
