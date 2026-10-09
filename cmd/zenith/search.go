package main

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/reranker"
	"github.com/spf13/cobra"
)

var searchFlags struct {
	maxResults  int
	offset      int
	where       []string
	filter      string
	rerank      bool
	rerankModel string

	sortField string
	sortDesc  bool
	explain   bool

	vectorWeight   float64
	phoneticWeight float64
	rrfK           float64

	facets    []string
	facetTopK int
}

var searchCmd = &cobra.Command{
	Use:   "search <query>",
	Short: "Query the local index",
	Long: `Loads the local index from zenith.db and runs a hybrid search query.
No server needed — the engine runs in-process.

The query goes through the full pipeline:
  1. Lexical scoring  (BM25 + edge n-grams + phonetic)
  2. Fuzzy matching   (BK-tree Levenshtein)
  3. Semantic scoring (vector cosine via embedder)
  4. RRF fusion

A double-quoted part of the query is a phrase every result must contain, words
adjacent and in order (after the same stemming and stop-word removal as the
documents). Keep the quotes away from your shell:

  zenith search '"machine learning" python'

--facets counts attribute values over every match (not just the page shown).
With --facets and no query, it counts over every document in the index.`,

	Args: func(cmd *cobra.Command, args []string) error {
		if len(args) == 0 && len(searchFlags.facets) == 0 {
			return errors.New("requires a query (or --facets with no query, for corpus-wide counts)")
		}
		return nil
	},
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		query := strings.Join(args, " ")

		filter, err := buildFilter(searchFlags.where, searchFlags.filter)
		if err != nil {
			return err
		}
		weights := index.Weights{
			Vector:   searchFlags.vectorWeight,
			Phonetic: searchFlags.phoneticWeight,
			RRF:      searchFlags.rrfK,
		}
		if err := weights.Validate(); err != nil {
			return fmt.Errorf("--vector-weight/--phonetic-weight/--rrf-k: %w", err)
		}
		if searchFlags.offset < 0 {
			return errors.New("--offset must not be negative")
		}
		facetFields, err := parseFacetFields(searchFlags.facets)
		if err != nil {
			return err
		}

		engine, _, alog, teardown, err := buildEngine(true, false)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		// Search is read-only — skip the save on exit.
		_ = teardown

		if query == "" {
			if filter != nil {
				return errors.New("--where/--filter need a query: corpus-wide facets count every document")
			}
			printHeader("facets", strings.Join(facetFields, ", "))
			facets, err := engine.CorpusFacetCounts(facetFields, searchFlags.facetTopK)
			if err != nil {
				return fmt.Errorf("facets: %w", err)
			}
			printFacets(facetFields, facets)
			return nil
		}

		printHeader("search", fmt.Sprintf("%q", query))

		ctx := context.Background()
		start := time.Now()
		results, err := engine.SearchFilteredWeighted(ctx, query, filter, weights)
		if err != nil {
			return fmt.Errorf("search: %w", err)
		}
		alog.Log("SEARCH", fmt.Sprintf("%q → %d results", query, len(results)))

		if searchFlags.rerank {
			rr, err := reranker.New(searchFlags.rerankModel, modelsDir())
			if err != nil {
				return fmt.Errorf("rerank: %w", err)
			}
			defer rr.Close()
			results = rr.Rerank(ctx, query, results, engine.GetText)
		}

		// Facets describe every match, so they are counted before --sort,
		// --offset and --max narrow the list down to the page shown.
		var facets index.Facets
		if len(facetFields) > 0 {
			ids := make([]string, len(results))
			for i, r := range results {
				ids[i] = r.ID
			}
			if facets, err = engine.FacetCounts(ids, facetFields, searchFlags.facetTopK); err != nil {
				return fmt.Errorf("facets: %w", err)
			}
		}

		// --sort replaces score ordering over the full candidate list, before
		// the page is sliced out of it.
		engine.SortByAttribute(results, searchFlags.sortField, searchFlags.sortDesc)

		var explained map[string]index.ExplainHit
		if searchFlags.explain {
			_, hits, err := engine.ExplainFiltered(ctx, query, filter)
			if err != nil {
				return fmt.Errorf("explain: %w", err)
			}
			explained = make(map[string]index.ExplainHit, len(hits))
			for _, h := range hits {
				explained[h.ID] = h
			}
		}
		elapsed := time.Since(start)

		total := len(results)
		page := pageOf(results, searchFlags.offset, searchFlags.maxResults)
		if len(page) == 0 {
			if total == 0 {
				fmt.Println(dim("  no results"))
			} else {
				fmt.Println(dim(fmt.Sprintf("  no results past offset %d (%d total)", searchFlags.offset, total)))
			}
			fmt.Println()
			printFacets(facetFields, facets)
			return nil
		}

		topScore := 0.0
		for _, r := range results {
			if r.Score > topScore {
				topScore = r.Score
			}
		}
		printDivider()
		for i, r := range page {
			printResult(searchFlags.offset+i+1, r.ID, r.Score, topScore)
			if explained != nil {
				printExplain(explained[r.ID])
			}
		}
		printDivider()
		printFacets(facetFields, facets)

		printFooter(
			fmt.Sprintf("%d of %d result%s", len(page), total, plural(int64(total))),
			elapsed.Round(time.Millisecond).String(),
		)
		return nil
	},
}

// pageOf returns results[offset : offset+max] clamped to the slice; max <= 0
// means "everything after offset".
func pageOf(results []index.SearchResponse, offset, max int) []index.SearchResponse {
	if offset >= len(results) {
		return nil
	}
	results = results[offset:]
	if max > 0 && max < len(results) {
		results = results[:max]
	}
	return results
}

// parseFacetFields flattens repeated/comma-separated --facets values.
func parseFacetFields(flags []string) ([]string, error) {
	var out []string
	seen := map[string]bool{}
	for _, f := range flags {
		for _, name := range strings.Split(f, ",") {
			name = strings.TrimSpace(name)
			if name == "" {
				return nil, fmt.Errorf("--facets %q: empty field name", f)
			}
			if !seen[name] {
				seen[name] = true
				out = append(out, name)
			}
		}
	}
	return out, nil
}

// printExplain prints the raw evidence behind one result (from --explain).
func printExplain(h index.ExplainHit) {
	var terms []string
	for _, t := range h.Terms {
		switch {
		case t.Synonym:
			terms = append(terms, fmt.Sprintf("%s→%s (synonym)", t.Term, t.Matched))
		case t.Dist > 0:
			terms = append(terms, fmt.Sprintf("%s→%s (edit %d)", t.Term, t.Matched, t.Dist))
		default:
			terms = append(terms, t.Term)
		}
	}
	if len(terms) == 0 {
		terms = []string{"none"}
	}
	fmt.Printf("      %s\n", muted(fmt.Sprintf("bm25 %.3f · cosine %.3f · terms: %s",
		h.Lexical, h.Semantic, strings.Join(terms, ", "))))
}

// printFacets prints each requested field's value counts.
func printFacets(fields []string, facets index.Facets) {
	if len(fields) == 0 {
		return
	}
	for _, f := range fields {
		fmt.Printf("  %s\n", bold(f))
		list := facets[f]
		if len(list) == 0 {
			fmt.Printf("    %s\n", dim("(no values)"))
			continue
		}
		for _, c := range list {
			fmt.Printf("    %-32s %s\n", formatAttrValue(c.Value), muted(strconv.Itoa(c.Count)))
		}
	}
	fmt.Println()
}

// formatAttrValue renders a scalar attribute value for display.
func formatAttrValue(v index.AttrValue) string {
	switch v.Kind {
	case index.AttrString:
		return v.S
	case index.AttrBool:
		return strconv.FormatBool(v.N != 0)
	case index.AttrNumber:
		return strconv.FormatFloat(v.N, 'g', -1, 64)
	}
	return fmt.Sprint(v.Any())
}

func init() {
	addEngineFlags(searchCmd)
	searchCmd.Flags().IntVarP(&searchFlags.maxResults, "max", "n", 10, "Maximum results to display")
	searchCmd.Flags().IntVar(&searchFlags.offset, "offset", 0, "Skip this many results from the top (pagination; applied after --sort)")
	searchCmd.Flags().StringArrayVar(&searchFlags.where, "where", nil,
		"Only documents whose attribute matches: key=value, key!=value, key>=n, key<=n (repeatable; all must hold)")
	searchCmd.Flags().StringVar(&searchFlags.filter, "filter", "",
		`Only documents matching a JSON filter, e.g. '{"op":"or","args":[{"op":"eq","field":"lang","value":"en"},{"op":"exists","field":"pinned"}]}'`)
	searchCmd.Flags().BoolVar(&searchFlags.rerank, "rerank", false,
		"Rerank the top candidates with a cross-encoder (needs: zenith models pull ms-marco-MiniLM-L-6-v2)")
	searchCmd.Flags().StringVar(&searchFlags.rerankModel, "rerank-model", "",
		"Reranker model id from `zenith models list --rerankers` (default: ms-marco-MiniLM-L-6-v2)")
	searchCmd.Flags().StringVar(&searchFlags.sortField, "sort", "",
		"Order results by this attribute instead of score (documents missing it sort last)")
	searchCmd.Flags().BoolVar(&searchFlags.sortDesc, "desc", false, "With --sort: descending order")
	searchCmd.Flags().BoolVar(&searchFlags.explain, "explain", false,
		"Show the raw evidence behind each result: BM25, cosine similarity and matched terms")
	searchCmd.Flags().Float64Var(&searchFlags.vectorWeight, "vector-weight", 0,
		"Override the RRF weight of the semantic list for this query (0 = engine default)")
	searchCmd.Flags().Float64Var(&searchFlags.phoneticWeight, "phonetic-weight", 0,
		"Override the per-match phonetic weight for this query (0 = engine default)")
	searchCmd.Flags().Float64Var(&searchFlags.rrfK, "rrf-k", 0,
		"Override the RRF k constant for this query (0 = engine default)")
	searchCmd.Flags().StringSliceVar(&searchFlags.facets, "facets", nil,
		"Count attribute values over all matches, e.g. --facets lang,tags (with no query: over every document)")
	searchCmd.Flags().IntVar(&searchFlags.facetTopK, "facet-top", 10, "With --facets: values shown per field (0 = all)")
}
