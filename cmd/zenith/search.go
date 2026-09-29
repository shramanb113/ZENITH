package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"
)

var searchFlags struct {
	maxResults int
	where      []string
	filter     string
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
  4. RRF fusion`,

	Args: cobra.MinimumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		query := strings.Join(args, " ")

		filter, err := buildFilter(searchFlags.where, searchFlags.filter)
		if err != nil {
			return err
		}

		printHeader("search", fmt.Sprintf("%q", query))

		engine, alog, teardown, err := buildEngine(true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		// Search is read-only — skip the save on exit.
		_ = teardown

		start := time.Now()
		results, err := engine.SearchFiltered(context.Background(), query, filter)
		if err != nil {
			return fmt.Errorf("search: %w", err)
		}
		alog.Log("SEARCH", fmt.Sprintf("%q → %d results", query, len(results)))
		elapsed := time.Since(start)

		if len(results) == 0 {
			fmt.Println(dim("  no results"))
			fmt.Println()
			return nil
		}

		max := searchFlags.maxResults
		if max <= 0 || max > len(results) {
			max = len(results)
		}

		topScore := results[0].Score
		printDivider()
		for i, r := range results[:max] {
			printResult(i+1, r.ID, r.Score, topScore)
		}
		printDivider()

		printFooter(
			fmt.Sprintf("%d result%s", max, plural(int64(max))),
			elapsed.Round(time.Millisecond).String(),
		)
		return nil
	},
}

func init() {
	addEngineFlags(searchCmd)
	searchCmd.Flags().IntVarP(&searchFlags.maxResults, "max", "n", 10, "Maximum results to display")
	searchCmd.Flags().StringArrayVar(&searchFlags.where, "where", nil,
		"Only documents whose attribute matches: key=value, key!=value, key>=n, key<=n (repeatable; all must hold)")
	searchCmd.Flags().StringVar(&searchFlags.filter, "filter", "",
		`Only documents matching a JSON filter, e.g. '{"op":"or","args":[{"op":"eq","field":"lang","value":"en"},{"op":"exists","field":"pinned"}]}'`)
}
