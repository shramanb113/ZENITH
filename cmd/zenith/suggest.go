package main

import (
	"fmt"

	"github.com/spf13/cobra"
)

var suggestFlags struct {
	max int
}

var suggestCmd = &cobra.Command{
	Use:   "suggest <prefix>",
	Short: "Autocomplete: list indexed terms starting with a prefix",
	Long: `Lists terms from the local index's vocabulary that start with <prefix>.

The terms are the analysed vocabulary — lowercased, stop words removed and
Porter2-stemmed — so "running" is listed as "run", and a prefix that runs past
a stem ("runn") finds nothing. They come back in lexicographic order, not
ranked by how often they occur.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		engine, _, _, teardown, err := buildEngine(true, false)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		// Read-only — skip the save on exit.
		_ = teardown

		terms, err := engine.Suggest(args[0], suggestFlags.max)
		if err != nil {
			return fmt.Errorf("suggest: %w", err)
		}
		if len(terms) == 0 {
			fmt.Println(dim("  no matching terms"))
			return nil
		}
		for _, t := range terms {
			fmt.Println(t)
		}
		return nil
	},
}

func init() {
	addEngineFlags(suggestCmd)
	suggestCmd.Flags().IntVarP(&suggestFlags.max, "max", "n", 10, "Maximum terms to list (capped at 1000)")
}
