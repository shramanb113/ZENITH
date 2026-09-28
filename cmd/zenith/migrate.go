package main

import (
	"errors"
	"fmt"

	"github.com/spf13/cobra"

	"github.com/shramanb113/ZENITH/internal/index"
)

var migrateFlags struct {
	db       string
	embedder string
}

func init() {
	migrateCmd.Flags().StringVar(&migrateFlags.db, "db", zenithDataPath("zenith.db"), "Index database file to convert")
	migrateCmd.Flags().StringVar(&migrateFlags.embedder, "embedder-name", "",
		"Embedding model that produced a format-4 index's vectors (format 5+ files record it themselves)")
	compactCmd.Flags().StringVar(&migrateFlags.db, "db", zenithDataPath("zenith.db"), "Index database file to compact")
}

var migrateCmd = &cobra.Command{
	Use:   "migrate",
	Short: "Convert an index from an older release to the current on-disk format",
	Long: `Converts an index file written by an older zenith (gob format, versions 4 and 5)
to the current memory-mapped segment format, in place.

  - No embedding model is needed: vectors are copied, not recomputed.
  - The original is copied to <db>.v<N>.bak first and kept.
  - The new index is written beside the old file and swapped in atomically, then
    re-opened and checksummed. A failure or crash at any point leaves your
    original index usable.

Stop any running 'zenith watch' / 'zenith serve' first.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		printHeader("migrate", migrateFlags.db)
		res, err := index.Migrate(migrateFlags.db, migrateFlags.embedder)
		if errors.Is(err, index.ErrAlreadyCurrent) {
			printFooter("nothing to do", "already in the current format")
			return nil
		}
		if err != nil {
			return err
		}
		fmt.Printf("  %s  converted format v%d → v6: %d documents (embedder %s)\n", green("✓"), res.FromVersion, res.Docs, res.Embedder)
		fmt.Printf("  %s  backup of the original: %s\n", muted("·"), res.Backup)
		printDivider()
		printFooter("migrated", "verified by reopening and checksumming")
		return nil
	},
}

var compactCmd = &cobra.Command{
	Use:   "compact",
	Short: "Merge index segments and reclaim space from deleted documents",
	Long: `Merges every segment of the index into one, dropping deleted documents and
their vocabulary. Reads touch fewer files afterwards and the disk shrinks.

zenith compacts automatically in the background once enough segments build up;
this command is for doing it on demand while nothing else is using the index.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		printHeader("compact", migrateFlags.db)
		before, after, err := index.CompactFile(migrateFlags.db)
		if err != nil {
			return err
		}
		if before == after {
			printFooter("nothing to do", fmt.Sprintf("%d segment(s), none deleted", after))
			return nil
		}
		printFooter("compacted", fmt.Sprintf("%d segments → %d", before, after))
		return nil
	},
}
