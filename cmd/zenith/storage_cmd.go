package main

import (
	"fmt"

	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/storage"
	"github.com/spf13/cobra"
)

var storageCmd = &cobra.Command{
	Use:   "storage",
	Short: "Inspect or maintain the Pebble-backed document journal directly",
}

var storageInspectCmd = &cobra.Command{
	Use:   "inspect <id>",
	Short: "Print the journaled entry for a document ID, if one exists",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		found, err := runStorageInspect(cmd, args)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf("%s: not journaled (it may still be live only in a saved segment — this command checks only the journal)", args[0])
		}
		return nil
	},
}

var storagePruneCmd = &cobra.Command{
	Use:   "prune",
	Short: "Snapshot, save the index, and prune the now-redundant journal entries",
	RunE:  runStoragePrune,
}

func init() {
	addEngineFlags(storageInspectCmd)
	addEngineFlags(storagePruneCmd)
	storageCmd.AddCommand(storageInspectCmd, storagePruneCmd)
}

// runStorageInspect opens its own short-lived, standalone storage.Engine —
// unlike txn add/remove or prune, this is read-only and never runs alongside
// another open in the same process in real usage (each CLI invocation is
// its own process), so there is no lock-conflict risk here the way there was
// for txn add/remove (see buildEngine's doc comment).
func runStorageInspect(cmd *cobra.Command, args []string) (bool, error) {
	id := args[0]
	storageEng, err := storage.Open(storage.EngineConfig{Dir: effectiveStorageDir()})
	if err != nil {
		return false, fmt.Errorf("open storage engine: %w", err)
	}
	defer storageEng.Close()

	raw, ok := storageEng.Get([]byte(id))
	if !ok {
		return false, nil
	}
	text, vector, attrs := index.DecodeJournalValue(raw)
	fmt.Printf("id:      %s\n", id)
	fmt.Printf("text:    %s\n", text)
	fmt.Printf("attrs:   %d field(s)\n", len(attrs))
	if vector != nil {
		fmt.Printf("vector:  present, %d dims\n", len(vector))
	} else {
		fmt.Printf("vector:  not journaled (legacy entry or embedding was unavailable)\n")
	}
	return true, nil
}

// runStoragePrune is exactly buildEngine's own Save+Prune teardown sequence,
// run on demand rather than only at process exit — useful for bounding
// journal growth mid-session on a long-running `zenith watch`.
func runStoragePrune(cmd *cobra.Command, args []string) error {
	_, _, _, teardown, err := buildEngine(true, true)
	if err != nil {
		return err
	}
	teardown()
	fmt.Println("Pruned.")
	return nil
}
