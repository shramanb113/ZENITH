package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/spf13/cobra"
)

var txnAddFlags struct {
	file string
}

var txnRemoveFlags struct {
	ids string
}

// jsonlDoc is one line of the --file input for `zenith txn add`.
type jsonlDoc struct {
	ID    string         `json:"id"`
	Text  string         `json:"text"`
	Attrs map[string]any `json:"attrs,omitempty"`
}

var txnCmd = &cobra.Command{
	Use:   "txn",
	Short: "Atomic multi-document operations (AddTransaction/RemoveBatch)",
}

var txnAddCmd = &cobra.Command{
	Use:   "add",
	Short: "Atomically add every document in a JSONL file (all-or-nothing)",
	RunE:  runTxnAdd,
}

var txnRemoveCmd = &cobra.Command{
	Use:   "remove",
	Short: "Atomically remove a comma-separated list of document IDs (all-or-nothing)",
	RunE:  runTxnRemove,
}

func init() {
	addEngineFlags(txnAddCmd)
	addEngineFlags(txnRemoveCmd)
	txnAddCmd.Flags().StringVar(&txnAddFlags.file, "file", "", `JSONL file, one {"id":...,"text":...,"attrs":{...}} per line (required)`)
	txnRemoveCmd.Flags().StringVar(&txnRemoveFlags.ids, "ids", "", "Comma-separated document IDs to remove (required)")
	txnCmd.AddCommand(txnAddCmd, txnRemoveCmd)
}

// toAttrs converts the loosely-typed JSON attrs map into index.Attrs via
// index.AttrValueFromAny — the same conversion pkg/zenith's AddWithAttrs
// uses, so the CLI and the library accept exactly the same values: a JSON
// string/bool/number, or an array of those (an AttrArray). null, a nested
// object, or a nested array is an error, not a silent drop: losing part of a
// document's metadata without telling the caller is worse than refusing the
// whole file up front.
func toAttrs(m map[string]any) (index.Attrs, error) {
	if len(m) == 0 {
		return nil, nil
	}
	out := make(index.Attrs, len(m))
	for k, v := range m {
		if k == "" {
			return nil, fmt.Errorf("attrs: keys must not be empty")
		}
		av, err := index.AttrValueFromAny(v)
		if err != nil {
			return nil, fmt.Errorf("attrs[%q]: %w", k, err)
		}
		out[k] = av
	}
	return out, nil
}

func runTxnAdd(cmd *cobra.Command, args []string) error {
	if txnAddFlags.file == "" {
		return fmt.Errorf("--file is required")
	}
	f, err := os.Open(txnAddFlags.file)
	if err != nil {
		return fmt.Errorf("open %s: %w", txnAddFlags.file, err)
	}
	defer f.Close()

	var docs []index.BatchDoc
	scanner := bufio.NewScanner(f)
	// bufio.Scanner's default max token size is 64KiB, which a realistic
	// document (not just its JSON wrapper) can easily exceed — the default
	// would abort the whole file with an unhelpful "token too long" and no
	// line number. 32MiB matches this project's own largest configured
	// single-item limit elsewhere (nothing smaller is documented as a hard
	// ceiling for one document's text).
	const maxLineSize = 32 * 1024 * 1024
	scanner.Buffer(make([]byte, 0, 64*1024), maxLineSize)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var jd jsonlDoc
		if err := json.Unmarshal([]byte(line), &jd); err != nil {
			return fmt.Errorf("%s: line %d: invalid JSON: %w", txnAddFlags.file, lineNo, err)
		}
		if jd.ID == "" {
			return fmt.Errorf("%s: line %d: missing \"id\"", txnAddFlags.file, lineNo)
		}
		attrs, err := toAttrs(jd.Attrs)
		if err != nil {
			return fmt.Errorf("%s: line %d: %w", txnAddFlags.file, lineNo, err)
		}
		docs = append(docs, index.BatchDoc{ID: jd.ID, Text: jd.Text, Attrs: attrs})
	}
	if err := scanner.Err(); err != nil {
		// scanner.Scan() already returned false for the line that failed
		// without incrementing lineNo, so lineNo+1 is that line.
		return fmt.Errorf("%s: line %d: scan: %w", txnAddFlags.file, lineNo+1, err)
	}
	if len(docs) == 0 {
		return fmt.Errorf("%s: no documents found", txnAddFlags.file)
	}

	engine, storageEng, _, teardown, err := buildEngine(true, true)
	if err != nil {
		return err
	}
	defer teardown()

	if err := engine.AddTransaction(context.Background(), docs, storageEng.NewTxn()); err != nil {
		return fmt.Errorf("AddTransaction: %w", err)
	}
	fmt.Printf("Added %d documents atomically.\n", len(docs))
	return nil
}

func runTxnRemove(cmd *cobra.Command, args []string) error {
	if txnRemoveFlags.ids == "" {
		return fmt.Errorf("--ids is required")
	}
	var ids []string
	for _, id := range strings.Split(txnRemoveFlags.ids, ",") {
		id = strings.TrimSpace(id)
		if id == "" {
			// A trailing/doubled comma ("a,", "a,,b") must not turn into a
			// RemoveBatch call for an empty ID — there is no such document,
			// and the printed count would otherwise include IDs that never
			// existed.
			continue
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return fmt.Errorf("--ids contained no non-empty IDs")
	}

	engine, storageEng, _, teardown, err := buildEngine(true, true)
	if err != nil {
		return err
	}
	defer teardown()

	if err := engine.RemoveBatch(context.Background(), ids, storageEng.NewTxn()); err != nil {
		return fmt.Errorf("RemoveBatch: %w", err)
	}
	fmt.Printf("Removed %d documents atomically.\n", len(ids))
	return nil
}
