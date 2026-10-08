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

// toAttrs converts the loosely-typed JSON attrs map into index.Attrs.
// index.AttrValue is {Kind AttrKind, S string, N float64, Arr []AttrValue}
// — bools are stored as Kind: AttrBool, N: 1/0, not a separate bool field.
// A JSON array value is silently dropped: this input format carries scalar
// attrs only, matching cmd/zenith/index.go's --attr k=v flag.
func toAttrs(m map[string]any) index.Attrs {
	if len(m) == 0 {
		return nil
	}
	out := make(index.Attrs, len(m))
	for k, v := range m {
		switch tv := v.(type) {
		case string:
			out[k] = index.AttrValue{Kind: index.AttrString, S: tv}
		case bool:
			n := 0.0
			if tv {
				n = 1
			}
			out[k] = index.AttrValue{Kind: index.AttrBool, N: n}
		case float64:
			out[k] = index.AttrValue{Kind: index.AttrNumber, N: tv}
		}
	}
	return out
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
		docs = append(docs, index.BatchDoc{ID: jd.ID, Text: jd.Text, Attrs: toAttrs(jd.Attrs)})
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("%s: scan: %w", txnAddFlags.file, err)
	}
	if len(docs) == 0 {
		return fmt.Errorf("%s: no documents found", txnAddFlags.file)
	}

	engine, storageEng, _, teardown, err := buildEngine(true)
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
	ids := strings.Split(txnRemoveFlags.ids, ",")
	for i, id := range ids {
		ids[i] = strings.TrimSpace(id)
	}

	engine, storageEng, _, teardown, err := buildEngine(true)
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
