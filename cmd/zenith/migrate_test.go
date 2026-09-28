package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/analysis"
	"github.com/shramanb113/ZENITH/internal/config"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/internal/ranking"
)

// goldenV5 is a real format-5 index written by the previous release's library
// (see internal/index/testdata/README.md).
const goldenV5 = "../../internal/index/testdata/golden-v5/golden.db"

func copyFile(t *testing.T, src, dst string) {
	t.Helper()
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func hasCheck(cs []doctorCheck, name string) (doctorCheck, bool) {
	for _, c := range cs {
		if c.name == name {
			return c, true
		}
	}
	return doctorCheck{}, false
}

// End to end at the CLI layer, on a file the previous release wrote: doctor
// flags it and names the fix, `migrate` converts it (keeping a byte-identical
// backup), doctor then passes and verifies checksums, and `compact` is a no-op.
func TestCLI_MigrateCompactDoctor_OnPreviousReleaseIndex(t *testing.T) {
	dir := t.TempDir()
	db := filepath.Join(dir, "zenith.db")
	copyFile(t, goldenV5, db)
	orig, _ := os.ReadFile(db)

	// doctor on the old format: fails, and points at `zenith migrate`.
	chk, ok := hasCheck(indexChecks(db), "index")
	if !ok || chk.ok || !strings.Contains(chk.fix, "zenith migrate") {
		t.Fatalf("doctor on a v5 file: %+v", chk)
	}

	migrateFlags.db, migrateFlags.embedder = db, ""
	if err := migrateCmd.RunE(migrateCmd, nil); err != nil {
		t.Fatalf("migrate: %v", err)
	}
	bak, err := os.ReadFile(db + ".v5.bak")
	if err != nil || !bytes.Equal(bak, orig) {
		t.Fatalf("backup missing or not byte-identical to the original (err=%v)", err)
	}
	info, err := index.Inspect(db)
	if err != nil || info.Version != index.CurrentFormat || info.Embedder != "test:bag32" || len(info.Segments) != 1 {
		t.Fatalf("after migrate: %+v, %v", info, err)
	}

	cs := indexChecks(db)
	for _, name := range []string{"index", "checksums"} {
		if c, ok := hasCheck(cs, name); !ok || !c.ok {
			t.Fatalf("doctor after migrate: %s check = %+v", name, c)
		}
	}

	// Migrating again is a harmless no-op, not an error and not another backup.
	if err := migrateCmd.RunE(migrateCmd, nil); err != nil {
		t.Fatalf("second migrate: %v", err)
	}
	if _, err := os.Stat(db + ".v5.bak.1"); err == nil {
		t.Fatal("second migrate made another backup")
	}

	compactFlagsDB := db
	migrateFlags.db = compactFlagsDB
	if err := compactCmd.RunE(compactCmd, nil); err != nil {
		t.Fatalf("compact: %v", err)
	}
	if info2, _ := index.Inspect(db); len(info2.Segments) != 1 {
		t.Fatalf("compact of a single clean segment changed the layout: %+v", info2)
	}
}

// doctor must report damage, not search through it.
func TestCLI_Doctor_DetectsCorruptSegment(t *testing.T) {
	dir := t.TempDir()
	db := filepath.Join(dir, "zenith.db")
	copyFile(t, goldenV5, db)
	migrateFlags.db, migrateFlags.embedder = db, ""
	if err := migrateCmd.RunE(migrateCmd, nil); err != nil {
		t.Fatal(err)
	}
	info, err := index.Inspect(db)
	if err != nil {
		t.Fatal(err)
	}
	seg := info.Segments[0]
	b, _ := os.ReadFile(seg)
	b[len(b)/2] ^= 0xFF // flip bits in the middle of the file
	if err := os.WriteFile(seg, b, 0o644); err != nil {
		t.Fatal(err)
	}
	c, ok := hasCheck(indexChecks(db), "checksums")
	if !ok || c.ok {
		t.Fatalf("doctor did not flag a corrupted segment: %+v", c)
	}
}

func TestCLI_ModelsList_MentionsEveryRegisteredModel(t *testing.T) {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w
	err := modelsListCmd.RunE(modelsListCmd, nil)
	w.Close()
	os.Stdout = old
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	out.ReadFrom(r)
	for _, id := range []string{"all-MiniLM-L6-v2", "gte-small", "bge-small-en-v1.5"} {
		if !strings.Contains(out.String(), id) {
			t.Errorf("`zenith models list` output lacks %s:\n%s", id, out.String())
		}
	}
}

// namedStub is an embedder that only carries an identity, to write an index that
// claims to have been built with a registry model.
type namedStub struct{ name string }

func (s namedStub) Name() string    { return s.name }
func (s namedStub) Dimensions() int { return 4 }
func (s namedStub) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0, 0, 0}, nil
}
func (s namedStub) EmbedBatch(_ context.Context, ts []string) ([][]float32, error) {
	out := make([][]float32, len(ts))
	for i := range out {
		out[i] = []float32{1, 0, 0, 0}
	}
	return out, nil
}

// When an index was built with another model, the error must say how to go on.
func TestMismatchHint_NamesTheModelThatBuiltTheIndex(t *testing.T) {
	db := filepath.Join(t.TempDir(), "zenith.db")
	e := index.NewEngine(config.DefaultConfig(), namedStub{"onnx:gte-small"}, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
	if err := e.Add(context.Background(), "a", "hello world"); err != nil {
		t.Fatal(err)
	}
	if err := e.Save(db); err != nil {
		t.Fatal(err)
	}
	e.Close()

	cliFlags.dbPath = db
	hint := mismatchHint(fmt.Errorf("wrapped: %w", index.ErrEmbedderMismatch))
	for _, want := range []string{"gte-small", "zenith models pull gte-small", "--model gte-small", "re-index"} {
		if !strings.Contains(hint, want) {
			t.Errorf("hint lacks %q:\n%s", want, hint)
		}
	}
	if got := mismatchHint(errors.New("something else")); got != "" {
		t.Errorf("hint for an unrelated error should be empty, got %q", got)
	}
}

func TestModelFromIndex_FollowsTheRecordedModel(t *testing.T) {
	write := func(name string) string {
		db := filepath.Join(t.TempDir(), "zenith.db")
		e := index.NewEngine(config.DefaultConfig(), namedStub{name}, ranking.NewRRFRanker(0, 0), analysis.NewStandardAnalyzer())
		if err := e.Add(context.Background(), "a", "hello world"); err != nil {
			t.Fatal(err)
		}
		if err := e.Save(db); err != nil {
			t.Fatal(err)
		}
		e.Close()
		return db
	}
	other := "gte-small"
	if other == localembedder.BundledID() {
		other = "bge-small-en-v1.5"
	}
	if got := modelFromIndex(write("onnx:" + other)); got != other {
		t.Errorf("index built with %s: modelFromIndex = %q, want it to follow the index", other, got)
	}
	if got := modelFromIndex(write("onnx:" + localembedder.BundledID())); got != "" {
		t.Errorf("index built with the bundled model needs no override, got %q", got)
	}
	if got := modelFromIndex(write("none:deterministic-fallback")); got != "" {
		t.Errorf("non-ONNX identity must not select a model, got %q", got)
	}
	if got := modelFromIndex(write("onnx:not-a-registered-model")); got != "" {
		t.Errorf("unregistered model id must be ignored, got %q", got)
	}
	if got := modelFromIndex(filepath.Join(t.TempDir(), "missing.db")); got != "" {
		t.Errorf("missing index: got %q", got)
	}
}
