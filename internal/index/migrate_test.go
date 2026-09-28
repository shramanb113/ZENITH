package index

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// writeLegacy writes e's state in the retired gob format, byte-compatible with
// what the pre-segment Save produced (state order = readLegacy's decode order).
func writeLegacy(t *testing.T, e *Engine, path string, version uint16) {
	t.Helper()
	e.mu.RLock()
	defer e.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	w.Write(saveFormatMagic[:])
	var v [2]byte
	binary.BigEndian.PutUint16(v[:], version)
	w.Write(v[:])
	if version >= 5 {
		name, dims := e.embedderIdentity()
		if err := writeHeaderString(w, name); err != nil {
			t.Fatal(err)
		}
		var d [4]byte
		binary.BigEndian.PutUint32(d[:], uint32(dims))
		w.Write(d[:])
	}

	lengths, termFreqs, docFreq, totalDocs, totalLen := e.bm25.State()
	// Legacy postings map fragment -> IDs; the delta holds exactly that.
	state := []any{
		e.inverted.GetData(), e.idMapping, e.vectors.GetVectors(),
		map[string]int{}, e.phonetics.GetData(), map[int][]string{},
		e.inverted.GetGlobalSeen(), e.vectors.GetWordVectors(), map[uint64][]string{},
		lengths, termFreqs, docFreq, totalDocs, totalLen,
		map[uint64]int{}, map[uint64]map[string]int{}, map[string]int{}, 0,
		e.inverted.GetDocTokens(), e.docText,
	}
	if version >= 5 {
		state = append(state, e.attrs)
	}
	enc := gob.NewEncoder(w)
	for _, s := range state {
		if err := enc.Encode(s); err != nil {
			t.Fatalf("encode legacy state: %v", err)
		}
	}
	if err := w.Flush(); err != nil {
		t.Fatal(err)
	}
}

func migrationCorpus(t *testing.T, e *Engine, withAttrs bool) {
	t.Helper()
	ctx := context.Background()
	r := rand.New(rand.NewSource(9))
	for i := 0; i < 120; i++ {
		attrs := Attrs{"tenant": {Kind: AttrString, S: string(rune('a' + r.Intn(3)))}}
		if !withAttrs {
			attrs = nil // format v4 predates attributes
		}
		text := diffText(r)
		if err := e.AddWithVectorAttrs(ctx, fmt.Sprintf("m-%d", i), text, e.EmbedText(ctx, text), attrs); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMigrate_V5_KeepsEverythingAndVerifies(t *testing.T) {
	ref := diffEngine()
	migrationCorpus(t, ref, true)

	path := filepath.Join(t.TempDir(), "old.db")
	writeLegacy(t, ref, path, 5)

	// A current engine refuses it with a typed error the callers can act on.
	e := diffEngine()
	err := e.Load(path)
	var lfe *LegacyFormatError
	if !errors.Is(err, ErrIncompatibleVersion) || !errors.As(err, &lfe) {
		t.Fatalf("Load of a legacy file: err = %v, want a *LegacyFormatError matching ErrIncompatibleVersion", err)
	}
	e.Close()

	res, err := Migrate(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if res.FromVersion != 5 || res.Docs != 120 {
		t.Fatalf("result = %+v", res)
	}
	if _, err := os.Stat(res.Backup); err != nil {
		t.Fatalf("backup missing: %v", err)
	}
	if err := Verify(path); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "migrated v5", ref, mustLoad(t, path))

	if _, err := Migrate(path, ""); !errors.Is(err, ErrAlreadyCurrent) {
		t.Fatalf("second Migrate: err = %v, want ErrAlreadyCurrent", err)
	}
}

func TestMigrate_V4_HasNoEmbedderIdentityUnlessGiven(t *testing.T) {
	ref := diffEngine()
	migrationCorpus(t, ref, false)
	path := filepath.Join(t.TempDir(), "v4.db")
	writeLegacy(t, ref, path, 4)

	res, err := Migrate(path, "")
	if err != nil {
		t.Fatal(err)
	}
	if res.FromVersion != 4 || res.Embedder != "unknown" || res.Dims != 64 {
		t.Fatalf("result = %+v", res)
	}
	assertSame(t, "migrated v4", ref, mustLoad(t, path))
}

// A migration that fails must leave the original file byte-for-byte intact.
func TestMigrate_CorruptSourceIsUntouched(t *testing.T) {
	ref := diffEngine()
	migrationCorpus(t, ref, true)
	path := filepath.Join(t.TempDir(), "bad.db")
	writeLegacy(t, ref, path, 5)
	b, _ := os.ReadFile(path)
	trunc := b[:len(b)/2]
	if err := os.WriteFile(path, trunc, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Migrate(path, ""); err == nil {
		t.Fatal("Migrate accepted a truncated file")
	}
	after, _ := os.ReadFile(path)
	if string(after) != string(trunc) {
		t.Fatal("failed migration modified the source file")
	}
}

// A fresh engine saving to a path that holds a readable old-format index must
// refuse rather than replace it with different contents (the CLI once fell
// through "load failed → start fresh → Save" and would have done exactly that).
func TestSave_RefusesToOverwriteLegacyFile(t *testing.T) {
	ref := diffEngine()
	migrationCorpus(t, ref, true)
	path := filepath.Join(t.TempDir(), "old.db")
	writeLegacy(t, ref, path, 5)
	before, _ := os.ReadFile(path)

	e := diffEngine()
	t.Cleanup(func() { e.Close() })
	if err := e.Add(context.Background(), "x", "unrelated document"); err != nil {
		t.Fatal(err)
	}
	err := e.Save(path)
	if !errors.Is(err, ErrIncompatibleVersion) {
		t.Fatalf("Save over a legacy file: err = %v, want ErrIncompatibleVersion", err)
	}
	after, _ := os.ReadFile(path)
	if string(before) != string(after) {
		t.Fatal("the legacy file was modified")
	}
	if e.DBPath() != "" {
		t.Fatalf("engine bound itself to %q despite the refused Save", e.DBPath())
	}
}
