package wal

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func appendKeys(t *testing.T, w *WAL, keys ...string) {
	t.Helper()
	for _, k := range keys {
		if _, err := w.Append(context.Background(), &Record{Op: OpTypePut, Key: []byte(k), Value: []byte("v-" + k)}); err != nil {
			t.Fatal(err)
		}
	}
}

func keysOf(recs []Record) []string {
	out := make([]string, len(recs))
	for i, r := range recs {
		out[i] = string(r.Key)
	}
	return out
}

func TestRotate_SplitsTheLogAtTheCutPoint(t *testing.T) {
	w, path := openTestWAL(t)
	appendKeys(t, w, "a", "b", "c")
	archive := path + ".old-000001"
	if err := w.Rotate(archive); err != nil {
		t.Fatal(err)
	}
	if w.Size() != 0 {
		t.Fatalf("live log size after rotate = %d, want 0", w.Size())
	}
	appendKeys(t, w, "d", "e")

	old, err := ReadFile(archive)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(keysOf(old)); got != "[a b c]" {
		t.Fatalf("archive holds %s, want [a b c]", got)
	}
	// The live file is what OpenWAL recovers: only the records after the cut.
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	w2, recs, err := OpenWAL(path, WALConfig{SyncMode: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if got := fmt.Sprint(keysOf(recs)); got != "[d e]" {
		t.Fatalf("live log recovered %s, want [d e]", got)
	}
}

func TestRotate_RefusesToOverwriteAnArchive(t *testing.T) {
	w, path := openTestWAL(t)
	appendKeys(t, w, "a")
	archive := path + ".old-000001"
	if err := os.WriteFile(archive, []byte("precious"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := w.Rotate(archive); err == nil {
		t.Fatal("Rotate overwrote an existing archive")
	}
	if b, _ := os.ReadFile(archive); string(b) != "precious" {
		t.Fatal("existing archive was modified")
	}
	// The log must keep working after the refusal.
	appendKeys(t, w, "b")
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	w2, recs, err := OpenWAL(path, WALConfig{SyncMode: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if got := fmt.Sprint(keysOf(recs)); got != "[a b]" {
		t.Fatalf("log recovered %s, want [a b]", got)
	}
}

func TestRotate_RepeatedRotationsKeepEveryRecordInOrder(t *testing.T) {
	w, path := openTestWAL(t)
	var all []string
	for i := 0; i < 4; i++ {
		k := fmt.Sprintf("k%d", i)
		appendKeys(t, w, k)
		all = append(all, k)
		if err := w.Rotate(fmt.Sprintf("%s.old-%06d", path, i+1)); err != nil {
			t.Fatal(err)
		}
	}
	appendKeys(t, w, "live")
	all = append(all, "live")
	w.Close()

	var got []string
	archives, _ := filepath.Glob(path + ".old-*")
	for _, a := range archives { // Glob returns them sorted
		recs, err := ReadFile(a)
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, keysOf(recs)...)
	}
	w2, live, err := OpenWAL(path, WALConfig{SyncMode: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	got = append(got, keysOf(live)...)
	if fmt.Sprint(got) != fmt.Sprint(all) {
		t.Fatalf("replay order %v, want %v", got, all)
	}
}

func TestReadFile_IgnoresATornTail(t *testing.T) {
	w, path := openTestWAL(t)
	appendKeys(t, w, "a", "b")
	w.Close()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte{1, 2, 3, 4, 5}) // half a header
	f.Close()
	recs, err := ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := fmt.Sprint(keysOf(recs)); got != "[a b]" {
		t.Fatalf("ReadFile = %s, want [a b]", got)
	}
}
