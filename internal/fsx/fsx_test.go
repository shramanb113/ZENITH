package fsx

import (
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func read(t *testing.T, dir, name string) (string, bool) {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		if os.IsNotExist(err) {
			return "", false
		}
		t.Fatal(err)
	}
	return string(b), true
}

func TestRecorder_CrashModel(t *testing.T) {
	root := t.TempDir()
	rec := NewRecorder(root)
	defer SetHooks(rec)()

	// data.tmp is written, fsynced, renamed over data (atomic replace) but the
	// directory is not synced; other.txt is written and never fsynced.
	f, err := Create(filepath.Join(root, "data.tmp"))
	if err != nil {
		t.Fatal(err)
	}
	f.Write([]byte("hello"))
	f.Sync()
	f.Close()
	if err := Rename(filepath.Join(root, "data.tmp"), filepath.Join(root, "data")); err != nil {
		t.Fatal(err)
	}
	afterRename := rec.Len()
	SyncDir(root)
	afterSyncDir := rec.Len()
	g, _ := Create(filepath.Join(root, "other.txt"))
	g.Write([]byte("unsynced"))
	g.Close()
	end := rec.Len()

	check := func(k int, p Persistence, seed int64) string {
		dst := filepath.Join(t.TempDir(), "c")
		if err := rec.CrashState(k, p, seed, dst); err != nil {
			t.Fatal(err)
		}
		var out string
		for _, n := range []string{"data", "data.tmp", "other.txt"} {
			if s, ok := read(t, dst, n); ok {
				out += n + "=" + s + ";"
			}
		}
		return out
	}

	// Before the directory sync the rename may not have happened: nothing is
	// guaranteed to exist yet (the create itself was never directory-synced).
	if got := check(afterRename, Nothing, 0); got != "" {
		t.Errorf("before any directory sync, pessimistic crash keeps %q, want nothing", got)
	}
	// If everything reaches disk, the renamed file is there.
	if got := check(afterRename, Everything, 0); got != "data=hello;" {
		t.Errorf("everything-survives crash = %q", got)
	}
	// Renames are atomic: in every torn outcome data.tmp and data are never both
	// missing once the create was durable... here neither was durable, so only
	// check the atomicity of an outcome that has either.
	for seed := int64(0); seed < 40; seed++ {
		got := check(afterRename, Torn, seed)
		if got != "" && got != "data.tmp=hello;" && got != "data=hello;" {
			t.Fatalf("torn crash produced an impossible state %q", got)
		}
	}
	// After the directory sync, the file is durable under its final name.
	if got := check(afterSyncDir, Nothing, 0); got != "data=hello;" {
		t.Errorf("after fsync+rename+dirsync, pessimistic crash = %q, want data=hello", got)
	}
	// A file that was never fsynced (and never directory-synced) does not survive
	// the pessimistic crash, but its content does under Everything.
	if got := check(end, Nothing, 0); got != "data=hello;" {
		t.Errorf("pessimistic crash at end = %q, want only the durable file", got)
	}
	if got := check(end, Everything, 0); got != "data=hello;other.txt=unsynced;" {
		t.Errorf("everything-survives crash at end = %q", got)
	}
}

func TestRecorder_FileNamedButDataNotSynced(t *testing.T) {
	root := t.TempDir()
	rec := NewRecorder(root)
	defer SetHooks(rec)()
	f, _ := Create(filepath.Join(root, "seg"))
	f.Write([]byte("0123456789"))
	f.Close()
	SyncDir(root) // the name is durable, the data is not
	dst := filepath.Join(t.TempDir(), "c")
	if err := rec.CrashState(rec.Len(), Nothing, 0, dst); err != nil {
		t.Fatal(err)
	}
	if s, ok := read(t, dst, "seg"); !ok || s != "" {
		t.Fatalf("pessimistic crash: file exists=%v content=%q, want an empty file", ok, s)
	}
}

func TestDiskFull_CutsAWriteAtTheBudget(t *testing.T) {
	root := t.TempDir()
	df := NewDiskFull(7)
	defer SetHooks(df)()
	f, err := Create(filepath.Join(root, "x"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	n, err := f.Write([]byte("0123456789"))
	if n != 7 || !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("Write = %d, %v; want 7, ENOSPC", n, err)
	}
	if _, err := f.Write([]byte("a")); !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("second Write err = %v, want ENOSPC", err)
	}
	df.SetBudget(-1)
	if _, err := f.Write([]byte("ok")); err != nil {
		t.Fatalf("Write after space returned: %v", err)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "x")); string(b) != "0123456ok" {
		t.Fatalf("file = %q, want 0123456ok", b)
	}
}
