package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRCBlock_AddIsIdempotentAndRemovable(t *testing.T) {
	orig := "export FOO=1\nalias ll='ls -l'\n"
	once := addRCBlock(orig, "/home/u/.local/bin")
	twice := addRCBlock(once, "/home/u/.local/bin")
	if once != twice {
		t.Error("adding the block twice must not duplicate it")
	}
	if strings.Count(twice, rcBeginMarker) != 1 {
		t.Errorf("want exactly one block, got:\n%s", twice)
	}
	if !strings.Contains(once, `export PATH="/home/u/.local/bin:$PATH"`) {
		t.Errorf("block missing PATH export:\n%s", once)
	}
	if got := removeRCBlock(once); got != orig {
		t.Errorf("removal must restore the original file exactly.\ngot:  %q\nwant: %q", got, orig)
	}
}

func TestRCBlock_ChangedDirReplacesOldBlock(t *testing.T) {
	c := addRCBlock("", "/old")
	c = addRCBlock(c, "/new")
	if strings.Contains(c, "/old") || !strings.Contains(c, "/new") {
		t.Errorf("stale block not replaced:\n%s", c)
	}
}

func TestRemoveRCBlock_NoBlockIsNoop(t *testing.T) {
	in := "just some\nshell config\n"
	if removeRCBlock(in) != in {
		t.Error("content without a block must be untouched")
	}
}

func TestPathContains(t *testing.T) {
	sep := string(os.PathListSeparator)
	list := strings.Join([]string{"/a/bin", "/b/bin", "/c/bin"}, sep)
	if !pathContains(list, "/b/bin") || pathContains(list, "/x") {
		t.Error("pathContains wrong")
	}
	if !pathContains(list, "/b/bin/") {
		t.Error("trailing separator should be ignored")
	}
}

func TestCopyBinary_InstallsAndUpgradesInPlace(t *testing.T) {
	src := filepath.Join(t.TempDir(), "src-zenith")
	os.WriteFile(src, []byte("v1"), 0o755)
	dir := filepath.Join(t.TempDir(), "nested", "bin")

	dst, err := copyBinary(src, dir)
	if err != nil {
		t.Fatalf("copyBinary: %v", err)
	}
	if got, _ := os.ReadFile(dst); string(got) != "v1" {
		t.Fatalf("installed content = %q", got)
	}

	os.WriteFile(src, []byte("v2"), 0o755)
	if _, err := copyBinary(src, dir); err != nil {
		t.Fatalf("upgrade: %v", err)
	}
	if got, _ := os.ReadFile(dst); string(got) != "v2" {
		t.Fatalf("upgraded content = %q", got)
	}

	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Errorf("install dir should hold only the binary, has %d entries", len(entries))
	}
}

func TestCopyBinary_SameFileIsNoop(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, binaryName())
	os.WriteFile(p, []byte("x"), 0o755)
	dst, err := copyBinary(p, dir)
	if err != nil || dst != p {
		t.Fatalf("dst=%q err=%v", dst, err)
	}
	if got, _ := os.ReadFile(p); string(got) != "x" {
		t.Error("installing a binary onto itself must not truncate it")
	}
}
