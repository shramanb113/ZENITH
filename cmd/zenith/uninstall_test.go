package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/autostart"
)

func TestRunUninstall_CancelledOnNo(t *testing.T) {
	tmpBin := filepath.Join(t.TempDir(), "zenith-test-binary")
	if err := os.WriteFile(tmpBin, []byte("fake binary"), 0o755); err != nil {
		t.Fatal(err)
	}

	origExec := executableFn
	executableFn = func() (string, error) { return tmpBin, nil }
	defer func() { executableFn = origExec }()

	input := strings.NewReader("no\n")
	if err := runUninstall(input); err != nil {
		t.Fatalf("runUninstall: %v", err)
	}

	if _, err := os.Stat(tmpBin); err != nil {
		t.Error("binary was deleted despite 'no' answer")
	}
}

func TestRunUninstall_CancelledOnEmpty(t *testing.T) {
	tmpBin := filepath.Join(t.TempDir(), "zenith-test-binary")
	_ = os.WriteFile(tmpBin, []byte("fake"), 0o755)

	origExec := executableFn
	executableFn = func() (string, error) { return tmpBin, nil }
	defer func() { executableFn = origExec }()

	input := strings.NewReader("\n")
	if err := runUninstall(input); err != nil {
		t.Fatalf("runUninstall: %v", err)
	}

	if _, err := os.Stat(tmpBin); err != nil {
		t.Error("binary was deleted despite empty answer")
	}
}

func TestResolveBinaryPath(t *testing.T) {
	origExec := executableFn
	executableFn = func() (string, error) { return "/tmp/zenith", nil }
	defer func() { executableFn = origExec }()

	p, err := resolveBinaryPath()
	if err != nil {
		t.Fatalf("resolveBinaryPath: %v", err)
	}
	if p == "" {
		t.Error("expected non-empty path")
	}
}

// ── seam-based tests: never touch real autostart / PATH / user data ──────────

type fakeAutostart struct {
	installed   bool
	uninstalled bool
}

func (f *fakeAutostart) Install(string) error { f.installed = true; return nil }
func (f *fakeAutostart) Uninstall() error     { f.uninstalled = true; f.installed = false; return nil }
func (f *fakeAutostart) IsInstalled() (bool, error) {
	return f.installed, nil
}

type uninstallEnv struct {
	bin, dataDir, nerveDir string
	auto                   *fakeAutostart
	pathCleaned            bool
}

func newUninstallEnv(t *testing.T, binName string) *uninstallEnv {
	t.Helper()
	root := t.TempDir()
	env := &uninstallEnv{
		bin:      filepath.Join(root, "bin", binName),
		dataDir:  filepath.Join(root, ".zenith"),
		nerveDir: filepath.Join(root, ".zenith", "nerve"),
		auto:     &fakeAutostart{installed: true},
	}
	for _, d := range []string{filepath.Dir(env.bin), env.dataDir, env.nerveDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(env.bin, []byte("bin"), 0o755)
	os.WriteFile(filepath.Join(env.dataDir, "zenith.db"), []byte("index"), 0o644)
	os.WriteFile(filepath.Join(env.nerveDir, "junk"), []byte("py"), 0o644)

	orig := struct {
		exe  func() (string, error)
		auto func() autostart.Manager
		path func(string) (bool, error)
		bin  func(string) error
		dd   func() (string, error)
		data func() string
		nerv func() string
	}{executableFn, autostartFn, removePathFn, removeBinFn, defaultDirFn, dataDirFn, legacyNerveFn}
	executableFn = func() (string, error) { return env.bin, nil }
	autostartFn = func() autostart.Manager { return env.auto }
	removePathFn = func(string) (bool, error) { env.pathCleaned = true; return true, nil }
	removeBinFn = os.Remove
	defaultDirFn = func() (string, error) { return filepath.Dir(env.bin), nil }
	dataDirFn = func() string { return env.dataDir }
	legacyNerveFn = func() string { return env.nerveDir }
	uninstallFlags.yes, uninstallFlags.purge = false, false
	t.Cleanup(func() {
		executableFn, autostartFn, removePathFn, removeBinFn = orig.exe, orig.auto, orig.path, orig.bin
		defaultDirFn, dataDirFn, legacyNerveFn = orig.dd, orig.data, orig.nerv
		uninstallFlags.yes, uninstallFlags.purge = false, false
	})
	return env
}

func exists(p string) bool { _, err := os.Stat(p); return err == nil }

func TestUninstall_KeepsIndexDataByDefault(t *testing.T) {
	env := newUninstallEnv(t, "zenith")
	uninstallFlags.yes = true
	if err := runUninstall(strings.NewReader("")); err != nil {
		t.Fatalf("runUninstall: %v", err)
	}
	if exists(env.bin) {
		t.Error("binary should be removed")
	}
	if !env.auto.uninstalled {
		t.Error("auto-start should be removed")
	}
	if !env.pathCleaned {
		t.Error("PATH entry should be cleaned")
	}
	if exists(env.nerveDir) {
		t.Error("legacy nerve dir should be removed")
	}
	if !exists(filepath.Join(env.dataDir, "zenith.db")) {
		t.Fatal("index data was deleted without --purge — data loss")
	}
}

func TestUninstall_PurgeDeletesData(t *testing.T) {
	env := newUninstallEnv(t, "zenith")
	uninstallFlags.yes, uninstallFlags.purge = true, true
	if err := runUninstall(strings.NewReader("")); err != nil {
		t.Fatalf("runUninstall: %v", err)
	}
	if exists(env.dataDir) {
		t.Error("--purge should delete the data directory")
	}
}

func TestUninstall_PromptRequiresYes(t *testing.T) {
	env := newUninstallEnv(t, "zenith")
	if err := runUninstall(strings.NewReader("y\n")); err != nil {
		t.Fatal(err)
	}
	if !exists(env.bin) || env.auto.uninstalled {
		t.Error("anything other than 'yes' must change nothing")
	}
	if err := runUninstall(strings.NewReader("yes\n")); err != nil {
		t.Fatal(err)
	}
	if exists(env.bin) {
		t.Error("'yes' should proceed")
	}
}

func TestUninstall_DoesNotDeleteUnrelatedExecutable(t *testing.T) {
	env := newUninstallEnv(t, "main.test")
	uninstallFlags.yes = true
	if err := runUninstall(strings.NewReader("")); err != nil {
		t.Fatal(err)
	}
	if !exists(env.bin) {
		t.Error("an executable not named zenith must be left alone")
	}
}

func TestLooksLikeZenith(t *testing.T) {
	for p, want := range map[string]bool{
		"/x/zenith": true, `C:\a\zenith.exe`: true, "/x/zenith-test-binary": true,
		"/x/main.test": false, "/x/go-build123/exe/zenithx": false, "/x/node": false,
	} {
		if got := looksLikeZenith(p); got != want {
			t.Errorf("looksLikeZenith(%q) = %v, want %v", p, got, want)
		}
	}
}
