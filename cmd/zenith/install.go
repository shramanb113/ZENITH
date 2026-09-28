package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/shramanb113/ZENITH/internal/autostart"
)

var installFlags struct {
	dir       string
	noPath    bool
	autostart bool
}

var installCmd = &cobra.Command{
	Use:   "install",
	Short: "Install this zenith binary onto your PATH (no Go toolchain needed)",
	Long: `Copies the running zenith binary into a per-user directory and puts that
directory on your PATH, so 'zenith' works from any new terminal.

  Windows        %LOCALAPPDATA%\Programs\zenith   (user PATH; no admin needed)
  Linux / macOS  ~/.local/bin                     (a marked block is added to your shell rc)

Safe to re-run: it upgrades the installed copy in place and never duplicates
PATH entries. Your index (~/.zenith) is never touched.

Undo everything with:  zenith uninstall`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runInstall()
	},
}

func init() {
	installCmd.Flags().StringVar(&installFlags.dir, "dir", "", "Install directory (default: per-user location for your OS)")
	installCmd.Flags().BoolVar(&installFlags.noPath, "no-path", false, "Do not modify PATH / shell rc files")
	installCmd.Flags().BoolVar(&installFlags.autostart, "autostart", false, "Also register 'zenith watch start' to run at login")
}

// binaryName is the installed executable's file name for the current OS.
func binaryName() string {
	if runtime.GOOS == "windows" {
		return "zenith.exe"
	}
	return "zenith"
}

// defaultInstallDir returns the per-user install directory for this OS.
func defaultInstallDir() (string, error) {
	if runtime.GOOS == "windows" {
		if base := os.Getenv("LOCALAPPDATA"); base != "" {
			return filepath.Join(base, "Programs", "zenith"), nil
		}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("could not determine home directory: %w", err)
	}
	if runtime.GOOS == "windows" {
		return filepath.Join(home, "AppData", "Local", "Programs", "zenith"), nil
	}
	return filepath.Join(home, ".local", "bin"), nil
}

// sameFile reports whether a and b refer to the same file on disk.
func sameFile(a, b string) bool {
	ai, err := os.Stat(a)
	if err != nil {
		return false
	}
	bi, err := os.Stat(b)
	if err != nil {
		return false
	}
	return os.SameFile(ai, bi)
}

// copyBinary copies src to dstDir/binaryName() atomically (temp file + rename)
// so an interrupted install never leaves a truncated executable, and returns
// the destination path. If src already is the destination, it is a no-op.
func copyBinary(src, dstDir string) (string, error) {
	dst := filepath.Join(dstDir, binaryName())
	if sameFile(src, dst) {
		return dst, nil
	}
	if err := os.MkdirAll(dstDir, 0o755); err != nil {
		return "", fmt.Errorf("create %s: %w", dstDir, err)
	}
	in, err := os.Open(src)
	if err != nil {
		return "", err
	}
	defer in.Close()

	tmp, err := os.CreateTemp(dstDir, ".zenith-install-*")
	if err != nil {
		return "", fmt.Errorf("write to %s: %w", dstDir, err)
	}
	tmpName := tmp.Name()
	cleanup := func() { tmp.Close(); os.Remove(tmpName) }
	if _, err := io.Copy(tmp, in); err != nil {
		cleanup()
		return "", err
	}
	if err := tmp.Chmod(0o755); err != nil && runtime.GOOS != "windows" {
		cleanup()
		return "", err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return "", err
	}
	// Windows cannot replace an executable that is running; move it aside first.
	if runtime.GOOS == "windows" {
		if _, err := os.Stat(dst); err == nil {
			old := dst + ".old"
			os.Remove(old)
			if err := os.Rename(dst, old); err != nil {
				os.Remove(tmpName)
				return "", fmt.Errorf("existing %s is in use — close any running 'zenith watch' and retry: %w", dst, err)
			}
			defer os.Remove(old)
		}
	}
	if err := os.Rename(tmpName, dst); err != nil {
		os.Remove(tmpName)
		return "", err
	}
	return dst, nil
}

func runInstall() error {
	printHeader("install", "put zenith on your PATH")

	src, err := resolveBinaryPath()
	if err != nil {
		return fmt.Errorf("could not locate the running binary: %w", err)
	}

	dir := installFlags.dir
	if dir == "" {
		if dir, err = defaultInstallDir(); err != nil {
			return err
		}
	}
	if dir, err = filepath.Abs(dir); err != nil {
		return err
	}

	dst, err := copyBinary(src, dir)
	if err != nil {
		return fmt.Errorf("install failed: %w", err)
	}
	fmt.Printf("  %s  installed  %s\n", green("✓"), muted(dst))

	pathNote := ""
	if !installFlags.noPath {
		changed, note, err := ensureOnPath(dir)
		switch {
		case err != nil:
			fmt.Printf("  %s  could not update PATH automatically: %v\n", yellow("!"), err)
			fmt.Printf("       add this directory to your PATH manually: %s\n", muted(dir))
		case changed:
			fmt.Printf("  %s  added to PATH  %s\n", green("✓"), muted(dir))
			pathNote = note
		default:
			fmt.Printf("  %s  already on PATH\n", green("✓"))
		}
	}

	if installFlags.autostart {
		if err := autostart.New().Install(dst); err != nil {
			fmt.Printf("  %s  auto-start not registered: %v\n", yellow("!"), err)
		} else {
			fmt.Printf("  %s  auto-start registered ('zenith watch start' at login)\n", green("✓"))
		}
	}

	if _, err := runVersion(dst); err == nil {
		fmt.Printf("  %s  verified  %s\n", green("✓"), muted("the installed binary starts correctly"))
	} else {
		fmt.Printf("  %s  installed binary did not start: %v\n", yellow("!"), err)
	}

	printDivider()
	if pathNote != "" {
		fmt.Printf("  %s\n\n", muted(pathNote))
	}
	printFooter("zenith installed", "open a new terminal, then run 'zenith doctor'")
	return nil
}
