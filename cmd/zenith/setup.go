package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/cobra"
)

var setupFlags struct {
	clean bool
}

var setupCmd = &cobra.Command{
	Use:   "setup",
	Short: "Migration helper for users upgrading from a Python-sidecar version",
	Long: `ZENITH now uses an embedded model — no Python or setup step required.

If you are upgrading from an older version, run:

  zenith setup --clean

This removes leftover Python files from ~/.zenith/nerve/ and recovers disk space.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runSetup()
	},
}

func init() {
	setupCmd.Flags().BoolVar(&setupFlags.clean, "clean", false,
		"Remove leftover Python sidecar files from a previous installation")
}

func runSetup() error {
	printHeader("setup", "ZENITH is ready — no setup required")

	if setupFlags.clean {
		runMigration(true)
		return nil
	}

	fmt.Printf("  %s  Embedded model loaded automatically — nothing to install.\n\n", green("✓"))
	fmt.Printf("  %s  To remove leftover Python files from an older version:\n", muted("·"))
	fmt.Printf("       zenith setup --clean\n\n")
	printDivider()
	printFooter("ZENITH is ready", "run 'zenith index <directory>' to get started")
	return nil
}

// runMigration cleans up Python nerve artifacts from a previous installation.
// When verbose is true, progress is printed to stdout.
// Called automatically on first run when ~/.zenith/nerve/ exists, and manually
// via 'zenith setup --clean'.
func runMigration(verbose bool) {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	nerveDir := filepath.Join(home, ".zenith", "nerve")
	if _, err := os.Stat(nerveDir); os.IsNotExist(err) {
		if verbose {
			fmt.Printf("  %s  No Python sidecar files found — nothing to clean.\n\n", green("✓"))
		}
		return
	}

	// Kill any running nerve process using the saved PID file.
	pidFile := filepath.Join(nerveDir, "nerve.pid")
	if data, err := os.ReadFile(pidFile); err == nil {
		if pid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil {
			if proc, err := os.FindProcess(pid); err == nil {
				_ = proc.Kill()
			}
		}
	}

	size := dirSizeBytes(nerveDir)

	if err := os.RemoveAll(nerveDir); err != nil && verbose {
		fmt.Printf("  %s  Could not remove %s: %v\n", yellow("!"), nerveDir, err)
	} else if verbose {
		fmt.Printf("  %s  Removed %s (%s freed)\n", green("✓"), nerveDir, formatBytes(size))
	}

	// Remove the old setup sentinel (keyed to nerve's hash, now meaningless).
	_ = os.Remove(filepath.Join(home, ".zenith", "setup_ok"))

	// Write migration sentinel so auto-migration runs exactly once.
	_ = os.WriteFile(filepath.Join(home, ".zenith", "migrated_v2"), []byte("done"), 0o644)
}

// autoMigrate runs a silent one-time cleanup when the new binary finds leftover
// Python nerve artifacts from a previous installation. Called from main() before
// any command executes.
func autoMigrate() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	// Already migrated — skip.
	if _, err := os.Stat(filepath.Join(home, ".zenith", "migrated_v2")); err == nil {
		return
	}
	// No nerve dir present — mark done and return.
	nerveDir := filepath.Join(home, ".zenith", "nerve")
	if _, err := os.Stat(nerveDir); os.IsNotExist(err) {
		_ = os.WriteFile(filepath.Join(home, ".zenith", "migrated_v2"), []byte("done"), 0o644)
		return
	}
	// Measure size BEFORE removal.
	size := dirSizeBytes(nerveDir)
	runMigration(false)
	fmt.Fprintf(os.Stderr, "\n  %s  Migrated to native Go (%s freed)\n\n",
		green("✓"), formatBytes(size))
}

func dirSizeBytes(path string) int64 {
	var size int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

func formatBytes(b int64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(b)/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(b)/(1<<20))
	default:
		return fmt.Sprintf("%d KB", b>>10)
	}
}
