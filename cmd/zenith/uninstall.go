package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"github.com/shramanb113/ZENITH/internal/autostart"
)

// Seams replaced in tests so they never touch the real OS autostart entry,
// PATH, or delete the test binary via a detached process.
var (
	executableFn  = os.Executable
	autostartFn   = func() autostart.Manager { return autostart.New() }
	removePathFn  = removeFromPath
	removeBinFn   = removeBinary
	defaultDirFn  = defaultInstallDir
	dataDirFn     = func() string { return zenithDataPath("") }
	legacyNerveFn = func() string { return zenithDataPath("nerve") }
)

var uninstallFlags struct {
	yes   bool
	purge bool
}

var uninstallCmd = &cobra.Command{
	Use:   "uninstall",
	Short: "Remove zenith (binary, PATH entry, auto-start). Index data is kept unless --purge",
	Long: `Removes everything 'zenith install' set up:

  - the zenith binary
  - the PATH entry / shell rc block added by 'zenith install'
  - the login auto-start entry from 'zenith watch install'
  - leftover Python sidecar files from very old versions (~/.zenith/nerve)

Your search index and settings in ~/.zenith (zenith.db, data/, watchlist.json,
zenith.log) are KEPT by default so reinstalling picks up where you left off.
Pass --purge to delete them too — that cannot be undone.

You are asked to confirm first; use --yes to skip the prompt in scripts.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runUninstall(os.Stdin)
	},
}

func init() {
	uninstallCmd.Flags().BoolVarP(&uninstallFlags.yes, "yes", "y", false, "Do not ask for confirmation")
	uninstallCmd.Flags().BoolVar(&uninstallFlags.purge, "purge", false,
		"Also delete all index data and settings in ~/.zenith (irreversible)")
}

func runUninstall(input io.Reader) error {
	binPath, binErr := resolveBinaryPath()
	dataDir := dataDirFn()
	nerveDir := legacyNerveFn()
	mgr := autostartFn()
	autoOn, _ := mgr.IsInstalled()

	printHeader("uninstall", "remove zenith from this machine")

	fmt.Printf("  %s  will be removed:\n", yellow("!"))
	if binErr == nil {
		fmt.Printf("       %s  %s\n", cc("✗", cErrCol), muted(binPath+"  (binary)"))
	}
	fmt.Printf("       %s  %s\n", cc("✗", cErrCol), muted("PATH entry added by 'zenith install' (if any)"))
	if autoOn {
		fmt.Printf("       %s  %s\n", cc("✗", cErrCol), muted("login auto-start entry"))
	}
	if _, err := os.Stat(nerveDir); err == nil {
		fmt.Printf("       %s  %s\n", cc("✗", cErrCol), muted(nerveDir+"  (old Python sidecar)"))
	}

	dataSize := dirSizeBytes(dataDir)
	if uninstallFlags.purge {
		fmt.Printf("       %s  %s\n", cc("✗", cErrCol),
			cc(fmt.Sprintf("%s  (ALL index data + settings, %s — irreversible)", dataDir, formatBytes(dataSize)), cErrCol))
	} else {
		fmt.Printf("\n  %s  kept:\n", green("✓"))
		fmt.Printf("       %s\n", muted(fmt.Sprintf("%s  (%s: index, watchlist, logs)", dataDir, formatBytes(dataSize))))
		fmt.Printf("       %s\n", muted("re-run with --purge to delete this too"))
		fmt.Printf("       %s\n", muted("(an index stored via a custom --db path is never touched)"))
	}
	fmt.Println()

	if !uninstallFlags.yes {
		fmt.Printf("  Type %s to continue: ", cc(`"yes"`, ansiBold+cBrand))
		scanner := bufio.NewScanner(input)
		scanner.Scan()
		if strings.TrimSpace(strings.ToLower(scanner.Text())) != "yes" {
			fmt.Printf("\n  %s  uninstall cancelled — nothing was changed\n\n", muted("·"))
			return nil
		}
	}
	fmt.Println()

	var failures []string
	fail := func(what string, err error) {
		failures = append(failures, fmt.Sprintf("%s: %v", what, err))
		fmt.Printf("  %s  %s: %v\n", yellow("!"), what, err)
	}

	// 1. Auto-start first, so nothing relaunches zenith while we remove it.
	if autoOn {
		if err := mgr.Uninstall(); err != nil {
			fail("removing auto-start", err)
		} else {
			fmt.Printf("  %s  auto-start removed\n", green("✓"))
		}
	}

	// 2. PATH entry / rc block.
	if pdir, err := defaultDirFn(); err == nil {
		if changed, err := removePathFn(pdir); err != nil {
			fail("cleaning PATH", err)
		} else if changed {
			fmt.Printf("  %s  PATH entry removed\n", green("✓"))
		}
	}

	// 3. Legacy Python sidecar (always safe: never holds user data).
	if _, err := os.Stat(nerveDir); err == nil {
		if err := os.RemoveAll(nerveDir); err != nil {
			fail("removing "+nerveDir, err)
		} else {
			fmt.Printf("  %s  %s\n", green("✓"), muted(nerveDir+" removed"))
		}
	}

	// 4. Index data — only on explicit --purge.
	if uninstallFlags.purge {
		if err := os.RemoveAll(dataDir); err != nil {
			fail("removing "+dataDir, err)
		} else {
			fmt.Printf("  %s  %s\n", green("✓"), muted(dataDir+" removed ("+formatBytes(dataSize)+" freed)"))
		}
	}

	// 5. The binary itself, last.
	if binErr == nil {
		if !looksLikeZenith(binPath) {
			fmt.Printf("  %s  %s\n", yellow("!"),
				fmt.Sprintf("%s does not look like an installed zenith binary — left in place", binPath))
		} else if err := removeBinFn(binPath); err != nil {
			fail("removing binary", err)
			fmt.Printf("       delete manually: %s\n", muted(binPath))
		} else {
			fmt.Printf("  %s  %s\n", green("✓"), muted(binPath+" removed"))
			_ = os.Remove(filepath.Dir(binPath)) // only succeeds if now empty
		}
	}

	printDivider()
	if len(failures) > 0 {
		printFooter("uninstall finished with problems", fmt.Sprintf("%d step(s) need manual attention", len(failures)))
		return fmt.Errorf("uninstall incomplete: %s", strings.Join(failures, "; "))
	}
	if uninstallFlags.purge {
		printFooter("zenith fully removed", "open a new terminal so PATH changes apply")
	} else {
		printFooter("zenith removed", "index data kept in "+shortenPath(dataDir))
	}
	return nil
}

// looksLikeZenith guards against deleting an unrelated executable (for
// example a 'go run' temp build or a test binary) when uninstall is invoked
// from somewhere unexpected.
func looksLikeZenith(p string) bool {
	base := strings.ToLower(filepath.Base(p))
	base = strings.TrimSuffix(base, ".exe")
	return base == "zenith" || strings.HasPrefix(base, "zenith-") || strings.HasPrefix(base, "zenith_")
}

func resolveBinaryPath() (string, error) {
	p, err := executableFn()
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(p)
	if err != nil {
		return p, nil
	}
	return resolved, nil
}
