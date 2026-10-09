package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/shramanb113/ZENITH/internal/autostart"
	"github.com/shramanb113/ZENITH/internal/crawler"
	"github.com/shramanb113/ZENITH/internal/fileindex"
	imageindexer "github.com/shramanb113/ZENITH/internal/image"
	"github.com/shramanb113/ZENITH/internal/pdf"
	"github.com/shramanb113/ZENITH/internal/watchlist"
	"github.com/spf13/cobra"
)

// ── parent ─────────────────────────────────────────────────────────────────────

var watchCmd = &cobra.Command{
	Use:   "watch",
	Short: "Manage persistent directory watching and OS boot auto-start",
	Long: `watch manages a persistent list of directories to keep indexed in real time.
File changes (create, modify, rename, delete) are detected immediately and
re-indexed without manual intervention.

Quick start:
  zenith watch add ~/Documents       # register a directory
  zenith watch install               # auto-start on every boot
  zenith watch start                 # start watching now (also runs at boot)

Subcommands:
  add        Add a directory to the watchlist
  remove     Remove a directory from the watchlist
  list       Show the current watchlist
  start      Watch all directories in the watchlist (blocking)
  run        One-shot watch of a single directory (not saved)
  install    Register OS boot auto-start
  uninstall  Remove OS boot auto-start`,
}

// ── watch add ─────────────────────────────────────────────────────────────────

var watchAddCmd = &cobra.Command{
	Use:   "add <directory>",
	Short: "Add a directory to the persistent watchlist",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := watchlist.Add(args[0]); err != nil {
			return err
		}
		abs, _ := filepath.Abs(args[0])
		fmt.Printf("\n  %s  %s\n", green("✓"), abs)
		fmt.Printf("  %s\n\n", dim("Run 'zenith watch start' to begin watching."))
		return nil
	},
}

// ── watch remove ──────────────────────────────────────────────────────────────

var watchRemoveCmd = &cobra.Command{
	Use:   "remove <directory>",
	Short: "Remove a directory from the watchlist",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		if err := watchlist.Remove(args[0]); err != nil {
			return err
		}
		abs, _ := filepath.Abs(args[0])
		fmt.Printf("\n  %s  removed: %s\n\n", green("✓"), abs)
		return nil
	},
}

// ── watch list ────────────────────────────────────────────────────────────────

var watchListCmd = &cobra.Command{
	Use:   "list",
	Short: "Show all directories in the watchlist",
	RunE: func(cmd *cobra.Command, args []string) error {
		dirs, err := watchlist.Load()
		if err != nil {
			return err
		}
		if len(dirs) == 0 {
			fmt.Println("\n  Watchlist is empty.")
			fmt.Println(dim("  Add a directory with: zenith watch add <directory>"))
			fmt.Println()
			return nil
		}
		fmt.Printf("\n  %s\n\n", bold("Watched directories"))
		for i, d := range dirs {
			fmt.Printf("  %s  %s\n", dim(fmt.Sprintf("%2d", i+1)), d)
		}
		fmt.Println()
		return nil
	},
}

// ── watch start ───────────────────────────────────────────────────────────────

var watchStartCmd = &cobra.Command{
	Use:   "start",
	Short: "Start watching all directories in the watchlist (blocking)",
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()

		dirs, err := watchlist.Load()
		if err != nil {
			return err
		}

		var valid []string
		for _, d := range dirs {
			if info, err := os.Stat(d); err == nil && info.IsDir() {
				valid = append(valid, d)
			} else {
				fmt.Printf("  %s  skipping missing directory: %s\n", yellow("!"), d)
			}
		}

		if len(valid) == 0 {
			fmt.Println("\n  No valid directories to watch.")
			fmt.Println(dim("  Add one with: zenith watch add <directory>"))
			fmt.Println()
			return nil
		}

		engine, _, alog, teardown, err := buildEngine(true, true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		defer teardown()

		w, err := crawler.NewWatcher(engine, alog)
		if err != nil {
			return err
		}
		defer w.Close()

		pi := pdf.NewIndexer(engine, alog)
		w.RegisterFileIndexer(".pdf", pi.WithAttrs(nil))
		ii := imageindexer.NewIndexer(engine, alog)
		for _, ext := range []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"} {
			w.RegisterFileIndexer(ext, ii)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		go func() {
			<-sigCh
			fmt.Println("\n  Shutting down watcher...")
			cancel()
		}()

		pluralSuffix := "ies"
		if len(valid) == 1 {
			pluralSuffix = "y"
		}
		printHeader("watch", fmt.Sprintf("%d director%s", len(valid), pluralSuffix))
		for _, d := range valid {
			fmt.Printf("  %s  %s\n", cc("→", cAccent), muted(d))
		}
		fmt.Printf("\n  %s\n\n", muted("Ctrl-C to stop"))

		return w.WatchMultiple(ctx, valid)
	},
}

// ── watch run ─────────────────────────────────────────────────────────────────

var watchRunFlags struct {
	indexFirst bool
}

var watchRunCmd = &cobra.Command{
	Use:   "run <directory>",
	Short: "One-shot watch of a single directory (not saved to watchlist)",
	Long: `Watches <directory> for file-system events and incrementally re-indexes on changes.
The directory is NOT added to the persistent watchlist.

Use --index-first to bulk-index before watching.
Use 'zenith watch add' + 'zenith watch start' for persistent watching.`,
	Args: cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		setupLogger()
		dir := args[0]

		if info, err := os.Stat(dir); err != nil || !info.IsDir() {
			return fmt.Errorf("not a directory: %s", dir)
		}

		engine, _, alog, teardown, err := buildEngine(true, true)
		if err != nil {
			return fmt.Errorf("engine init: %w", err)
		}
		defer teardown()

		w, err := crawler.NewWatcher(engine, alog)
		if err != nil {
			return err
		}
		defer w.Close()

		pi := pdf.NewIndexer(engine, alog)
		w.RegisterFileIndexer(".pdf", pi.WithAttrs(nil))
		ii := imageindexer.NewIndexer(engine, alog)
		for _, ext := range []string{".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"} {
			w.RegisterFileIndexer(ext, ii)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		go func() {
			<-sigCh
			fmt.Println("\n  Shutting down watcher...")
			cancel()
		}()

		if watchRunFlags.indexFirst {
			fmt.Printf("  Bulk-indexing %s ...\n", dir)
			if fi, err := fileindex.Open(effectiveFileHashPath()); err == nil {
				w.SetSkipFile(fi.IsUpToDate)
				w.SetAfterFile(func(path string) { _ = fi.Mark(path) })
				defer fi.Save()
			}
			if err := w.IndexDir(ctx, dir); err != nil {
				return fmt.Errorf("initial index: %w", err)
			}
		}

		printHeader("watch", dir)
		fmt.Printf("  %s\n\n", muted("Ctrl-C to stop"))
		return w.Watch(ctx, dir)
	},
}

// ── watch install ─────────────────────────────────────────────────────────────

var watchInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Register 'zenith watch start' as an OS boot auto-start",
	Long: `Registers 'zenith watch start' to run automatically at user login.

  Windows  — Task Scheduler task "ZenithWatch" (no admin required)
  Linux    — systemd user service ~/.config/systemd/user/zenith-watch.service
  macOS    — launchd agent ~/Library/LaunchAgents/com.zenith.watch.plist

Run 'zenith watch start' in a separate terminal to begin watching immediately,
or wait for the next login where it will start automatically.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		execPath, err := os.Executable()
		if err != nil {
			return fmt.Errorf("could not determine binary path: %w", err)
		}
		if resolved, err := filepath.EvalSymlinks(execPath); err == nil {
			execPath = resolved
		}

		mgr := autostart.New()
		if err := mgr.Install(execPath); err != nil {
			return fmt.Errorf("install failed: %w", err)
		}
		fmt.Printf("\n  %s  auto-start registered\n", green("✓"))
		fmt.Printf("       binary: %s\n", dim(execPath))
		fmt.Printf("\n  %s\n\n", dim("'zenith watch start' will run on every boot."))
		return nil
	},
}

// ── watch uninstall ───────────────────────────────────────────────────────────

var watchUninstallCmd = &cobra.Command{
	Use:   "uninstall",
	Short: "Remove the OS boot auto-start entry",
	RunE: func(cmd *cobra.Command, args []string) error {
		mgr := autostart.New()
		if err := mgr.Uninstall(); err != nil {
			return fmt.Errorf("uninstall failed: %w", err)
		}
		fmt.Printf("\n  %s  auto-start removed\n\n", green("✓"))
		return nil
	},
}

// ── init ──────────────────────────────────────────────────────────────────────

func init() {
	watchCmd.AddCommand(
		watchAddCmd,
		watchRemoveCmd,
		watchListCmd,
		watchStartCmd,
		watchRunCmd,
		watchInstallCmd,
		watchUninstallCmd,
	)
	addEngineFlags(watchStartCmd)
	addEngineFlags(watchRunCmd)
	watchRunCmd.Flags().BoolVar(&watchRunFlags.indexFirst, "index-first", false,
		"Bulk-index directory before starting the watcher")
}
