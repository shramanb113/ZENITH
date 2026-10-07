package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/spf13/cobra"

	"github.com/shramanb113/ZENITH/internal/image"
	"github.com/shramanb113/ZENITH/internal/index"
	"github.com/shramanb113/ZENITH/internal/localembedder"
)

var doctorFlags struct {
	json bool
	db   string
}

func init() {
	doctorCmd.Flags().BoolVar(&doctorFlags.json, "json", false,
		"Print a machine-readable report (safe to attach to a support request: paths and versions only, never index contents)")
	doctorCmd.Flags().StringVar(&doctorFlags.db, "db", "", "Index file to check (default: ~/.zenith/zenith.db)")
}

var doctorCmd = &cobra.Command{
	Use:   "doctor",
	Short: "Diagnose install problems (PATH, data dir, auto-start, embedder)",
	Long: `Checks the things that most often go wrong after installing:

  - is this binary on your PATH, and is it the one your shell will run?
  - does ~/.zenith exist and is it writable? how big is the index?
  - is the login auto-start registered?
  - can the embedded ONNX model load, or is search running lexical-only?
  - is a Go toolchain available for 'zenith update'?

Each problem comes with the command that fixes it. Exit status is non-zero if
anything needs attention.`,
	RunE: func(cmd *cobra.Command, args []string) error {
		return runDoctor()
	},
}

type doctorCheck struct {
	name string
	ok   bool
	warn bool // not fatal, but worth knowing
	msg  string
	fix  string
}

func runDoctor() error {
	if !doctorFlags.json {
		printHeader("doctor", "checking your zenith installation")
	}
	var checks []doctorCheck

	// Binary + PATH.
	bin, err := resolveBinaryPath()
	if err != nil {
		checks = append(checks, doctorCheck{name: "binary", msg: "cannot locate the running binary: " + err.Error()})
	} else {
		checks = append(checks, doctorCheck{name: "binary", ok: true, msg: fmt.Sprintf("%s (version %s)", bin, version)})
		onPath, lerr := exec.LookPath("zenith")
		switch {
		case lerr != nil:
			checks = append(checks, doctorCheck{name: "PATH", msg: "'zenith' is not found on your PATH",
				fix: "zenith install     (then open a new terminal)"})
		case !sameFile(onPath, bin):
			checks = append(checks, doctorCheck{name: "PATH", warn: true,
				msg: fmt.Sprintf("your shell runs a different zenith (%s) than this one (%s)", onPath, bin),
				fix: "zenith install     (updates the copy on PATH), or remove the stale copy"})
		default:
			checks = append(checks, doctorCheck{name: "PATH", ok: true, msg: "on PATH: " + onPath})
		}
	}

	// Data directory.
	dataDir := dataDirFn()
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		checks = append(checks, doctorCheck{name: "data dir", msg: fmt.Sprintf("cannot create %s: %v", dataDir, err),
			fix: "check permissions on your home directory"})
	} else if f, err := os.CreateTemp(dataDir, ".doctor-*"); err != nil {
		checks = append(checks, doctorCheck{name: "data dir", msg: fmt.Sprintf("%s is not writable: %v", dataDir, err),
			fix: "fix the directory permissions"})
	} else {
		f.Close()
		os.Remove(f.Name())
		msg := dataDir + " (empty — nothing indexed yet)"
		if size := dirSizeBytes(dataDir); size > 0 {
			msg = fmt.Sprintf("%s (%s)", dataDir, formatBytes(size))
		}
		checks = append(checks, doctorCheck{name: "data dir", ok: true, msg: msg})
	}

	dbPath := filepath.Join(dataDir, "zenith.db")
	if doctorFlags.db != "" {
		dbPath = doctorFlags.db
	}
	checks = append(checks, indexChecks(dbPath)...)

	// Auto-start.
	if on, _ := autostartFn().IsInstalled(); on {
		checks = append(checks, doctorCheck{name: "auto-start", ok: true, msg: "registered (zenith watch start at login)"})
	} else {
		checks = append(checks, doctorCheck{name: "auto-start", warn: true, msg: "not registered (optional)", fix: "zenith watch install"})
	}

	// Embedder.
	if _, err := localembedder.New(); err != nil {
		checks = append(checks, doctorCheck{name: "embedder", warn: true,
			msg: "ONNX model unavailable — semantic search disabled, lexical search works normally (" + err.Error() + ")",
			fix: "use a build with CGO_ENABLED=1 and a C compiler, or run 'zenith index --embedder ollama'"})
	} else {
		checks = append(checks, doctorCheck{name: "embedder", ok: true, msg: "embedded ONNX model loads (semantic search on)"})
	}

	// Image OCR (opt-in, built with -tags ocr against libtesseract).
	if image.OCRAvailable() {
		checks = append(checks, doctorCheck{name: "image OCR", ok: true, msg: "Tesseract available (images indexed by pixel text + filename)"})
	} else {
		checks = append(checks, doctorCheck{name: "image OCR", warn: true,
			msg: "Tesseract not built in — images indexed by filename only",
			fix: "build with CGO_ENABLED=1 -tags ocr against libtesseract (see the Docker image)"})
	}

	// Go toolchain (only needed for 'zenith update').
	if gp, err := exec.LookPath("go"); err != nil {
		checks = append(checks, doctorCheck{name: "go", warn: true, msg: "no Go toolchain — 'zenith update' will not work (optional)",
			fix: "download a newer release binary and run 'zenith install' again"})
	} else {
		checks = append(checks, doctorCheck{name: "go", ok: true, msg: gp})
	}

	if doctorFlags.json {
		return emitDoctorJSON(checks)
	}

	problems := 0
	for _, c := range checks {
		mark := green("✓")
		switch {
		case !c.ok && !c.warn:
			mark = cc("✗", cErrCol)
			problems++
		case c.warn:
			mark = yellow("!")
		}
		fmt.Printf("  %s  %-10s %s\n", mark, c.name, muted(c.msg))
		if c.fix != "" && !c.ok {
			fmt.Printf("       %s %s\n", muted("fix:"), c.fix)
		}
	}
	printDivider()
	if problems > 0 {
		printFooter("problems found", fmt.Sprintf("%d issue(s) need attention", problems))
		return fmt.Errorf("%d check(s) failed", problems)
	}
	printFooter("all good")
	return nil
}

// emitDoctorJSON prints the report as JSON. It contains only versions, OS/arch,
// file paths and check outcomes — never index contents, queries or document
// text — so users can attach it to a support request. ZENITH sends nothing
// anywhere on its own; this output only leaves the machine if the user shares it.
func emitDoctorJSON(checks []doctorCheck) error {
	type jc struct {
		Name    string `json:"name"`
		Status  string `json:"status"`
		Message string `json:"message"`
		Fix     string `json:"fix,omitempty"`
	}
	report := struct {
		Version string `json:"version"`
		OS      string `json:"os"`
		Arch    string `json:"arch"`
		Checks  []jc   `json:"checks"`
	}{Version: version, OS: runtime.GOOS, Arch: runtime.GOARCH}
	failed := 0
	for _, c := range checks {
		st := "ok"
		switch {
		case !c.ok && !c.warn:
			st = "fail"
			failed++
		case c.warn:
			st = "warn"
		}
		report.Checks = append(report.Checks, jc{c.name, st, c.msg, c.fix})
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		return err
	}
	if failed > 0 {
		return fmt.Errorf("%d check(s) failed", failed)
	}
	return nil
}

// indexChecks inspects the index file without loading it: on-disk format,
// segment count, and — for the current format — a full checksum pass over every
// segment, so silent disk corruption is found here rather than mid-search.
func indexChecks(path string) []doctorCheck {
	if _, err := os.Stat(path); err != nil {
		return []doctorCheck{{name: "index", warn: true, msg: "no index yet", fix: "zenith index <directory>"}}
	}
	info, err := index.Inspect(path)
	if err != nil {
		return []doctorCheck{{name: "index", msg: "cannot read " + path + ": " + err.Error(),
			fix: "if this is damage, restore a backup or remove the file and re-index; check the disk"}}
	}
	if info.Version != index.CurrentFormat {
		return []doctorCheck{{name: "index", msg: fmt.Sprintf("%s uses the older on-disk format v%d (current: v%d)", path, info.Version, index.CurrentFormat),
			fix: "zenith migrate     (keeps a backup of the original)"}}
	}
	var size int64
	for _, s := range info.Segments {
		if st, err := os.Stat(s); err == nil {
			size += st.Size()
		}
	}
	out := []doctorCheck{{name: "index", ok: true, msg: fmt.Sprintf("format v%d, %d segment(s), %s, embedder %s", info.Version, len(info.Segments), formatBytes(size), info.Embedder)}}
	if err := index.Verify(path); err != nil {
		out = append(out, doctorCheck{name: "checksums", msg: "index failed verification: " + err.Error(),
			fix: "restore from a backup, or remove the index and re-index"})
	} else {
		out = append(out, doctorCheck{name: "checksums", ok: true, msg: "every segment passes its CRC-32C checks"})
	}
	if len(info.Segments) > 8 {
		out = append(out, doctorCheck{name: "segments", warn: true, msg: fmt.Sprintf("%d segments — reads consult every one", len(info.Segments)),
			fix: "zenith compact"})
	}
	return out
}
