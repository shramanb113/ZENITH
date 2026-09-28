package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/shramanb113/ZENITH/internal/localembedder"
)

// modelsDir is where `zenith models pull` installs non-bundled models.
func modelsDir() string { return zenithDataPath("models") }

var modelsCmd = &cobra.Command{
	Use:   "models",
	Short: "List and install embedding models",
	Long: `ZENITH ships one embedding model inside the binary. Others from the registry can
be installed here and selected with --model.

An index records which model produced its vectors and refuses to open with a
different one, so switching models means re-indexing (your documents are not
touched; only the search index is rebuilt).`,
}

var modelsListCmd = &cobra.Command{
	Use:   "list",
	Short: "Show the models ZENITH can run and which are installed",
	RunE: func(cmd *cobra.Command, args []string) error {
		printHeader("models", "embedding model registry")
		bundled := localembedder.BundledID()
		for _, m := range localembedder.Models() {
			status := muted("not installed  (zenith models pull " + m.ID + ")")
			switch {
			case strings.EqualFold(m.ID, bundled):
				status = green("bundled in this binary")
			case fileExists(filepath.Join(modelsDir(), m.ID, "model.onnx")):
				status = green("installed")
			}
			fmt.Printf("  %-20s %3d dims  ~%3d MB  %s\n", m.ID, m.Dims, m.SizeMB, status)
			fmt.Printf("  %-20s %s\n", "", muted(m.Description+" ["+m.Languages+"]"))
		}
		printDivider()
		printFooter("use one with", "--model <id>")
		return nil
	},
}

var modelsPullCmd = &cobra.Command{
	Use:   "pull <id>",
	Short: "Download a model from the registry into ~/.zenith/models",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		spec, err := localembedder.Lookup(args[0])
		if err != nil {
			return err
		}
		if strings.EqualFold(spec.ID, localembedder.BundledID()) {
			fmt.Printf("%s is already bundled in this binary — nothing to download.\n", spec.ID)
			return nil
		}
		dir := filepath.Join(modelsDir(), spec.ID)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		printHeader("models pull", fmt.Sprintf("%s (~%d MB)", spec.ID, spec.SizeMB))
		sum, n, err := downloadModel(spec.ModelURL, filepath.Join(dir, "model.onnx"))
		if err != nil {
			return fmt.Errorf("download failed: %w", err)
		}
		// The registry does not pin checksums (it names upstream files that can be
		// re-exported), so record what was actually fetched next to it: a later
		// `sha256sum` or support request can tell exactly which bytes are in use.
		_ = os.WriteFile(filepath.Join(dir, "model.sha256"), []byte(sum+"  model.onnx\n"), 0o644)
		fmt.Printf("  %s  %s (%s)\n", green("✓"), filepath.Join(dir, "model.onnx"), formatBytes(n))
		fmt.Printf("  %s  sha256 %s\n", muted("·"), sum)
		printDivider()
		printFooter("installed", "use with --model "+spec.ID+" (needs a fresh index: vectors differ per model)")
		return nil
	},
}

func init() {
	modelsCmd.AddCommand(modelsListCmd, modelsPullCmd)
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// downloadModel fetches url to dest atomically (temp file + rename), returning
// the SHA-256 and size of what was written.
func downloadModel(url, dest string) (sum string, n int64, err error) {
	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Get(url)
	if err != nil {
		return "", 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", 0, fmt.Errorf("HTTP %d from %s", resp.StatusCode, url)
	}
	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return "", 0, err
	}
	h := sha256.New()
	n, err = io.Copy(io.MultiWriter(f, h), resp.Body)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return "", 0, err
	}
	if n == 0 {
		os.Remove(tmp)
		return "", 0, fmt.Errorf("empty response from %s", url)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}
