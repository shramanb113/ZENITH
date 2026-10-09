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

var modelsListFlags struct {
	rerankers bool
}

var modelsListCmd = &cobra.Command{
	Use:   "list",
	Short: "Show the models ZENITH can run and which are installed",
	RunE: func(cmd *cobra.Command, args []string) error {
		if modelsListFlags.rerankers {
			printHeader("models", "cross-encoder reranker registry")
			for _, m := range localembedder.RerankerModels() {
				status := muted("not installed  (zenith models pull " + m.ID + ")")
				if fileExists(filepath.Join(modelsDir(), m.ID, "model.onnx")) {
					status = green("installed")
				}
				fmt.Printf("  %-24s ~%3d MB  %s\n", m.ID, m.SizeMB, status)
				fmt.Printf("  %-24s %s\n", "", muted(m.Description))
			}
			printDivider()
			printFooter("use one with", "--rerank --rerank-model <id>")
			return nil
		}

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
		printFooter("use one with", "--model <id>  (see also: zenith models list --rerankers)")

		printDivider()
		printHeader("models", "visual (image+text) embedding model registry")
		for _, m := range localembedder.VisualModels() {
			status := muted("not installed  (zenith models pull " + m.ID + ")")
			if fileExists(filepath.Join(modelsDir(), m.ID, "vision_model.onnx")) {
				status = green("installed")
			}
			fmt.Printf("  %-24s %3d dims  ~%3d MB  %s\n", m.ID, m.Dims, m.SizeMB, status)
			fmt.Printf("  %-24s %s\n", "", muted(m.Description))
		}
		printDivider()
		printFooter("visual search wiring is not yet available", "see the visual-image-search design spec")
		return nil
	},
}

var modelsPullCmd = &cobra.Command{
	Use:   "pull <id>",
	Short: "Download a model from the registry into ~/.zenith/models",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		id, sizeMB, url, vocabURL, denseURL, usage, isBundled := args[0], 0, "", "", "", "", false
		var visualSpec localembedder.VisualSpec
		isVisual := false
		if spec, err := localembedder.Lookup(args[0]); err == nil {
			id, sizeMB, url, vocabURL, denseURL = spec.ID, spec.SizeMB, spec.ModelURL, spec.VocabURL, spec.DenseURL
			isBundled = strings.EqualFold(spec.ID, localembedder.BundledID())
			usage = "use with --model " + spec.ID + " (needs a fresh index: vectors differ per model)"
		} else if rspec, rerr := localembedder.LookupReranker(args[0]); rerr == nil {
			id, sizeMB, url = rspec.ID, rspec.SizeMB, rspec.ModelURL
			usage = "use with --rerank --rerank-model " + rspec.ID
		} else if vspec, verr := localembedder.LookupVisual(args[0]); verr == nil {
			id, sizeMB, visualSpec, isVisual = vspec.ID, vspec.SizeMB, vspec, true
			usage = "visual search wiring is not yet available (see the visual-image-search design spec)"
		} else {
			return err
		}
		if isBundled {
			fmt.Printf("%s is already bundled in this binary — nothing to download.\n", id)
			return nil
		}
		dir := filepath.Join(modelsDir(), id)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		printHeader("models pull", fmt.Sprintf("%s (~%d MB)", id, sizeMB))

		if isVisual {
			if err := pullVisualFiles(visualSpec, dir); err != nil {
				return err
			}
			printDivider()
			printFooter("installed", usage)
			return nil
		}

		sum, n, err := downloadModel(url, filepath.Join(dir, "model.onnx"))
		if err != nil {
			return fmt.Errorf("download failed: %w", err)
		}
		// The registry does not pin checksums (it names upstream files that can be
		// re-exported), so record what was actually fetched next to it: a later
		// `sha256sum` or support request can tell exactly which bytes are in use.
		_ = os.WriteFile(filepath.Join(dir, "model.sha256"), []byte(sum+"  model.onnx\n"), 0o644)
		fmt.Printf("  %s  %s (%s)\n", green("✓"), filepath.Join(dir, "model.onnx"), formatBytes(n))
		fmt.Printf("  %s  sha256 %s\n", muted("·"), sum)
		if vocabURL != "" {
			vsum, vn, verr := downloadModel(vocabURL, filepath.Join(dir, "vocab.txt"))
			if verr != nil {
				return fmt.Errorf("model downloaded but vocab download failed: %w", verr)
			}
			fmt.Printf("  %s  %s (%s)\n", green("✓"), filepath.Join(dir, "vocab.txt"), formatBytes(vn))
			fmt.Printf("  %s  sha256 %s\n", muted("·"), vsum)
		}
		if denseURL != "" {
			dsum, dn, derr := downloadModel(denseURL, filepath.Join(dir, "dense.safetensors"))
			if derr != nil {
				return fmt.Errorf("model downloaded but dense projection download failed: %w", derr)
			}
			fmt.Printf("  %s  %s (%s)\n", green("✓"), filepath.Join(dir, "dense.safetensors"), formatBytes(dn))
			fmt.Printf("  %s  sha256 %s\n", muted("·"), dsum)
		}
		printDivider()
		printFooter("installed", usage)
		return nil
	},
}

func init() {
	modelsListCmd.Flags().BoolVar(&modelsListFlags.rerankers, "rerankers", false,
		"List cross-encoder rerankers instead of embedding models")
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

// pullVisualFiles downloads a visual model's four files (vision tower, text
// tower, vocab, merges) into dir, using the exact filenames NewCLIPByID
// expects. Separated from modelsPullCmd.RunE so it is testable against a
// fake VisualSpec pointing at a local httptest.Server, instead of the real
// registry's huggingface.co URLs.
func pullVisualFiles(spec localembedder.VisualSpec, dir string) error {
	files := []struct{ url, name string }{
		{spec.VisionModelURL, "vision_model.onnx"},
		{spec.TextModelURL, "text_model.onnx"},
		{spec.VocabURL, "vocab.json"},
		{spec.MergesURL, "merges.txt"},
	}
	for _, f := range files {
		sum, n, err := downloadModel(f.url, filepath.Join(dir, f.name))
		if err != nil {
			return fmt.Errorf("download failed for %s: %w", f.name, err)
		}
		fmt.Printf("  %s  %s (%s)\n", green("✓"), filepath.Join(dir, f.name), formatBytes(n))
		fmt.Printf("  %s  sha256 %s\n", muted("·"), sum)
	}
	return nil
}
