package localembedder

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/modelspec"
)

// The registry is what index-file identities are derived from ("onnx:<ID>"), so
// its invariants are worth pinning: an ID must never silently change meaning.
func TestRegistry_Invariants(t *testing.T) {
	ms := Models()
	if len(ms) < 2 {
		t.Fatalf("registry has %d models", len(ms))
	}
	if !sort.SliceIsSorted(ms, func(i, j int) bool { return ms[i].ID < ms[j].ID }) {
		t.Error("Models() is not sorted by ID")
	}
	seen := map[string]bool{}
	for _, m := range ms {
		if seen[strings.ToLower(m.ID)] {
			t.Errorf("duplicate model ID %q", m.ID)
		}
		seen[strings.ToLower(m.ID)] = true
		if m.Dims <= 0 || m.SizeMB <= 0 || m.Description == "" || m.Languages == "" {
			t.Errorf("%s: incomplete spec %+v", m.ID, m)
		}
		if !strings.HasPrefix(m.ModelURL, "https://") {
			t.Errorf("%s: model URL %q must be https", m.ID, m.ModelURL)
		}
		if m.IndexName() != "onnx:"+m.ID {
			t.Errorf("%s: index identity %q", m.ID, m.IndexName())
		}
	}
	if _, err := Lookup(modelspec.DefaultID); err != nil {
		t.Errorf("DefaultID %q is not in the registry: %v", modelspec.DefaultID, err)
	}
	// The bundled model (assets/model.id) must be a registered model, or a binary
	// would record an identity nothing can reproduce.
	if _, err := Lookup(BundledID()); err != nil {
		t.Errorf("assets/model.id names %q, which is not registered: %v", BundledID(), err)
	}
}

func TestLookup_CaseInsensitiveAndUnknown(t *testing.T) {
	if s, err := Lookup("GTE-SMALL"); err != nil || s.ID != "gte-small" {
		t.Fatalf("Lookup is not case-insensitive: %+v, %v", s, err)
	}
	_, err := Lookup("no-such-model")
	if err == nil || !strings.Contains(err.Error(), "available:") {
		t.Fatalf("unknown model error should list the available ones, got %v", err)
	}
}

func TestNewByID_NotInstalledPointsAtPull(t *testing.T) {
	other := "gte-small"
	if strings.EqualFold(BundledID(), other) {
		other = "bge-small-en-v1.5"
	}
	_, err := NewByID(other, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "zenith models pull "+other) {
		t.Fatalf("a model that is not installed must say how to install it, got %v", err)
	}
	// A models dir that has the id directory but no model file is the same failure.
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, other), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := NewByID(other, dir); err == nil {
		t.Fatal("expected an error for a model directory without model.onnx")
	}
}
