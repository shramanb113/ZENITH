package zenith_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/localembedder"
	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// An explicitly requested model that cannot be loaded is an error from Open —
// never a silent fall back to lexical-only or to a different model, which
// would produce an index whose vectors the caller did not ask for.
func TestOpen_WithModel_UnknownAndNotInstalledFailLoudly(t *testing.T) {
	if _, err := zenith.Open(":memory:", zenith.WithModel("no-such-model")); err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("unknown model: err = %v", err)
	}

	other := ""
	for _, m := range localembedder.Models() {
		if !strings.EqualFold(m.ID, localembedder.BundledID()) {
			other = m.ID
			break
		}
	}
	if other == "" {
		t.Skip("registry has only the bundled model")
	}
	_, err := zenith.Open(":memory:", zenith.WithModel(other), zenith.WithModelsDir(t.TempDir()))
	if err == nil || !strings.Contains(err.Error(), "zenith models pull "+other) {
		t.Fatalf("model %q not installed: err = %v, want a hint to run `zenith models pull`", other, err)
	}
}

func TestOpen_WithModel_RejectsEmpty(t *testing.T) {
	if _, err := zenith.Open(":memory:", zenith.WithModel("  ")); !errors.Is(err, zenith.ErrInvalidOption) {
		t.Fatalf("empty model id: err = %v, want ErrInvalidOption", err)
	}
	if _, err := zenith.Open(":memory:", zenith.WithModelsDir("")); !errors.Is(err, zenith.ErrInvalidOption) {
		t.Fatalf("empty models dir: err = %v, want ErrInvalidOption", err)
	}
}

// WithBM25Only wins over WithModel: lexical-only mode must never touch a model.
func TestOpen_WithModel_IgnoredInBM25Only(t *testing.T) {
	db, err := zenith.Open(":memory:", zenith.WithBM25Only(), zenith.WithModel("no-such-model"))
	if err != nil {
		t.Fatalf("BM25-only should not resolve a model: %v", err)
	}
	db.Close()
}
