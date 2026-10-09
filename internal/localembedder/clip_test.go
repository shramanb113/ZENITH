package localembedder

import (
	"testing"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

func TestNewCLIPByID_NotInstalled(t *testing.T) {
	_, err := NewCLIPByID("clip-vit-base-patch32", t.TempDir())
	if err == nil {
		t.Fatal("NewCLIPByID should fail when the model files are not present")
	}
}

func TestNewCLIPByID_UnknownModel(t *testing.T) {
	_, err := NewCLIPByID("not-a-real-visual-model", t.TempDir())
	if err == nil {
		t.Fatal("NewCLIPByID should fail for an unregistered model id")
	}
}

func TestCLIPEmbedder_SatisfiesVisualEmbedder(t *testing.T) {
	var _ embedding.VisualEmbedder = (*CLIPEmbedder)(nil)
	var _ embedding.Named = (*CLIPEmbedder)(nil)
}
