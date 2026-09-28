//go:build !cgo

package localembedder

// Without CGo there is no ONNX runtime, so the model is not bundled at all.
var (
	modelBytes []byte
	vocabBytes []byte
	// ortLibBytes/ortLibFilename exist so shared code compiles; never used.
	ortLibBytes    []byte
	ortLibFilename = "zenith-ort-*"
)

// unavailable reports why no embedder can be built in this binary. Checked
// first so the caller sees "CGo required", not a tokenizer error about the
// model files that were deliberately left out of this build.
func unavailable() error { return errNoCGo }
