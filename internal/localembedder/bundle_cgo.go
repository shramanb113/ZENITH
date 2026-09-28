//go:build cgo

package localembedder

import _ "embed"

// The bundled model and vocabulary are compiled in only when the ONNX runtime
// can actually run them. A build without CGo (lexical-only) stays ~40 MB
// smaller instead of carrying bytes it can never use.

//go:embed assets/model.onnx
var modelBytes []byte

//go:embed assets/vocab.txt
var vocabBytes []byte

// unavailable reports why no embedder can be built in this binary (nil: it can).
func unavailable() error { return nil }
