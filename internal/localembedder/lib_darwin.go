//go:build cgo && darwin

package localembedder

import _ "embed"

//go:embed assets/libonnxruntime.dylib
var ortLibBytes []byte

const ortLibFilename = "zenith-ort-*.dylib"
