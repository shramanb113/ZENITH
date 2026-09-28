//go:build cgo && windows

package localembedder

import _ "embed"

//go:embed assets/onnxruntime.dll
var ortLibBytes []byte

const ortLibFilename = "zenith-ort-*.dll"
