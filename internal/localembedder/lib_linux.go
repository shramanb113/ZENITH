//go:build cgo && linux

package localembedder

import _ "embed"

//go:embed assets/libonnxruntime.so
var ortLibBytes []byte

const ortLibFilename = "zenith-ort-*.so"
