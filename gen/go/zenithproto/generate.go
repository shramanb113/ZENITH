package zenithproto

// Regenerate document.pb.go and document_grpc.pb.go from
// internal/proto/document.proto (run `go generate ./gen/go/zenithproto` from
// the repo root). Equivalent to, from the repo root:
//
//	protoc -I internal/proto --go_out=. --go-grpc_out=. internal/proto/document.proto
//
// The committed files were produced with protoc v6.33.4 (libprotoc 33.4),
// protoc-gen-go v1.36.11 and protoc-gen-go-grpc v1.6.2; use the same versions
// to keep regeneration diffs limited to real schema changes.

//go:generate protoc -I ../../../internal/proto --go_out=../../.. --go-grpc_out=../../.. ../../../internal/proto/document.proto
