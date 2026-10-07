package collections

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"
)

const metaFileName = "collection.json"

// Meta is a collection's on-disk metadata, written alongside its index.db.
type Meta struct {
	Version      int       `json:"version"` // always 1
	ID           string    `json:"id"`
	CreatedAt    time.Time `json:"created_at"`
	Embedder     string    `json:"embedder"`   // identity at creation, e.g. "onnx:gte-small", "none:bm25-only"
	KeySHA256    string    `json:"key_sha256"` // hex of sha256(plaintext key)
	MaxDocs      int       `json:"max_docs"`
	MaxBodyBytes int64     `json:"max_body_bytes"`
	DocCount     int       `json:"doc_count"`
	Dirty        bool      `json:"dirty"` // true = a write was in flight; DocCount is an upper bound
}

func readMeta(dir string) (Meta, error) {
	data, err := os.ReadFile(filepath.Join(dir, metaFileName))
	if err != nil {
		return Meta{}, err
	}
	var m Meta
	if err := json.Unmarshal(data, &m); err != nil {
		return Meta{}, fmt.Errorf("collections: parse %s: %w", metaFileName, err)
	}
	return m, nil
}

// writeMeta persists m atomically: write a temp file, fsync it, rename it
// over collection.json, then (except on Windows, where directory fsync is
// unsupported) fsync the directory so the rename itself survives a crash.
func writeMeta(dir string, m Meta) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	final := filepath.Join(dir, metaFileName)
	tmp := final + ".tmp"

	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmp, final); err != nil {
		return err
	}
	if runtime.GOOS == "windows" {
		return nil
	}
	d, err := os.Open(dir)
	if err != nil {
		return nil // best effort: the rename already landed
	}
	defer d.Close()
	_ = d.Sync()
	return nil
}
