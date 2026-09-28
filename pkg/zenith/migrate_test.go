package zenith_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/shramanb113/ZENITH/pkg/zenith"
)

// legacyHeader is the start of a version-5 gob index file: magic, version,
// embedder name, dims. Only the header decides "this is an old format".
func legacyHeader(version uint16) []byte {
	var b bytes.Buffer
	b.WriteString("ZNTH")
	var v [2]byte
	binary.BigEndian.PutUint16(v[:], version)
	b.Write(v[:])
	if version >= 5 {
		name := "onnx:all-MiniLM-L6-v2"
		var l [4]byte
		binary.BigEndian.PutUint32(l[:], uint32(len(name)))
		b.Write(l[:])
		b.WriteString(name)
		var d [4]byte
		binary.BigEndian.PutUint32(d[:], 384)
		b.Write(d[:])
	}
	b.WriteString("gob body would follow")
	return b.Bytes()
}

// Opening an old-format file must fail with a clear, typed error and must not
// touch the file — Open followed by Close would otherwise checkpoint an empty
// index over the user's data.
func TestOpen_LegacyFormatIsRefusedAndLeftIntact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.db")
	orig := legacyHeader(5)
	if err := os.WriteFile(path, orig, 0o644); err != nil {
		t.Fatal(err)
	}
	db, err := zenith.Open(path, zenith.WithBM25Only())
	if !errors.Is(err, zenith.ErrIncompatibleVersion) {
		if db != nil {
			db.Close()
		}
		t.Fatalf("Open err = %v, want ErrIncompatibleVersion", err)
	}
	after, _ := os.ReadFile(path)
	if !bytes.Equal(after, orig) {
		t.Fatal("Open modified an old-format index file it refused to read")
	}
	if _, err := os.Stat(path + ".lock"); err == nil {
		t.Fatal("lock file left behind after refused Open")
	}
}

func TestMigrate_RejectsCurrentFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cur.db")
	db, err := zenith.Open(path, zenith.WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Add(context.Background(), "a", "hello world"); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := zenith.Migrate(path, ""); err == nil {
		t.Fatal("Migrate accepted a current-format index")
	}
}
