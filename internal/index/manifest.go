package index

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"log/slog"
	"os"
	pathutil "path/filepath"
	"strings"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// The index "file" a caller names (e.g. zenith.db) is a small manifest, not the
// data. It records which segment files make up the index, in order. Segments
// live next to it as "<manifest>.seg-NNNNNN". Committing a flush or a
// compaction means (1) fully writing and fsyncing the new segment, then (2)
// atomically replacing the manifest (temp file + rename). A crash before (2)
// leaves the old manifest — and so the old, complete index — in place; the
// stray segment file is garbage-collected on the next open.
//
// Manifest layout (format version 6; multi-byte fields big-endian, matching the
// version 3–5 header so tools can read the version and embedder of any file):
//
//	"ZNTH" | version u16 | embedder-name (u32 len + bytes) | dims u32 |
//	body length u32 | JSON body | CRC-32C of body u32
const manifestVersion uint16 = 6

// CurrentFormat is the on-disk format version this build reads and writes.
const CurrentFormat = manifestVersion

// LegacyFormatError reports a gob-format index file (format versions 3–5) that
// the current engine cannot open directly. `zenith migrate` converts it.
type LegacyFormatError struct {
	Version uint16
}

func (e *LegacyFormatError) Error() string {
	return fmt.Sprintf("index: file is format version %d (the current format is %d) — convert it with: zenith migrate", e.Version, manifestVersion)
}

func (e *LegacyFormatError) Is(target error) bool { return target == ErrIncompatibleVersion }

type manifestSegment struct {
	File  string `json:"file"` // base name, relative to the manifest's directory
	Gen   uint64 `json:"gen"`
	Docs  int    `json:"docs"`
	Bytes int64  `json:"bytes"`
}

type manifestBody struct {
	Generation uint64            `json:"generation"` // bumped on every commit
	NextGen    uint64            `json:"next_gen"`   // next segment file number to allocate
	Segments   []manifestSegment `json:"segments"`   // oldest first
}

// FileInfo describes an index file's header without loading it.
type FileInfo struct {
	Version  uint16
	Embedder string // "" for pre-v5 files, which did not record one
	Dims     int
	Segments []string // segment file paths, v6 only
	Docs     int      // live-or-dead rows across segments, v6 only
}

var crcTable = crc32.MakeTable(crc32.Castagnoli)

func segmentFileName(manifestPath string, gen uint64) string {
	return fmt.Sprintf("%s.seg-%06d", manifestPath, gen)
}

func writeManifest(path, embName string, dims int, m manifestBody) error {
	body, err := json.Marshal(m)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := fsx.Create(tmp)
	if err != nil {
		return err
	}
	fail := func(err error) error { f.Close(); fsx.Remove(tmp); return err }

	w := bufio.NewWriter(f)
	w.WriteString("ZNTH")
	var b2 [2]byte
	binary.BigEndian.PutUint16(b2[:], manifestVersion)
	w.Write(b2[:])
	if err := writeHeaderString(w, embName); err != nil {
		return fail(err)
	}
	var b4 [4]byte
	binary.BigEndian.PutUint32(b4[:], uint32(dims))
	w.Write(b4[:])
	binary.BigEndian.PutUint32(b4[:], uint32(len(body)))
	w.Write(b4[:])
	w.Write(body)
	binary.BigEndian.PutUint32(b4[:], crc32.Checksum(body, crcTable))
	w.Write(b4[:])
	if err := w.Flush(); err != nil {
		return fail(err)
	}
	if err := f.Sync(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		fsx.Remove(tmp)
		return err
	}
	if err := fsx.Rename(tmp, path); err != nil {
		fsx.Remove(tmp)
		return err
	}
	syncDir(pathutil.Dir(path))
	return nil
}

// syncDir makes a rename durable where the platform supports it (best effort:
// Windows cannot fsync a directory handle).
func syncDir(dir string) {
	if err := fsx.SyncDir(dir); err != nil {
		slog.Debug("index: directory fsync failed", "error", err)
	}
}

// readHeader reads magic, version and (for v5+) the embedder identity.
func readHeader(f io.Reader) (version uint16, name string, dims int, err error) {
	var magic [4]byte
	if _, err = io.ReadFull(f, magic[:]); err != nil {
		return 0, "", 0, fmt.Errorf("index: failed to read file header: %w", err)
	}
	if magic != saveFormatMagic {
		return 0, "", 0, fmt.Errorf("index: not a ZENITH index file (bad magic bytes)")
	}
	var vbuf [2]byte
	if _, err = io.ReadFull(f, vbuf[:]); err != nil {
		return 0, "", 0, fmt.Errorf("index: failed to read version: %w", err)
	}
	version = binary.BigEndian.Uint16(vbuf[:])
	if version < 5 {
		return version, "", 0, nil
	}
	if name, err = readHeaderString(f); err != nil {
		return version, "", 0, fmt.Errorf("index: failed to read embedder header: %w", err)
	}
	var d [4]byte
	if _, err = io.ReadFull(f, d[:]); err != nil {
		return version, "", 0, fmt.Errorf("index: failed to read embedder dimensions: %w", err)
	}
	return version, name, int(binary.BigEndian.Uint32(d[:])), nil
}

// readManifest parses a v6 manifest. Older formats yield a *LegacyFormatError.
func readManifest(path string) (name string, dims int, m manifestBody, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, m, err
	}
	defer f.Close()
	r := bufio.NewReader(f)
	version, name, dims, err := readHeader(r)
	if err != nil {
		return "", 0, m, err
	}
	if version != manifestVersion {
		if version >= 3 && version < manifestVersion {
			return name, dims, m, &LegacyFormatError{Version: version}
		}
		return name, dims, m, ErrIncompatibleVersion
	}
	var b4 [4]byte
	if _, err = io.ReadFull(r, b4[:]); err != nil {
		return "", 0, m, fmt.Errorf("index: manifest truncated: %w", err)
	}
	n := binary.BigEndian.Uint32(b4[:])
	if n > 64<<20 {
		return "", 0, m, fmt.Errorf("index: manifest is corrupt (body length %d)", n)
	}
	body := make([]byte, n)
	if _, err = io.ReadFull(r, body); err != nil {
		return "", 0, m, fmt.Errorf("index: manifest truncated: %w", err)
	}
	if _, err = io.ReadFull(r, b4[:]); err != nil {
		return "", 0, m, fmt.Errorf("index: manifest truncated: %w", err)
	}
	if binary.BigEndian.Uint32(b4[:]) != crc32.Checksum(body, crcTable) {
		return "", 0, m, errors.New("index: manifest checksum mismatch — the file is corrupt")
	}
	if err = json.Unmarshal(body, &m); err != nil {
		return "", 0, m, fmt.Errorf("index: manifest is corrupt: %w", err)
	}
	return name, dims, m, nil
}

// Inspect reads an index file's header (and, for v6, its segment list) without
// loading any data. Used by `zenith doctor` and `zenith migrate`.
func Inspect(path string) (FileInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return FileInfo{}, err
	}
	version, name, dims, err := readHeader(bufio.NewReader(f))
	f.Close()
	if err != nil {
		return FileInfo{}, err
	}
	info := FileInfo{Version: version, Embedder: name, Dims: dims}
	if version == manifestVersion {
		_, _, m, err := readManifest(path)
		if err != nil {
			return info, err
		}
		for _, s := range m.Segments {
			info.Segments = append(info.Segments, pathutil.Join(pathutil.Dir(path), s.File))
			info.Docs += s.Docs
		}
	}
	return info, nil
}

// gcSegments deletes segment files next to the manifest that it does not list:
// leftovers of a flush or compaction that crashed before committing, or of
// segments a completed compaction superseded. Also removes stray temp files.
func gcSegments(path string, m manifestBody) {
	dir, base := pathutil.Split(path)
	if dir == "" {
		dir = "."
	}
	keep := map[string]bool{}
	for _, s := range m.Segments {
		keep[s.File] = true
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	for _, en := range entries {
		name := en.Name()
		if !strings.HasPrefix(name, base+".seg-") && name != base+".tmp" {
			continue
		}
		if keep[name] {
			continue
		}
		if err := fsx.Remove(pathutil.Join(dir, name)); err == nil {
			slog.Info("index: removed orphaned file", "file", name)
		}
	}
}
