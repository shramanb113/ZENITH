// Package segment implements ZENITH's immutable, memory-mapped index segment
// file: the on-disk unit the engine flushes to and serves reads from.
//
// A segment is written once (never modified) and opened by mapping the file
// read-only, so opening is O(sections) regardless of index size and the
// documents, vectors and posting lists live in the OS page cache instead of
// the Go heap. All integers are little-endian; every section is 8-byte
// aligned so fixed-width arrays can be viewed in place without copying.
//
// Layout:
//
//	[header 64B][section]...[section][directory][footer 16B]
//
// The footer locates the directory; the directory lists each section's kind,
// offset, length and CRC-32C; the footer also carries a CRC over the header
// and directory. Open verifies that CRC and the section bounds; Verify checks
// every section's CRC (slower — it reads the whole file).
package segment

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
)

const (
	magic        = "ZNSG"
	footerMagic  = "ZSGE"
	headerSize   = 64
	footerSize   = 16
	dirEntrySize = 24

	// FormatVersion is the segment layout version. It is independent of the
	// index-directory manifest version; bump it on any incompatible layout change.
	FormatVersion = 1
)

// Section kinds. Values are part of the file format — never renumber.
const (
	secDocIDs     uint32 = 1 // u64[N] sorted ascending internal doc IDs
	secDocOffs    uint32 = 2 // u64[N+1] offsets into secDocBlob
	secDocBlob    uint32 = 3 // per doc: orig, text (each uvarint length + bytes)
	secDocLen     uint32 = 4 // u32[N] BM25 document length (token count)
	secVecIdx     uint32 = 5 // i32[N] row in secVectors, or -1 for no vector
	secVectors    uint32 = 6 // u16[V*dims] float16 bits
	secTermOffs   uint32 = 7 // u64[T+1]
	secTermData   uint32 = 8 // per term: key, cf, df, postings(doc,tf pairs)
	secFragOffs   uint32 = 9
	secFragData   uint32 = 10 // per fragment: key, postings(doc deltas)
	secPhonOffs   uint32 = 11
	secPhonData   uint32 = 12
	secFwdOffs    uint32 = 13 // u64[N+1]
	secFwdData    uint32 = 14 // per doc: n, (term idx delta, tf) pairs
	secWordOffs   uint32 = 15 // u64[W+1]
	secWordData   uint32 = 16 // per word: key
	secWordVecs   uint32 = 17 // u16[W*dims]
	secDels       uint32 = 18 // u64[K] doc IDs this segment deletes from older segments
	secMeta       uint32 = 19 // opaque bytes (JSON)
	secPostBlocks uint32 = 20 // reserved
	secAttrRows   uint32 = 21 // u32[K] rows that carry attributes, ascending
	secAttrOffs   uint32 = 22 // u64[K+1] offsets into secAttrBlob
	secAttrBlob   uint32 = 23 // encoded attributes, one blob per row in secAttrRows
)

// ErrCorrupt is returned when a segment fails structural or checksum validation.
var ErrCorrupt = errors.New("segment: file is corrupt or truncated")

var crcTable = crc32.MakeTable(crc32.Castagnoli)

type dirEntry struct {
	kind uint32
	off  uint64
	n    uint64
	crc  uint32
}

type header struct {
	dims     uint32
	numDocs  uint32
	numVecs  uint32
	numTerms uint32
	numFrags uint32
	numPhon  uint32
	numWords uint32
	numDels  uint32
}

func (h header) encode() []byte {
	b := make([]byte, headerSize)
	copy(b, magic)
	binary.LittleEndian.PutUint16(b[4:], FormatVersion)
	// b[6:8] flags, reserved
	binary.LittleEndian.PutUint32(b[8:], h.dims)
	binary.LittleEndian.PutUint32(b[12:], h.numDocs)
	binary.LittleEndian.PutUint32(b[16:], h.numVecs)
	binary.LittleEndian.PutUint32(b[20:], h.numTerms)
	binary.LittleEndian.PutUint32(b[24:], h.numFrags)
	binary.LittleEndian.PutUint32(b[28:], h.numPhon)
	binary.LittleEndian.PutUint32(b[32:], h.numWords)
	binary.LittleEndian.PutUint32(b[36:], h.numDels)
	return b
}

func decodeHeader(b []byte) (header, error) {
	if len(b) < headerSize || string(b[:4]) != magic {
		return header{}, fmt.Errorf("%w: bad magic", ErrCorrupt)
	}
	if v := binary.LittleEndian.Uint16(b[4:]); v != FormatVersion {
		return header{}, fmt.Errorf("segment: unsupported layout version %d (this build reads %d)", v, FormatVersion)
	}
	return header{
		dims:     binary.LittleEndian.Uint32(b[8:]),
		numDocs:  binary.LittleEndian.Uint32(b[12:]),
		numVecs:  binary.LittleEndian.Uint32(b[16:]),
		numTerms: binary.LittleEndian.Uint32(b[20:]),
		numFrags: binary.LittleEndian.Uint32(b[24:]),
		numPhon:  binary.LittleEndian.Uint32(b[28:]),
		numWords: binary.LittleEndian.Uint32(b[32:]),
		numDels:  binary.LittleEndian.Uint32(b[36:]),
	}, nil
}
