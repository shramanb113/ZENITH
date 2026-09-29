package segment

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"sort"
	"unsafe"
)

// Segment is an opened, immutable segment. Every accessor is safe for
// concurrent use. Byte slices and []uint16 vectors returned by accessors whose
// doc says "aliases the mapping" are views into the memory map: they are valid
// only until Close, must not be modified, and must not be retained across a
// Close. Accessors that return string copy their bytes.
type Segment struct {
	path string
	mm   *mapping
	data []byte
	hdr  header
	dir  map[uint32][]byte // section kind -> bytes (aliases data)

	docIDs   []uint64
	docOffs  []uint64
	docBlob  []byte
	attrRows []uint32
	attrOffs []uint64
	attrBlob []byte
	docLen   []uint32
	vecIdx   []uint32
	vectors  []uint16
	terms    dict
	frags    dict
	phon     dict
	fwdOffs  []uint64
	fwdData  []byte
	words    dict
	wordVecs []uint16
	dels     []uint64
	meta     []byte
}

// dict is a sorted table of variable-length entries: offs[i]..offs[i+1]
// delimits entry i inside data, and every entry starts with uvarint key length
// followed by the key bytes.
type dict struct {
	offs []uint64
	data []byte
}

func (d dict) n() int {
	if len(d.offs) == 0 {
		return 0
	}
	return len(d.offs) - 1
}

// entry returns the key and the remaining payload bytes of entry i.
func (d dict) entry(i int) (key, payload []byte) {
	e := d.data[d.offs[i]:d.offs[i+1]]
	klen, n := binary.Uvarint(e)
	return e[n : n+int(klen)], e[n+int(klen):]
}

func (d dict) key(i int) []byte {
	k, _ := d.entry(i)
	return k
}

// find returns the index of key, or -1.
func (d dict) find(key string) int {
	n := d.n()
	i := sort.Search(n, func(i int) bool { return bytes.Compare(d.key(i), unsafeBytes(key)) >= 0 })
	if i < n && bytes.Equal(d.key(i), unsafeBytes(key)) {
		return i
	}
	return -1
}

// lowerBound returns the first index whose key is >= key.
func (d dict) lowerBound(key string) int {
	return sort.Search(d.n(), func(i int) bool { return bytes.Compare(d.key(i), unsafeBytes(key)) >= 0 })
}

func unsafeBytes(s string) []byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Slice(unsafe.StringData(s), len(s))
}

// Open maps the segment file at path and validates its structure. It does not
// read every byte; call Verify for a full checksum pass.
func Open(path string) (*Segment, error) {
	mm, err := mapFile(path)
	if err != nil {
		return nil, err
	}
	s, err := fromBytes(mm.data)
	if err != nil {
		mm.close()
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	s.path = path
	s.mm = mm
	return s, nil
}

func fromBytes(data []byte) (*Segment, error) {
	if !littleEndianHost() {
		return nil, fmt.Errorf("segment: big-endian hosts are not supported")
	}
	if len(data) < headerSize+footerSize {
		return nil, fmt.Errorf("%w: file too small", ErrCorrupt)
	}
	foot := data[len(data)-footerSize:]
	if string(foot[12:]) != footerMagic {
		return nil, fmt.Errorf("%w: missing footer", ErrCorrupt)
	}
	dirOff := binary.LittleEndian.Uint64(foot[0:])
	wantCRC := binary.LittleEndian.Uint32(foot[8:])
	dirEnd := uint64(len(data) - footerSize)
	if dirOff < headerSize || dirOff > dirEnd || (dirEnd-dirOff)%dirEntrySize != 0 {
		return nil, fmt.Errorf("%w: bad directory offset", ErrCorrupt)
	}
	dirBytes := data[dirOff:dirEnd]
	gotCRC := crc32.Update(crc32.Update(0, crcTable, data[:headerSize]), crcTable, dirBytes)
	if gotCRC != wantCRC {
		return nil, fmt.Errorf("%w: header/directory checksum mismatch", ErrCorrupt)
	}
	hdr, err := decodeHeader(data[:headerSize])
	if err != nil {
		return nil, err
	}

	s := &Segment{data: data, hdr: hdr, dir: map[uint32][]byte{}}
	for off := 0; off < len(dirBytes); off += dirEntrySize {
		kind := binary.LittleEndian.Uint32(dirBytes[off:])
		start := binary.LittleEndian.Uint64(dirBytes[off+4:])
		n := binary.LittleEndian.Uint64(dirBytes[off+12:])
		if start > dirOff || n > dirOff-start {
			return nil, fmt.Errorf("%w: section %d out of bounds", ErrCorrupt, kind)
		}
		s.dir[kind] = data[start : start+n]
	}

	nDocs := int(hdr.numDocs)
	dims := int(hdr.dims)
	s.docIDs = viewU64(s.dir[secDocIDs])
	s.docOffs = viewU64(s.dir[secDocOffs])
	s.docBlob = s.dir[secDocBlob]
	s.attrRows = viewU32(s.dir[secAttrRows])
	s.attrOffs = viewU64(s.dir[secAttrOffs])
	s.attrBlob = s.dir[secAttrBlob]
	s.docLen = viewU32(s.dir[secDocLen])
	s.vecIdx = viewU32(s.dir[secVecIdx])
	s.vectors = viewU16(s.dir[secVectors])
	s.terms = dict{offs: viewU64(s.dir[secTermOffs]), data: s.dir[secTermData]}
	s.frags = dict{offs: viewU64(s.dir[secFragOffs]), data: s.dir[secFragData]}
	s.phon = dict{offs: viewU64(s.dir[secPhonOffs]), data: s.dir[secPhonData]}
	s.fwdOffs = viewU64(s.dir[secFwdOffs])
	s.fwdData = s.dir[secFwdData]
	s.words = dict{offs: viewU64(s.dir[secWordOffs]), data: s.dir[secWordData]}
	s.wordVecs = viewU16(s.dir[secWordVecs])
	s.dels = viewU64(s.dir[secDels])
	s.meta = s.dir[secMeta]

	check := func(name string, ok bool) error {
		if !ok {
			return fmt.Errorf("%w: %s section has the wrong size", ErrCorrupt, name)
		}
		return nil
	}
	for _, c := range []struct {
		name string
		ok   bool
	}{
		{"doc ids", len(s.docIDs) == nDocs},
		{"doc offsets", len(s.docOffs) == nDocs+1},
		{"attributes", len(s.attrOffs) == len(s.attrRows)+1},
		{"doc lengths", len(s.docLen) == nDocs},
		{"vector index", len(s.vecIdx) == nDocs},
		{"vectors", len(s.vectors) == int(hdr.numVecs)*dims},
		{"term offsets", s.terms.n() == int(hdr.numTerms)},
		{"fragment offsets", s.frags.n() == int(hdr.numFrags)},
		{"phonetic offsets", s.phon.n() == int(hdr.numPhon)},
		{"forward offsets", len(s.fwdOffs) == nDocs+1},
		{"word offsets", s.words.n() == int(hdr.numWords)},
		{"word vectors", len(s.wordVecs) == int(hdr.numWords)*dims},
		{"deletes", len(s.dels) == int(hdr.numDels)},
	} {
		if err := check(c.name, c.ok); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func littleEndianHost() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}

// alignTo returns b, copied into a fresh allocation if its address is not a
// multiple of align. Sections are laid out back-to-back with no padding, so a
// section's start offset into the mmap'd file is not guaranteed aligned for
// the wider type its view functions below reinterpret it as. Reinterpreting
// an unaligned byte slice via unsafe.Pointer is undefined behaviour: x86
// tolerates it silently, but on arm64 (e.g. macOS CI runners) the compiler
// can emit aligned-load or vectorized instructions for the resulting slice
// that silently read from the wrong address, corrupting a handful of values
// without a crash (observed as a rare score mismatch after a segment
// reload — never reproduced on amd64). A fresh Go allocation is aligned to
// at least 8 bytes, satisfying every width used here.
func alignTo(b []byte, align uintptr) []byte {
	if len(b) == 0 || uintptr(unsafe.Pointer(&b[0]))%align == 0 {
		return b
	}
	return append([]byte(nil), b...)
}

func viewU64(b []byte) []uint64 {
	if len(b) < 8 {
		return nil
	}
	b = alignTo(b, 8)
	return unsafe.Slice((*uint64)(unsafe.Pointer(&b[0])), len(b)/8)
}

func viewU32(b []byte) []uint32 {
	if len(b) < 4 {
		return nil
	}
	b = alignTo(b, 4)
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func viewU16(b []byte) []uint16 {
	if len(b) < 2 {
		return nil
	}
	b = alignTo(b, 2)
	return unsafe.Slice((*uint16)(unsafe.Pointer(&b[0])), len(b)/2)
}

// Close unmaps the file. The Segment and everything aliasing it must not be
// used afterwards. On Windows the file cannot be deleted or replaced until
// this has been called.
func (s *Segment) Close() error {
	if s == nil {
		return nil
	}
	err := s.mm.close()
	s.data, s.docIDs, s.docBlob, s.vectors, s.wordVecs = nil, nil, nil, nil, nil
	return err
}

// Path is the file the segment was opened from.
func (s *Segment) Path() string { return s.path }

// Size is the file size in bytes.
func (s *Segment) Size() int { return len(s.data) }

// Verify checks every section's CRC-32C against the directory. It reads the
// whole file.
func (s *Segment) Verify() error {
	foot := s.data[len(s.data)-footerSize:]
	dirOff := binary.LittleEndian.Uint64(foot[0:])
	dirBytes := s.data[dirOff : len(s.data)-footerSize]
	for off := 0; off < len(dirBytes); off += dirEntrySize {
		kind := binary.LittleEndian.Uint32(dirBytes[off:])
		want := binary.LittleEndian.Uint32(dirBytes[off+20:])
		if got := crc32.Update(0, crcTable, s.dir[kind]); got != want {
			return fmt.Errorf("%w: section %d checksum mismatch", ErrCorrupt, kind)
		}
	}
	return nil
}

// Dims is the vector dimensionality.
func (s *Segment) Dims() int { return int(s.hdr.dims) }

// Meta returns the opaque metadata blob written at Finish (a copy).
func (s *Segment) Meta() []byte { return append([]byte(nil), s.meta...) }

// ---- documents ----

// NumDocs is the number of documents in the segment (including any the
// engine has since marked deleted).
func (s *Segment) NumDocs() int { return int(s.hdr.numDocs) }

// DocID returns the internal ID of the document at row i.
func (s *Segment) DocID(i int) uint64 { return s.docIDs[i] }

// FindDoc returns the row of the document with the given internal ID.
func (s *Segment) FindDoc(id uint64) (int, bool) {
	i := sort.Search(len(s.docIDs), func(i int) bool { return s.docIDs[i] >= id })
	if i < len(s.docIDs) && s.docIDs[i] == id {
		return i, true
	}
	return 0, false
}

func (s *Segment) docFields(i int) (orig, text []byte) {
	b := s.docBlob[s.docOffs[i]:s.docOffs[i+1]]
	next := func() []byte {
		n, w := binary.Uvarint(b)
		f := b[w : w+int(n)]
		b = b[w+int(n):]
		return f
	}
	return next(), next()
}

// Orig returns the caller-supplied document ID at row i (copied).
func (s *Segment) Orig(i int) string {
	o, _ := s.docFields(i)
	return string(o)
}

// Text returns the original document text at row i (copied).
func (s *Segment) Text(i int) string {
	_, t := s.docFields(i)
	return string(t)
}

// Attrs returns the encoded attributes at row i (copied); nil when none.
func (s *Segment) Attrs(i int) []byte {
	k := sort.Search(len(s.attrRows), func(k int) bool { return int(s.attrRows[k]) >= i })
	if k >= len(s.attrRows) || int(s.attrRows[k]) != i {
		return nil
	}
	return s.AttrsAt(k)
}

// NumAttrRows is the number of documents that carry attributes.
func (s *Segment) NumAttrRows() int { return len(s.attrRows) }

// AttrRow returns the document row of the k-th attribute blob.
func (s *Segment) AttrRow(k int) int { return int(s.attrRows[k]) }

// AttrsAt returns the k-th encoded attribute blob (copied).
func (s *Segment) AttrsAt(k int) []byte {
	return append([]byte(nil), s.attrBlob[s.attrOffs[k]:s.attrOffs[k+1]]...)
}

// DocLen is the BM25 length (token count) of the document at row i.
func (s *Segment) DocLen(i int) int { return int(s.docLen[i]) }

// Vec returns the float16 vector of the document at row i, or nil if it has
// none. Aliases the mapping.
func (s *Segment) Vec(i int) []uint16 {
	v := s.vecIdx[i]
	if v == ^uint32(0) {
		return nil
	}
	d := int(s.hdr.dims)
	return s.vectors[int(v)*d : int(v)*d+d : int(v)*d+d]
}

// NumVecs is the number of documents that carry a vector.
func (s *Segment) NumVecs() int { return int(s.hdr.numVecs) }

// ---- terms (BM25 vocabulary) ----

// NumTerms is the size of the term dictionary.
func (s *Segment) NumTerms() int { return s.terms.n() }

// TermKey returns term i's bytes. Aliases the mapping.
func (s *Segment) TermKey(i int) []byte { return s.terms.key(i) }

// FindTerm returns the dictionary index of term, or -1.
func (s *Segment) FindTerm(term string) int { return s.terms.find(term) }

func (s *Segment) termPayload(i int) (cf, df int, posts []byte) {
	_, p := s.terms.entry(i)
	c, n := binary.Uvarint(p)
	d, m := binary.Uvarint(p[n:])
	return int(c), int(d), p[n+m:]
}

// TermCF is term i's total occurrence count in this segment.
func (s *Segment) TermCF(i int) int { c, _, _ := s.termPayload(i); return c }

// TermDF is the number of documents in this segment containing term i.
func (s *Segment) TermDF(i int) int { _, d, _ := s.termPayload(i); return d }

// TermPostings calls fn(row, tf) for each document containing term i, in row
// order, until fn returns false.
func (s *Segment) TermPostings(i int, fn func(row, tf int) bool) {
	_, df, p := s.termPayload(i)
	row := 0
	for k := 0; k < df; k++ {
		d, n := binary.Uvarint(p)
		p = p[n:]
		tf, m := binary.Uvarint(p)
		p = p[m:]
		row += int(d)
		if !fn(row, int(tf)) {
			return
		}
	}
}

// ---- fragments / phonetic ----

func keyPostings(d dict, i int, fn func(row int) bool) {
	_, p := d.entry(i)
	cnt, n := binary.Uvarint(p)
	p = p[n:]
	row := 0
	for k := uint64(0); k < cnt; k++ {
		delta, w := binary.Uvarint(p)
		p = p[w:]
		row += int(delta)
		if !fn(row) {
			return
		}
	}
}

func keyCount(d dict, i int) int {
	_, p := d.entry(i)
	c, _ := binary.Uvarint(p)
	return int(c)
}

// NumFrags is the number of distinct edge-n-gram fragments.
func (s *Segment) NumFrags() int { return s.frags.n() }

// FragKey returns fragment i's bytes. Aliases the mapping.
func (s *Segment) FragKey(i int) []byte { return s.frags.key(i) }

// FindFrag returns the index of a fragment, or -1.
func (s *Segment) FindFrag(f string) int { return s.frags.find(f) }

// FragPostings calls fn(row) for each document containing fragment i.
func (s *Segment) FragPostings(i int, fn func(row int) bool) { keyPostings(s.frags, i, fn) }

// FragCount is the number of documents containing fragment i.
func (s *Segment) FragCount(i int) int { return keyCount(s.frags, i) }

// NumPhon is the number of distinct phonetic codes.
func (s *Segment) NumPhon() int { return s.phon.n() }

// PhonKey returns phonetic code i's bytes. Aliases the mapping.
func (s *Segment) PhonKey(i int) []byte { return s.phon.key(i) }

// FindPhon returns the index of a phonetic code, or -1.
func (s *Segment) FindPhon(f string) int { return s.phon.find(f) }

// PhonPostings calls fn(row) for each document containing phonetic code i.
func (s *Segment) PhonPostings(i int, fn func(row int) bool) { keyPostings(s.phon, i, fn) }

// ---- forward index ----

// Forward calls fn(termIdx, tf) for each distinct term of the document at row i.
func (s *Segment) Forward(i int, fn func(term, tf int)) {
	b := s.fwdData[s.fwdOffs[i]:s.fwdOffs[i+1]]
	cnt, n := binary.Uvarint(b)
	b = b[n:]
	term := 0
	for k := uint64(0); k < cnt; k++ {
		d, w := binary.Uvarint(b)
		b = b[w:]
		tf, m := binary.Uvarint(b)
		b = b[m:]
		term += int(d)
		fn(term, int(tf))
	}
}

// ---- word vectors ----

// NumWords is the number of stored word vectors.
func (s *Segment) NumWords() int { return s.words.n() }

// Word returns word i (copied).
func (s *Segment) Word(i int) string { return string(s.words.key(i)) }

// WordVec returns word i's float16 vector. Aliases the mapping.
func (s *Segment) WordVec(i int) []uint16 {
	d := int(s.hdr.dims)
	return s.wordVecs[i*d : i*d+d : i*d+d]
}

// FindWord returns the index of a word, or -1.
func (s *Segment) FindWord(w string) int { return s.words.find(w) }

// ---- deletes ----

// Dels returns the IDs of documents this segment deletes from older segments.
// Aliases the mapping.
func (s *Segment) Dels() []uint64 { return s.dels }

// ---- zero-copy accessors for merging ----
//
// These alias the mapping (valid until Close) and exist so a compaction can
// stream documents from one segment into another without copying each field
// several times.

// DocRaw returns the original ID and text bytes at row i. Aliases the mapping.
func (s *Segment) DocRaw(i int) (orig, text []byte) { return s.docFields(i) }

// AttrsRaw returns the encoded attributes at row i, or nil. Aliases the mapping.
func (s *Segment) AttrsRaw(i int) []byte {
	k := sort.Search(len(s.attrRows), func(k int) bool { return int(s.attrRows[k]) >= i })
	if k >= len(s.attrRows) || int(s.attrRows[k]) != i {
		return nil
	}
	return s.attrBlob[s.attrOffs[k]:s.attrOffs[k+1]:s.attrOffs[k+1]]
}

// WordKey returns word i's bytes. Aliases the mapping.
func (s *Segment) WordKey(i int) []byte { return s.words.key(i) }
