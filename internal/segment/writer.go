package segment

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// DocIn is one document to write. Docs must be supplied in ascending ID order.
type DocIn struct {
	ID    uint64
	Orig  string
	Text  string
	Attrs []byte
	Len   int      // BM25 document length (token count)
	Vec   []uint16 // float16 bits; nil = no vector, otherwise len == dims
}

// Posting is one (document row, term frequency) pair. Rows index the docs
// written by WriteDocs (0-based position in ID order).
type Posting struct {
	Doc int
	TF  int
}

// TermIn is one BM25 vocabulary term. Terms must be supplied in ascending
// byte order; Posts in ascending Doc order. CF is the term's total occurrence
// count across the segment's documents (used to maintain the global term
// reference counts).
type TermIn struct {
	Term  string
	CF    int
	Posts []Posting
}

// KeyIn is one lexical posting key (edge n-gram fragment or phonetic code)
// with the ascending rows of the documents it occurs in. Keys ascend.
type KeyIn struct {
	Key  string
	Docs []int
}

// TermFreq is one entry of a document's forward term list. Term is an index
// into the segment's term dictionary; lists must be ascending by Term.
type TermFreq struct {
	Term int
	TF   int
}

// Writer builds a segment file. Sections are appended in call order and the
// file only becomes a valid segment once Finish returns nil; until then it is
// an incomplete temp file and must be discarded on error (Abort).
//
// Write errors are sticky: the first one is remembered, later calls become
// no-ops, and Finish reports it.
type Writer struct {
	path string
	f    *fsx.File
	bw   *bufio.Writer
	pos  int64
	dir  []dirEntry
	hdr  header
	err  error
	done bool
}

// Create starts a new segment at path (truncating any existing file).
func Create(path string, dims int) (*Writer, error) {
	f, err := fsx.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_RDWR, 0o644)
	if err != nil {
		return nil, err
	}
	w := &Writer{path: path, f: f, bw: bufio.NewWriterSize(f, 1<<20)}
	w.hdr.dims = uint32(dims)
	w.write(make([]byte, headerSize)) // patched in Finish
	return w, w.err
}

func (w *Writer) write(p []byte) {
	if w.err != nil {
		return
	}
	n, err := w.bw.Write(p)
	w.pos += int64(n)
	if err != nil {
		w.err = err
	}
}

type secWriter struct {
	w   *Writer
	crc uint32
	n   uint64
}

func (s *secWriter) Write(p []byte) (int, error) {
	s.w.write(p)
	s.crc = crc32.Update(s.crc, crcTable, p)
	s.n += uint64(len(p))
	return len(p), s.w.err
}

func (s *secWriter) uvarint(v uint64) {
	var b [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(b[:], v)
	s.Write(b[:n])
}

func (s *secWriter) writeString(str string) {
	if len(str) == 0 {
		return
	}
	s.Write(unsafe.Slice(unsafe.StringData(str), len(str)))
}

func (w *Writer) section(kind uint32, fill func(s *secWriter)) {
	if w.err != nil {
		return
	}
	if pad := (8 - w.pos%8) % 8; pad != 0 {
		w.write(make([]byte, pad))
	}
	s := &secWriter{w: w}
	start := w.pos
	fill(s)
	w.dir = append(w.dir, dirEntry{kind: kind, off: uint64(start), n: s.n, crc: s.crc})
}

func putU64s(s *secWriter, vs []uint64) {
	var b [8]byte
	for _, v := range vs {
		binary.LittleEndian.PutUint64(b[:], v)
		s.Write(b[:])
	}
}

func putU32s(s *secWriter, vs []uint32) {
	var b [4]byte
	for _, v := range vs {
		binary.LittleEndian.PutUint32(b[:], v)
		s.Write(b[:])
	}
}

func putU16s(s *secWriter, vs []uint16) {
	var b [2]byte
	for _, v := range vs {
		binary.LittleEndian.PutUint16(b[:], v)
		s.Write(b[:])
	}
}

// WriteDocs writes n documents, supplied by at(i) for i in [0,n) in ascending
// ID order. at is called more than once per index (once per section), so it
// must be cheap and return the same value each time.
func (w *Writer) WriteDocs(n int, at func(i int) DocIn) {
	w.hdr.numDocs = uint32(n)

	w.section(secDocIDs, func(s *secWriter) {
		ids := make([]uint64, n)
		for i := 0; i < n; i++ {
			ids[i] = at(i).ID
			if i > 0 && ids[i] <= ids[i-1] && w.err == nil {
				w.err = fmt.Errorf("segment: docs not in ascending ID order at %d", i)
			}
		}
		putU64s(s, ids)
	})

	offs := make([]uint64, 0, n+1)
	w.section(secDocBlob, func(s *secWriter) {
		for i := 0; i < n; i++ {
			d := at(i)
			offs = append(offs, s.n)
			s.uvarint(uint64(len(d.Orig)))
			s.writeString(d.Orig)
			s.uvarint(uint64(len(d.Text)))
			s.writeString(d.Text)
		}
		offs = append(offs, s.n)
	})
	w.section(secDocOffs, func(s *secWriter) { putU64s(s, offs) })

	// Attributes live in their own sections so opening an index can load them
	// without paging in every document's text.
	var attrRows []uint32
	attrOffs := []uint64{0}
	w.section(secAttrBlob, func(s *secWriter) {
		for i := 0; i < n; i++ {
			if a := at(i).Attrs; len(a) > 0 {
				attrRows = append(attrRows, uint32(i))
				s.Write(a)
				attrOffs = append(attrOffs, s.n)
			}
		}
	})
	w.section(secAttrOffs, func(s *secWriter) { putU64s(s, attrOffs) })
	w.section(secAttrRows, func(s *secWriter) { putU32s(s, attrRows) })

	w.section(secDocLen, func(s *secWriter) {
		ls := make([]uint32, n)
		for i := 0; i < n; i++ {
			ls[i] = uint32(at(i).Len)
		}
		putU32s(s, ls)
	})

	vecIdx := make([]uint32, n)
	numVecs := 0
	for i := 0; i < n; i++ {
		if v := at(i).Vec; v != nil {
			if len(v) != int(w.hdr.dims) && w.err == nil {
				w.err = fmt.Errorf("segment: doc %d vector has %d dims, want %d", i, len(v), w.hdr.dims)
			}
			vecIdx[i] = uint32(numVecs)
			numVecs++
		} else {
			vecIdx[i] = ^uint32(0)
		}
	}
	w.hdr.numVecs = uint32(numVecs)
	w.section(secVecIdx, func(s *secWriter) { putU32s(s, vecIdx) })
	w.section(secVectors, func(s *secWriter) {
		for i := 0; i < n; i++ {
			if v := at(i).Vec; v != nil {
				putU16s(s, v)
			}
		}
	})
}

// writeDict writes a blob of variable-length entries plus its offset table
// and returns the entry count. emit is called once per entry, and the fill
// function it is given writes that entry's bytes.
func (w *Writer) writeDict(offKind, dataKind uint32, entries func(emit func(fill func(s *secWriter)))) int {
	var offs []uint64
	w.section(dataKind, func(s *secWriter) {
		entries(func(fill func(s *secWriter)) {
			offs = append(offs, s.n)
			fill(s)
		})
		offs = append(offs, s.n)
	})
	w.section(offKind, func(s *secWriter) { putU64s(s, offs) })
	if len(offs) == 0 {
		return 0
	}
	return len(offs) - 1
}

// WriteTerms writes the BM25 vocabulary; next yields terms in ascending order.
func (w *Writer) WriteTerms(next func() (TermIn, bool)) {
	w.hdr.numTerms = uint32(w.writeDict(secTermOffs, secTermData, func(emit func(func(*secWriter))) {
		for {
			t, ok := next()
			if !ok || w.err != nil {
				return
			}
			emit(func(s *secWriter) {
				s.uvarint(uint64(len(t.Term)))
				s.writeString(t.Term)
				s.uvarint(uint64(t.CF))
				s.uvarint(uint64(len(t.Posts)))
				prev := 0
				for _, p := range t.Posts {
					s.uvarint(uint64(p.Doc - prev))
					s.uvarint(uint64(p.TF))
					prev = p.Doc
				}
			})
		}
	}))
}

func (w *Writer) writeKeys(offKind, dataKind uint32, next func() (KeyIn, bool)) int {
	return w.writeDict(offKind, dataKind, func(emit func(func(*secWriter))) {
		for {
			k, ok := next()
			if !ok || w.err != nil {
				return
			}
			emit(func(s *secWriter) {
				s.uvarint(uint64(len(k.Key)))
				s.writeString(k.Key)
				s.uvarint(uint64(len(k.Docs)))
				prev := 0
				for _, d := range k.Docs {
					s.uvarint(uint64(d - prev))
					prev = d
				}
			})
		}
	})
}

// WriteFrags writes the edge-n-gram fragment postings; keys ascend.
func (w *Writer) WriteFrags(next func() (KeyIn, bool)) {
	w.hdr.numFrags = uint32(w.writeKeys(secFragOffs, secFragData, next))
}

// WritePhon writes the phonetic-code postings; keys ascend.
func (w *Writer) WritePhon(next func() (KeyIn, bool)) {
	w.hdr.numPhon = uint32(w.writeKeys(secPhonOffs, secPhonData, next))
}

// WriteForward writes each document's (term, tf) list. Call after WriteTerms:
// Term values index that dictionary. at is called once per doc, in order.
func (w *Writer) WriteForward(n int, at func(i int) []TermFreq) {
	var offs []uint64
	w.section(secFwdData, func(s *secWriter) {
		for i := 0; i < n; i++ {
			offs = append(offs, s.n)
			tfs := at(i)
			s.uvarint(uint64(len(tfs)))
			prev := 0
			for _, t := range tfs {
				s.uvarint(uint64(t.Term - prev))
				s.uvarint(uint64(t.TF))
				prev = t.Term
			}
		}
		offs = append(offs, s.n)
	})
	w.section(secFwdOffs, func(s *secWriter) { putU64s(s, offs) })
}

// WriteWords writes word vectors (for neural query expansion); words ascend.
// at is called twice per index (dictionary pass, then vector pass).
func (w *Writer) WriteWords(n int, at func(i int) (string, []uint16)) {
	w.hdr.numWords = uint32(n)
	var offs []uint64
	w.section(secWordData, func(s *secWriter) {
		for i := 0; i < n; i++ {
			offs = append(offs, s.n)
			word, _ := at(i)
			s.uvarint(uint64(len(word)))
			s.writeString(word)
		}
		offs = append(offs, s.n)
	})
	w.section(secWordOffs, func(s *secWriter) { putU64s(s, offs) })
	w.section(secWordVecs, func(s *secWriter) {
		for i := 0; i < n; i++ {
			_, v := at(i)
			if len(v) != int(w.hdr.dims) && w.err == nil {
				w.err = fmt.Errorf("segment: word %d vector has %d dims, want %d", i, len(v), w.hdr.dims)
			}
			putU16s(s, v)
		}
	})
}

// WriteDels records the IDs of documents this segment deletes from older ones.
func (w *Writer) WriteDels(ids []uint64) {
	w.hdr.numDels = uint32(len(ids))
	w.section(secDels, func(s *secWriter) { putU64s(s, ids) })
}

// Finish writes the directory and footer, fsyncs and closes the file. On any
// error the partial file is removed.
func (w *Writer) Finish(meta []byte) error {
	if w.done {
		return fmt.Errorf("segment: writer already finished")
	}
	w.done = true
	w.section(secMeta, func(s *secWriter) { s.Write(meta) })
	if w.err == nil {
		if pad := (8 - w.pos%8) % 8; pad != 0 {
			w.write(make([]byte, pad))
		}
		dirOff := w.pos
		dirBytes := make([]byte, 0, len(w.dir)*dirEntrySize)
		for _, e := range w.dir {
			var b [dirEntrySize]byte
			binary.LittleEndian.PutUint32(b[0:], e.kind)
			binary.LittleEndian.PutUint64(b[4:], e.off)
			binary.LittleEndian.PutUint64(b[12:], e.n)
			binary.LittleEndian.PutUint32(b[20:], e.crc)
			dirBytes = append(dirBytes, b[:]...)
		}
		w.write(dirBytes)

		hdr := w.hdr.encode()
		crc := crc32.Update(crc32.Update(0, crcTable, hdr), crcTable, dirBytes)
		var foot [footerSize]byte
		binary.LittleEndian.PutUint64(foot[0:], uint64(dirOff))
		binary.LittleEndian.PutUint32(foot[8:], crc)
		copy(foot[12:], footerMagic)
		w.write(foot[:])

		if w.err == nil {
			w.err = w.bw.Flush()
		}
		if w.err == nil {
			_, w.err = w.f.WriteAt(hdr, 0)
		}
		if w.err == nil {
			w.err = w.f.Sync()
		}
	}
	cerr := w.f.Close()
	if w.err != nil {
		fsx.Remove(w.path)
		return w.err
	}
	if cerr != nil {
		fsx.Remove(w.path)
		return cerr
	}
	// Make the new directory entry durable before anyone commits a reference to
	// this file (the manifest rename): with the data fsynced above, a crash can
	// then never leave a manifest that names a segment whose file is missing.
	fsx.SyncDir(filepath.Dir(w.path))
	return nil
}

// Abort discards a partially written segment.
func (w *Writer) Abort() {
	if w.done {
		return
	}
	w.done = true
	w.f.Close()
	fsx.Remove(w.path)
}

// Err reports the first write error, if any.
func (w *Writer) Err() error { return w.err }
