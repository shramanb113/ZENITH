package index

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"io"
	"log/slog"
	"os"
)

// legacyIndex is the decoded contents of a gob-format index file (versions 4
// and 5). The gob format is no longer written; it can only be read, to migrate
// it to the segment format (Engine.LoadLegacy, `zenith migrate`).
type legacyIndex struct {
	Version  uint16
	Embedder string // "" when the file predates the embedder header (v4)
	Dims     int

	Data        map[string][]uint64
	IDMapping   map[uint64]string
	Vectors     map[uint64]VectorEntry
	Phon        map[string][]uint64
	Seen        map[string]int
	WordVectors map[string]VectorEntry

	BM25Lengths   map[uint64]int
	BM25TermFreqs map[uint64]map[string]int
	BM25DocFreq   map[string]int
	BM25TotalDocs int
	BM25TotalLen  int

	DocToks map[uint64][]string
	DocText map[uint64]string
	Attrs   map[uint64]Attrs
}

// readLegacy decodes a v4 or v5 gob index file. It decodes into brand-new maps
// so a failure part-way through never leaves partial state anywhere.
func readLegacy(path string) (*legacyIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	version, name, dims, err := readHeader(r)
	if err != nil {
		return nil, err
	}
	if version != 4 && version != 5 {
		return nil, fmt.Errorf("index: cannot migrate format version %d (supported: 4 and 5) — rebuild the index instead", version)
	}
	li := &legacyIndex{Version: version, Embedder: name, Dims: dims}

	var (
		vVocab map[int][]string    // dead state, decoded and discarded
		vToken map[string]int      // dead state
		vFrag  map[uint64][]string // recomputed from tokens now
		tfLen  map[uint64]int      // TF-IDF, never used for ranking
		tfTF   map[uint64]map[string]int
		tfDF   map[string]int
		tfN    int
	)
	li.Data = make(map[string][]uint64)
	li.IDMapping = make(map[uint64]string)
	li.Vectors = make(map[uint64]VectorEntry)
	li.Phon = make(map[string][]uint64)
	li.Seen = make(map[string]int)
	li.WordVectors = make(map[string]VectorEntry)
	li.DocToks = make(map[uint64][]string)
	li.DocText = make(map[uint64]string)
	li.Attrs = make(map[uint64]Attrs)

	state := []any{
		&li.Data, &li.IDMapping, &li.Vectors,
		&vToken, &li.Phon, &vVocab,
		&li.Seen, &li.WordVectors, &vFrag,
		&li.BM25Lengths, &li.BM25TermFreqs, &li.BM25DocFreq, &li.BM25TotalDocs, &li.BM25TotalLen,
		&tfLen, &tfTF, &tfDF, &tfN,
		&li.DocToks, // v3
		&li.DocText, // v4
	}
	if version >= 5 {
		state = append(state, &li.Attrs)
	}
	dec := gob.NewDecoder(r)
	for _, s := range state {
		if err := dec.Decode(s); err != nil {
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return nil, fmt.Errorf("index: decode legacy index state: %w", err)
		}
	}
	return li, nil
}

// LoadLegacy replaces the engine's contents with a gob-format (v4/v5) index
// file, held in memory as the mutable delta. The next Save writes it out in
// the segment format. For v5 files the recorded embedder identity is checked
// exactly as Load does; v4 files recorded none, so the caller is trusted.
func (e *Engine) LoadLegacy(path string) error {
	li, err := readLegacy(path)
	if err != nil {
		return err
	}
	e.compactMu.Lock()
	defer e.compactMu.Unlock()
	e.mu.Lock()
	defer e.mu.Unlock()

	if li.Version >= 5 {
		if err := e.checkEmbedder(li.Embedder, li.Dims); err != nil {
			return err
		}
	}
	e.closeLayersLocked()

	e.inverted.Lock()
	e.vectors.Lock()
	e.phonetics.Lock()
	e.inverted.ReplaceAll(li.Data, li.Seen, li.DocToks)
	e.vectors.ReplaceAll(li.Vectors, li.WordVectors)
	e.phonetics.ReplaceAll(li.Phon)
	e.idMapping = li.IDMapping
	e.docText = li.DocText
	e.attrs = li.Attrs
	if e.attrs == nil {
		e.attrs = make(map[uint64]Attrs)
	}
	e.attrIdx = rebuildAttrIndex(e.attrs)
	e.inverted.Unlock()
	e.vectors.Unlock()
	e.phonetics.Unlock()

	e.bm25.LoadState(li.BM25Lengths, li.BM25TermFreqs, li.BM25DocFreq, li.BM25TotalDocs, li.BM25TotalLen)
	e.bm25.SetBacking(segBacking{e})
	e.pendingDels = nil
	e.dbPath = ""
	e.bk = nil

	e.rebuildANNAfterLoadLocked()
	if err := e.rebuildFSTLocked(); err != nil {
		slog.Warn("index: FST rebuild after legacy load failed", "error", err)
	}
	slog.Info("Legacy index loaded", "docs", len(e.idMapping), "format", li.Version)
	return nil
}
