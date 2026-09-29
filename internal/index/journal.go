package index

import (
	"bytes"
	"encoding/binary"
	"hash/fnv"
)

// The write-ahead journal an embedding server keeps (DocumentJournal) records
// each added document. Text alone used to be enough; attributes must survive a
// crash too, or a replayed document would silently lose the metadata its
// filters rely on.
//
// A journalled value is the document text, unchanged, when the document has no
// attributes (so journals written before attributes existed replay as before).
// With attributes it is
//
//	0xFF 'Z' 'A' '1' | uvarint(len(attrs JSON)) | attrs JSON | text
//
// 0xFF can never begin valid UTF-8, so plain text is never mistaken for it.

var journalMagic = []byte{0xFF, 'Z', 'A', '1'}

func encodeJournalValue(text string, attrs Attrs) []byte {
	enc := encodeAttrs(attrs)
	if len(enc) == 0 {
		return []byte(text)
	}
	var n [binary.MaxVarintLen64]byte
	k := binary.PutUvarint(n[:], uint64(len(enc)))
	out := make([]byte, 0, len(journalMagic)+k+len(enc)+len(text))
	out = append(out, journalMagic...)
	out = append(out, n[:k]...)
	out = append(out, enc...)
	return append(out, text...)
}

// DecodeJournalValue splits a journalled value into the document text and its
// attributes (nil if it had none).
func DecodeJournalValue(v []byte) (text string, attrs Attrs) {
	if !bytes.HasPrefix(v, journalMagic) {
		return string(v), nil
	}
	rest := v[len(journalMagic):]
	n, k := binary.Uvarint(rest)
	if k <= 0 || uint64(len(rest)-k) < n {
		return string(v), nil // not ours after all
	}
	return string(rest[k+int(n):]), decodeAttrs(rest[k : k+int(n)])
}

// GetAttrs returns a copy of the attributes stored with the document
// originalID (nil if it has none or does not exist).
func (e *Engine) GetAttrs(originalID string) Attrs {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return copyAttrs(e.attrs[internalDocID(originalID)])
}

// internalDocID is the engine's internal identifier for a caller-supplied ID.
func internalDocID(originalID string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(originalID))
	return h.Sum64()
}
