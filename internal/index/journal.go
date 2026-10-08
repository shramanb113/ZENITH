package index

import (
	"bytes"
	"encoding/binary"
	"hash/fnv"
	"math"
)

// The write-ahead journal an embedding server keeps (DocumentJournal) records
// each added document. A journalled value is the document text, unchanged,
// when the document has neither attributes nor a vector (so journals written
// before either existed replay as before).
//
// v1 (attrs, no vector — superseded by v2 for new writes, kept decodable):
//
//	0xFF 'Z' 'A' '1' | uvarint(len(attrs JSON)) | attrs JSON | text
//
// v2 (attrs and/or vector):
//
//	0xFF 'Z' 'A' '2' | uvarint(len(attrs JSON)) | attrs JSON |
//	    uvarint(vecDim) | vecDim*4 bytes (float32 little-endian) | text
//
// vecDim=0 means no vector was available to journal (e.g. embedding failed);
// a replay then falls back to re-embedding, same as before this format
// existed. 0xFF can never begin valid UTF-8, so plain text is never mistaken
// for either tagged format.

var journalMagicV1 = []byte{0xFF, 'Z', 'A', '1'}
var journalMagicV2 = []byte{0xFF, 'Z', 'A', '2'}

func encodeJournalValue(text string, vector []float32, attrs Attrs) []byte {
	enc := encodeAttrs(attrs)
	if len(enc) == 0 && len(vector) == 0 {
		return []byte(text)
	}

	var attrsLenBuf [binary.MaxVarintLen64]byte
	attrsLenN := binary.PutUvarint(attrsLenBuf[:], uint64(len(enc)))

	var vecDimBuf [binary.MaxVarintLen64]byte
	vecDimN := binary.PutUvarint(vecDimBuf[:], uint64(len(vector)))

	out := make([]byte, 0, len(journalMagicV2)+attrsLenN+len(enc)+vecDimN+len(vector)*4+len(text))
	out = append(out, journalMagicV2...)
	out = append(out, attrsLenBuf[:attrsLenN]...)
	out = append(out, enc...)
	out = append(out, vecDimBuf[:vecDimN]...)
	for _, f := range vector {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
		out = append(out, b[:]...)
	}
	return append(out, text...)
}

// DecodeJournalValue splits a journalled value into the document text, its
// vector (nil if none was journalled), and its attributes (nil if it had
// none).
func DecodeJournalValue(v []byte) (text string, vector []float32, attrs Attrs) {
	switch {
	case bytes.HasPrefix(v, journalMagicV2):
		return decodeJournalV2(v[len(journalMagicV2):], v)
	case bytes.HasPrefix(v, journalMagicV1):
		rest := v[len(journalMagicV1):]
		n, k := binary.Uvarint(rest)
		if k <= 0 || uint64(len(rest)-k) < n {
			return string(v), nil, nil // not ours after all
		}
		return string(rest[k+int(n):]), nil, decodeAttrs(rest[k : k+int(n)])
	default:
		return string(v), nil, nil
	}
}

func decodeJournalV2(rest []byte, whole []byte) (text string, vector []float32, attrs Attrs) {
	attrsLen, k1 := binary.Uvarint(rest)
	if k1 <= 0 || uint64(len(rest)-k1) < attrsLen {
		return string(whole), nil, nil
	}
	attrsJSON := rest[k1 : k1+int(attrsLen)]
	rest = rest[k1+int(attrsLen):]

	vecDim, k2 := binary.Uvarint(rest)
	// Division, not vecDim*4 compared against a length: multiplying an
	// attacker/corruption-controlled vecDim by 4 can overflow uint64 and
	// wrap to a small number, which would let an oversized vecDim slip
	// past this guard and then panic inside make([]float32, vecDim) below.
	// Dividing the (bounded, real slice) length instead can never overflow.
	if k2 <= 0 || vecDim > uint64(len(rest)-k2)/4 {
		return string(whole), nil, nil
	}
	vecBytes := rest[k2 : k2+int(vecDim)*4]
	rest = rest[k2+int(vecDim)*4:]

	var vec []float32
	if vecDim > 0 {
		vec = make([]float32, vecDim)
		for i := range vec {
			vec[i] = math.Float32frombits(binary.LittleEndian.Uint32(vecBytes[i*4 : i*4+4]))
		}
	}
	return string(rest), vec, decodeAttrs(attrsJSON)
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
