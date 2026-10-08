package index

import (
	"encoding/binary"
	"testing"
)

func TestEncodeDecodeJournalValue_V2WithVectorAndAttrs(t *testing.T) {
	vec := []float32{0.1, -0.2, 0.3}
	attrs := Attrs{"lang": AttrValue{Kind: AttrString, S: "en"}}
	encoded := encodeJournalValue("hello world", vec, attrs)

	text, gotVec, gotAttrs := DecodeJournalValue(encoded)
	if text != "hello world" {
		t.Errorf("text = %q, want %q", text, "hello world")
	}
	if len(gotVec) != len(vec) {
		t.Fatalf("vector length = %d, want %d", len(gotVec), len(vec))
	}
	for i := range vec {
		if gotVec[i] != vec[i] {
			t.Errorf("vector[%d] = %v, want %v", i, gotVec[i], vec[i])
		}
	}
	if gotAttrs["lang"].S != "en" {
		t.Errorf("attrs[lang] = %q, want %q", gotAttrs["lang"].S, "en")
	}
}

func TestEncodeDecodeJournalValue_V2VectorNoAttrs(t *testing.T) {
	vec := []float32{1, 2, 3, 4}
	encoded := encodeJournalValue("plain doc", vec, nil)
	text, gotVec, gotAttrs := DecodeJournalValue(encoded)
	if text != "plain doc" || len(gotVec) != 4 || gotAttrs != nil {
		t.Errorf("got (%q, len=%d, %v), want (%q, len=4, nil)", text, len(gotVec), gotAttrs, "plain doc")
	}
}

func TestEncodeDecodeJournalValue_NoVectorNoAttrsIsPlainText(t *testing.T) {
	encoded := encodeJournalValue("just text", nil, nil)
	if string(encoded) != "just text" {
		t.Errorf("encoded = %q, want plain %q (no magic bytes)", encoded, "just text")
	}
	text, vec, attrs := DecodeJournalValue(encoded)
	if text != "just text" || vec != nil || attrs != nil {
		t.Errorf("got (%q, %v, %v), want (%q, nil, nil)", text, vec, attrs, "just text")
	}
}

func TestDecodeJournalValue_LegacyV1StillDecodes(t *testing.T) {
	// v1 layout: magic 'Z''A''1' | uvarint(len(attrsJSON)) | attrsJSON | text
	attrsJSON := encodeAttrs(Attrs{"k": AttrValue{Kind: AttrString, S: "v"}})
	var buf []byte
	buf = append(buf, 0xFF, 'Z', 'A', '1')
	var n [binary.MaxVarintLen64]byte
	k := binary.PutUvarint(n[:], uint64(len(attrsJSON)))
	buf = append(buf, n[:k]...)
	buf = append(buf, attrsJSON...)
	buf = append(buf, []byte("legacy text")...)

	text, vec, attrs := DecodeJournalValue(buf)
	if text != "legacy text" || vec != nil || attrs["k"].S != "v" {
		t.Errorf("got (%q, %v, %v), want (%q, nil, k=v)", text, vec, attrs, "legacy text")
	}
}

func TestDecodeJournalValue_UnrecognizedMagicFallsBackToPlainText(t *testing.T) {
	buf := append([]byte{0xFF, 'Z', 'A', '9'}, []byte("whatever")...)
	text, vec, attrs := DecodeJournalValue(buf)
	if vec != nil || attrs != nil {
		t.Errorf("got vec=%v attrs=%v, want both nil for unrecognized magic", vec, attrs)
	}
	if text != string(buf) {
		t.Errorf("text = %q, want the whole raw buffer treated as text", text)
	}
}
