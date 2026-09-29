package localembedder

import (
	"strings"
	"testing"
)

// ── encodePair ───────────────────────────────────────────────────────────────

func TestEncodePair_Shape(t *testing.T) {
	tok := mustTokenizer(t)
	ids, mask, typ := tok.encodePair("how many calories in an apple", "an apple has about 95 calories", 256)

	if len(ids) != len(mask) || len(ids) != len(typ) {
		t.Fatalf("mismatched lengths: ids=%d mask=%d typ=%d", len(ids), len(mask), len(typ))
	}
	if ids[0] != tokenCLS {
		t.Fatalf("ids[0] = %d, want [CLS] (%d)", ids[0], tokenCLS)
	}
	if ids[len(ids)-1] != tokenSEP {
		t.Fatalf("last id = %d, want [SEP] (%d)", ids[len(ids)-1], tokenSEP)
	}
	// Exactly two [SEP] tokens: one closing the query, one closing the doc.
	seps := 0
	for _, id := range ids {
		if id == tokenSEP {
			seps++
		}
	}
	if seps != 2 {
		t.Fatalf("found %d [SEP] tokens, want 2", seps)
	}
	for _, m := range mask {
		if m != 1 {
			t.Fatalf("encodePair produced a non-1 mask entry %d before any padding is applied", m)
		}
	}
}

func TestEncodePair_TypeIDsSplitAtFirstSEP(t *testing.T) {
	tok := mustTokenizer(t)
	ids, _, typ := tok.encodePair("apple calories", "an apple has about 95 calories today", 256)

	firstSEP := sepIndex(ids)
	if firstSEP < 0 {
		t.Fatal("no [SEP] found")
	}
	for i := 0; i <= firstSEP; i++ {
		if typ[i] != 0 {
			t.Fatalf("typeIDs[%d] = %d, want 0 (query segment)", i, typ[i])
		}
	}
	for i := firstSEP + 1; i < len(typ); i++ {
		if typ[i] != 1 {
			t.Fatalf("typeIDs[%d] = %d, want 1 (doc segment)", i, typ[i])
		}
	}
}

func TestEncodePair_LongDocTruncatesNotQuery(t *testing.T) {
	tok := mustTokenizer(t)
	query := "how many calories in an apple"
	doc := strings.Repeat("word ", 500)

	ids, _, _ := tok.encodePair(query, doc, 64)
	if len(ids) > 64 {
		t.Fatalf("encodePair produced %d tokens, want <= 64", len(ids))
	}
	// The query's own tokens must all survive even though the doc was cut.
	qIDs, _, _ := tok.encodePair(query, "", 64)
	qLen := len(qIDs) - 3 // minus [CLS] and the two [SEP]s (query-close and doc-close)
	firstSEP := sepIndex(ids)
	if firstSEP != qLen+1 { // [CLS] + query tokens, then [SEP]
		t.Fatalf("query segment length = %d, want %d (query truncated by a long doc)", firstSEP-1, qLen)
	}
}

func TestEncodePair_EmptyDoc(t *testing.T) {
	tok := mustTokenizer(t)
	ids, mask, typ := tok.encodePair("apple calories", "", 256)
	if len(ids) == 0 || ids[len(ids)-1] != tokenSEP {
		t.Fatalf("encodePair with empty doc should still end in [SEP]: %v", ids)
	}
	if len(ids) != len(mask) || len(ids) != len(typ) {
		t.Fatalf("mismatched lengths with empty doc")
	}
}

// ── Reranker error paths ────────────────────────────────────────────────────
// No model is installed in a fresh temp dir, and a build without CGo can't
// load one either way — NewReranker must report an error, never panic or
// return a non-functional Reranker silently.
func TestNewReranker_ErrorsWhenModelMissing(t *testing.T) {
	_, err := NewReranker("", t.TempDir())
	if err == nil {
		t.Fatal("NewReranker succeeded with no model installed and no CGo guarantee")
	}
}

func TestNewReranker_UnknownID(t *testing.T) {
	if err := unavailable(); err != nil {
		t.Skip("CGo unavailable in this build; unknown-id lookup is unreachable before it")
	}
	_, err := NewReranker("not-a-real-reranker", t.TempDir())
	if err == nil {
		t.Fatal("NewReranker succeeded with an unregistered id")
	}
}
