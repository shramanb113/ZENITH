package analysis

import (
	"reflect"
	"regexp"
	"strings"
	"testing"
)

func terms(sp []Span) []string {
	out := make([]string, len(sp))
	for i, s := range sp {
		out[i] = s.Term
	}
	return out
}

func TestSegmentDevanagariKeepsMatras(t *testing.T) {
	got := Segment("पानी भर जाता है")
	want := []Span{{"पानी", 0, 4, 0}, {"भर", 5, 7, 0}, {"जाता", 8, 12, 0}, {"है", 13, 15, 0}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Segment = %#v, want %#v", got, want)
	}
}

func TestAnalyzeSpansDropsHindiStopWordsAndNumbersPositions(t *testing.T) {
	a := NewStandardAnalyzer()
	got := a.AnalyzeSpans("पानी भर जाता है")
	if want := []string{"पानी", "भर", "जाता"}; !reflect.DeepEqual(terms(got), want) {
		t.Fatalf("terms = %v, want %v", terms(got), want)
	}
	for i, s := range got {
		if s.Pos != i {
			t.Fatalf("Pos[%d] = %d", i, s.Pos)
		}
	}
}

func TestFoldIndicNuktaAndJoiners(t *testing.T) {
	if FoldIndic("ज़्यादा") != FoldIndic("ज्यादा") {
		t.Fatalf("nukta not folded: %q vs %q", FoldIndic("ज़्यादा"), FoldIndic("ज्यादा"))
	}
	if FoldIndic("क्‍ष") != "क्ष" || FoldIndic("क्‌ष") != "क्ष" {
		t.Fatal("ZWJ/ZWNJ not stripped")
	}
	if FoldIndic("ऩ") != "न" { // precomposed NNNA (न + nukta) must fold too
		t.Fatalf("precomposed nukta letter not folded: %q", FoldIndic("ऩ"))
	}
}

func TestAnalyzeSpansMixedScriptOffsets(t *testing.T) {
	a := NewStandardAnalyzer()
	text := "Basement mein पानी भर gaya, ज़्यादा"
	runes := []rune(text)
	for _, s := range a.AnalyzeSpans(text) {
		surface := string(runes[s.Start:s.End])
		if s.Term == "पानी" && surface != "पानी" {
			t.Fatalf("span for पानी points at %q", surface)
		}
		if s.Term == FoldIndic("ज़्यादा") && surface != "ज़्यादा" {
			t.Fatalf("span for folded term points at %q", surface)
		}
		if s.Term == "basement" && surface != "Basement" {
			t.Fatalf("span for basement points at %q", surface)
		}
	}
	for _, s := range a.AnalyzeSpans(text) {
		if s.Term == "mein" {
			t.Fatal("hinglish stop word 'mein' should be dropped")
		}
	}
}

// legacyTokenize is the pre-Z3 implementation, kept as an oracle for ASCII input.
func legacyTokenize(a *StandardAnalyzer, text string) []string {
	re := regexp.MustCompile(`[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`)
	var out []string
	for _, tok := range re.FindAllString(text, -1) {
		tok = strings.ToLower(tok)
		if _, stop := a.stopWords[tok]; stop {
			continue
		}
		if s := stem(tok); s != "" {
			out = append(out, a.resolveTerm(s))
		}
	}
	return out
}

func TestTokenizeASCIIUnchanged(t *testing.T) {
	a := NewStandardAnalyzer()
	for _, text := range []string{
		"Waterlogging in the basement during heavy rains!",
		"camelCaseTokens and HTTPServer v2 2BHK flats_near-metro",
		"It's the builder's fault: possession delayed by 24 months (again).",
		"",
	} {
		if got, want := a.Tokenize(text), legacyTokenize(a, text); !reflect.DeepEqual(got, want) {
			t.Fatalf("Tokenize(%q) = %v, legacy %v", text, got, want)
		}
	}
}

func TestSoundexNonASCIIIsEmpty(t *testing.T) {
	if got := Soundex("पानी"); got != "" {
		t.Fatalf("Soundex(पानी) = %q, want empty", got)
	}
	if Soundex("water") == "" {
		t.Fatal("Soundex(water) must still work")
	}
}
