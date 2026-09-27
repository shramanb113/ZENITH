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

// legacyTokenize is an oracle for ASCII input that doesn't involve acronyms or
// camelCase run boundaries (Segment's acronym-aware scanner replaced the old
// regex `[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`, which shredded acronyms into single
// letters — see TestTokenizeAcronymsNotShredded / TestTokenizeCamelCaseWithAcronymPrefix
// below for that behavior specifically). Tokenize never resolves through the FST
// (that only happens in AnalyzeQuery), so this oracle doesn't either.
func legacyTokenize(a *StandardAnalyzer, text string) []string {
	re := regexp.MustCompile(`[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`)
	var out []string
	for _, tok := range re.FindAllString(text, -1) {
		tok = strings.ToLower(tok)
		if _, stop := a.stopWords[tok]; stop {
			continue
		}
		if s := stem(tok); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func TestTokenizeASCIIUnchanged(t *testing.T) {
	a := NewStandardAnalyzer()
	for _, text := range []string{
		"Waterlogging in the basement during heavy rains!",
		"It's the builder's fault: possession delayed by 24 months (again).",
		"",
	} {
		if got, want := a.Tokenize(text), legacyTokenize(a, text); !reflect.DeepEqual(got, want) {
			t.Fatalf("Tokenize(%q) = %v, legacy %v", text, got, want)
		}
	}
}

func TestTokenizeNeverResolvesThroughFST(t *testing.T) {
	// C1 regression: Analyze()/Tokenize() must never rewrite a document's own
	// terms through the FST. Build an FST containing "card", then analyze text
	// containing the unrelated word "car" — it must come back as "car", not
	// silently become "card".
	a := NewStandardAnalyzer()
	fst := NewFSTDictionary()
	if err := fst.Build([]string{"card"}); err != nil {
		t.Fatalf("Build: %v", err)
	}
	a.SetFST(fst)

	got := a.Tokenize("car racing")
	want := []string{"car", "race"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Tokenize(%q) = %v, want %v (must not resolve 'car' to 'card')", "car racing", got, want)
	}

	tokens := a.Analyze("car racing")
	for _, tok := range tokens {
		if tok.Term == "card" {
			t.Fatalf("Analyze() resolved a term through the FST: %v", tokens)
		}
	}
}

func TestTokenizeAcronymsNotShredded(t *testing.T) {
	a := NewStandardAnalyzer()
	cases := []struct{ word, want string }{
		{"HTTP", "http"},
		{"NASA", "nasa"},
		{"API", "api"},
		{"SQL", "sql"},
	}
	for _, c := range cases {
		got := a.Tokenize(c.word)
		if len(got) != 1 || got[0] != c.want {
			t.Fatalf("Tokenize(%q) = %v, want [%q]", c.word, got, c.want)
		}
	}
}

func TestTokenizeAcronymMatchesLowercaseQuery(t *testing.T) {
	a := NewStandardAnalyzer()
	docTokens := a.Tokenize("Our server exposes an HTTP API.")
	queryTokens := a.Tokenize("http api")

	docSet := make(map[string]struct{}, len(docTokens))
	for _, tok := range docTokens {
		docSet[tok] = struct{}{}
	}
	for _, qt := range queryTokens {
		if _, ok := docSet[qt]; !ok {
			t.Fatalf("query token %q not found in doc tokens %v", qt, docTokens)
		}
	}
}

func TestTokenizeCamelCaseWithAcronymPrefix(t *testing.T) {
	a := NewStandardAnalyzer()
	got := a.Tokenize("HTTPServer")
	want := []string{"http", "server"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Tokenize(HTTPServer) = %v, want %v", got, want)
	}
}

func TestTokenizeIPv6AddressSplitsSensibly(t *testing.T) {
	a := NewStandardAnalyzer()
	got := a.Tokenize("IPv6Address")
	for _, tok := range got {
		if len(tok) == 1 {
			t.Fatalf("Tokenize(IPv6Address) produced a shredded single-letter token: %v", got)
		}
	}
	found := false
	for _, tok := range got {
		if strings.HasPrefix(tok, "address") {
			found = true
		}
	}
	if !found {
		t.Fatalf("Tokenize(IPv6Address) = %v, expected an 'address' token", got)
	}
}

func TestTokenize2BHKSplitsDigitFromAcronym(t *testing.T) {
	a := NewStandardAnalyzer()
	got := a.Tokenize("2BHK flat")
	want := []string{"2", "bhk", "flat"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Tokenize(2BHK flat) = %v, want %v", got, want)
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
