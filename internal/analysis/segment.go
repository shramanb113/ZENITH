// internal/analysis/segment.go
package analysis

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

const (
	zwnj  = '‌'
	zwj   = '‍'
	nukta = '़'
)

// Span is one token with its position in the ORIGINAL text. Start and End are rune (code point)
// offsets, so text[Start:End] in Python (or []rune(text)[Start:End] in Go) is the surface form.
// Pos is the token's index among the kept (analysed) tokens; Segment leaves it 0.
type Span struct {
	Term       string
	Start, End int
	Pos        int
}

// asciiWord is the pre-Z3 camelCase-aware pattern. It is applied only inside ASCII runs so English
// tokenisation is byte-for-byte unchanged.
var asciiWord = regexp.MustCompile(`[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`)

func isWordRune(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r) || r == zwj || r == zwnj
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= 0x80 {
			return false
		}
	}
	return true
}

// FoldIndic normalises Devanagari spelling variants: NFD, drop nukta and zero-width joiners, NFC.
// NFD first so precomposed nukta letters (U+0929, U+0958–U+095F) fold as well.
func FoldIndic(s string) string {
	s = strings.Map(func(r rune) rune {
		switch r {
		case nukta, zwj, zwnj:
			return -1
		}
		return r
	}, norm.NFD.String(s))
	return norm.NFC.String(s)
}

// Segment splits text into raw word runs (not lowercased, not stemmed) with rune offsets.
// A run is a maximal sequence of letters, numbers, combining marks and joiners, so Devanagari
// matras stay inside their word. ASCII runs are further split with the legacy pattern.
func Segment(text string) []Span {
	runes := []rune(text)
	var out []Span
	for i := 0; i < len(runes); {
		if !isWordRune(runes[i]) {
			i++
			continue
		}
		j := i
		for j < len(runes) && isWordRune(runes[j]) {
			j++
		}
		run := string(runes[i:j])
		if isASCII(run) {
			for _, loc := range asciiWord.FindAllStringIndex(run, -1) {
				out = append(out, Span{Term: run[loc[0]:loc[1]], Start: i + loc[0], End: i + loc[1]})
			}
		} else {
			out = append(out, Span{Term: run, Start: i, End: j})
		}
		i = j
	}
	return out
}
