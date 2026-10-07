// internal/analysis/segment.go
package analysis

import (
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

// splitASCIIWord splits an ASCII word run into camelCase/acronym-aware sub-tokens.
//
// A maximal run of uppercase letters is an acronym ("HTTP", "NASA"), except that
// when it is immediately followed by a lowercase letter, the LAST uppercase letter
// starts a new Capitalized word instead of belonging to the acronym — this is how
// acronym+CamelCase identifiers are conventionally written ("HTTPServer" ->
// "HTTP","Server", not five single letters). Lowercase letters and digits share a
// class so digits stay attached to the word they're adjacent to ("v6" in "IPv6").
//
// Go's RE2 engine has no lookahead, so a single regexp can't express "a run of
// capitals that isn't immediately followed by a lowercase letter" — hence a
// hand-written scanner instead of the old `[A-Z][a-z0-9]*|[a-z0-9]+|[A-Z]+`
// pattern (whose first alternative always won on a single capital, so the
// `[A-Z]+` acronym alternative could never match and "HTTP"/"NASA"/"API" were
// shredded into single dropped letters).
func splitASCIIWord(run string) [][2]int {
	n := len(run)
	isUpper := func(i int) bool { return run[i] >= 'A' && run[i] <= 'Z' }
	isLowerOrDigit := func(i int) bool {
		return (run[i] >= 'a' && run[i] <= 'z') || (run[i] >= '0' && run[i] <= '9')
	}

	var out [][2]int
	for i := 0; i < n; {
		if isLowerOrDigit(i) {
			j := i + 1
			for j < n && isLowerOrDigit(j) {
				j++
			}
			out = append(out, [2]int{i, j})
			i = j
			continue
		}

		// isUpper(i) holds: run only contains ASCII letters and digits.
		j := i + 1
		for j < n && isUpper(j) {
			j++
		}
		upperLen := j - i

		switch {
		case upperLen == 1:
			// Single capital: consume a following lower/digit run with it
			// as one Capitalized word ("Case", "Server").
			k := j
			for k < n && isLowerOrDigit(k) {
				k++
			}
			out = append(out, [2]int{i, k})
			i = k
		case j < n && run[j] >= 'a' && run[j] <= 'z':
			// Acronym run followed by a lowercase letter: the last capital
			// starts the next Capitalized word ("HTTPServer" -> "HTTP","Server").
			out = append(out, [2]int{i, j - 1})
			i = j - 1
		default:
			// Acronym run not followed by a lowercase letter (end of run,
			// or a digit): keep the whole run together ("NASA", "API").
			out = append(out, [2]int{i, j})
			i = j
		}
	}
	return out
}

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

func isASCIIRunes(rs []rune) bool {
	for _, r := range rs {
		if r >= 0x80 {
			return false
		}
	}
	return true
}

// isCJK reports whether r belongs to a script written with no whitespace
// between words (Han, Hiragana, Katakana, Hangul). Segment splits these into
// one span per character, since there is no dictionary segmenter in this
// module (deliberately not added — see CLAUDE.md) and a maximal run would
// otherwise become a single oversized token covering a whole sentence.
// Other non-ASCII scripts (Devanagari, Arabic, Cyrillic, ...) are
// whitespace-delimited already, so a run stays one span there.
func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) || unicode.Is(unicode.Hangul, r)
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
// matras stay inside their word. ASCII runs are further split with the legacy pattern; CJK
// characters inside a non-ASCII run are split one-per-span (see isCJK).
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
		run := runes[i:j]
		switch {
		case isASCIIRunes(run):
			// Materialise the string only now: slicing a string shares its
			// backing array, so this is the one allocation this branch
			// needs, not one per candidate run.
			s := string(run)
			for _, loc := range splitASCIIWord(s) {
				out = append(out, Span{Term: s[loc[0]:loc[1]], Start: i + loc[0], End: i + loc[1]})
			}
		default:
			segStart := i
			for k := i; k < j; k++ {
				if !isCJK(runes[k]) {
					continue
				}
				if k > segStart {
					out = append(out, Span{Term: string(runes[segStart:k]), Start: segStart, End: k})
				}
				out = append(out, Span{Term: string(runes[k : k+1]), Start: k, End: k + 1})
				segStart = k + 1
			}
			if segStart < j {
				out = append(out, Span{Term: string(runes[segStart:j]), Start: segStart, End: j})
			}
		}
		i = j
	}
	return out
}
