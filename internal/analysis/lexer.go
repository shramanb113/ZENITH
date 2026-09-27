package analysis

import (
	"strings"
	"unicode"

	"github.com/kljensen/snowball"
)

// Lexer tokenizes text the same way seroost does:
//   - skip whitespace
//   - numeric runs  → token as-is
//   - alphabetic runs → lowercased → Snowball Porter2 stemmed
//   - single punctuation chars → token as-is (filtered by CollectAlpha)
//
// Uses kljensen/snowball directly — same library as StandardAnalyzer —
// so stemming output is identical across both tokenization paths.
//
// Usage:
//
//	tokens := NewLexer(text).CollectAlpha()
type Lexer struct {
	runes []rune
	pos   int
}

// NewLexer creates a Lexer over text.
func NewLexer(text string) *Lexer {
	return &Lexer{runes: []rune(text)}
}

// HasNext returns true if there are more tokens remaining.
func (l *Lexer) HasNext() bool {
	l.trimLeft()
	return l.pos < len(l.runes)
}

// Next returns the next token.
// Alphabetic runs → lowercased + Snowball Porter2 stemmed.
// Numeric runs → returned as-is.
// Single symbol → returned as-is (filter downstream with CollectAlpha).
// Returns "" when exhausted.
func (l *Lexer) Next() string {
	l.trimLeft()
	if l.pos >= len(l.runes) {
		return ""
	}

	ch := l.runes[l.pos]

	if unicode.IsNumber(ch) {
		return l.chopWhile(unicode.IsNumber)
	}

	if unicode.IsLetter(ch) {
		raw := l.chopWhile(func(r rune) bool {
			return unicode.IsLetter(r) || unicode.IsNumber(r) || unicode.IsMark(r)
		})
		lower := strings.ToLower(raw)
		if !isASCII(lower) {
			return FoldIndic(lower)
		}
		stemmed, err := snowball.Stem(lower, "english", true)
		if err != nil {
			return lower
		}
		return stemmed
	}

	l.pos++
	return string(ch)
}

// Collect drains the Lexer and returns every token including punctuation.
func (l *Lexer) Collect() []string {
	var tokens []string
	for l.HasNext() {
		if tok := l.Next(); tok != "" {
			tokens = append(tokens, tok)
		}
	}
	return tokens
}

// CollectAlpha returns only meaningful tokens (length > 1, or single letter).
// No lone punctuation. This is what TFIDFScorer and BM25Scorer consume.
func (l *Lexer) CollectAlpha() []string {
	var tokens []string
	for l.HasNext() {
		tok := l.Next()
		if tok == "" {
			continue
		}
		r := []rune(tok)
		if len(r) > 1 || (len(r) == 1 && unicode.IsLetter(r[0])) {
			tokens = append(tokens, tok)
		}
	}
	return tokens
}

func (l *Lexer) trimLeft() {
	for l.pos < len(l.runes) && unicode.IsSpace(l.runes[l.pos]) {
		l.pos++
	}
}

func (l *Lexer) chopWhile(pred func(rune) bool) string {
	start := l.pos
	for l.pos < len(l.runes) && pred(l.runes[l.pos]) {
		l.pos++
	}
	return string(l.runes[start:l.pos])
}

// TokenizeString is a one-shot convenience wrapper.
// Returns all meaningful alpha tokens, stemmed and filtered.
func TokenizeString(text string) []string {
	return NewLexer(text).CollectAlpha()
}
