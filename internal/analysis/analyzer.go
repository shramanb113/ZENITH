package analysis

import (
	"strings"

	"github.com/kljensen/snowball"
)

// TokenType classifies what kind of token was produced.
type TokenType int

const (
	WORD     TokenType = iota
	NGRAM              // produced by edge n-gram expansion
	PHONETIC           // produced by Soundex/Metaphone
	SYNONYM            // produced by synonym expansion
)

// Token is a single unit of analysis with position and type metadata.
type Token struct {
	Term     string
	Position int
	Type     TokenType
}

// Analyzer is the interface the Engine depends on.
type Analyzer interface {
	Analyze(text string) []Token
}

// FSTWirer is implemented by analyzers that accept an FSTDictionary for
// query-time term resolution. index.Engine uses this interface to wire the
// rebuilt FST into the analyzer after every batch of documents, enabling
// prefix-based term resolution without importing a concrete type.
type FSTWirer interface {
	SetFST(fst *FSTDictionary)
}

// QueryAnalyzer extends Analyzer with a synonym-aware query path.
// index.Engine uses this interface so that AnalyzeQuery (which expands
// synonyms) is called at search time instead of the indexing-only Analyze.
// StandardAnalyzer implements both FSTWirer and QueryAnalyzer.
type QueryAnalyzer interface {
	Analyzer
	AnalyzeQuery(query string) []Token
}

// StandardAnalyzer is the default analysis pipeline:
//  1. Unicode-aware segmentation (Segment); ASCII runs keep the camelCase-aware pattern.
//  2. Lowercasing
//  3. Stop-word removal
//  4. Snowball Porter2 stemming for ASCII tokens only; Devanagari is FoldIndic-normalised.
//
// FST integration:
//   - The analyzer holds a reference to the shared FSTDictionary.
//   - After stemming, each token is checked against the FST.
//   - If the FST is built and the stemmed token IS found: used as-is.
//   - If the FST is built and the token is NOT found: prefix-expand via
//     PrefixSearch to find the closest indexed term. This handles cases
//     where a query contains an un-indexed stem that is a prefix of
//     something that was indexed (e.g. query "kubern" → index has "kubernet").
//   - If the FST is not built yet: fall through silently (no-op).
//
// Synonym expansion:
//   - Applied at query time via AnalyzeQuery() — NOT in Analyze().
//   - Indexing with synonyms bloats the index and breaks IDF weights.
type StandardAnalyzer struct {
	stopWords map[string]struct{}
	fst       *FSTDictionary // may be nil if not wired
}

// NewStandardAnalyzer constructs a StandardAnalyzer without FST.
// Call SetFST() after building the index to enable FST-assisted lookup.
func NewStandardAnalyzer() *StandardAnalyzer {
	stopList := []string{
		"a", "about", "above", "after", "again", "against", "all", "am",
		"an", "and", "any", "are", "as", "at", "be", "because", "been",
		"before", "being", "below", "between", "both", "but", "by", "can",
		"did", "do", "does", "doing", "don", "down", "during", "each",
		"few", "for", "from", "further", "had", "has", "have", "having",
		"he", "her", "here", "hers", "herself", "him", "himself", "his",
		"how", "i", "if", "in", "into", "is", "it", "its", "itself",
		"just", "me", "more", "most", "my", "myself", "no", "nor", "not",
		"now", "of", "off", "on", "once", "only", "or", "other", "our",
		"ours", "ourselves", "out", "over", "own", "s", "same", "she",
		"should", "so", "some", "such", "t", "than", "that", "the",
		"their", "theirs", "them", "themselves", "then", "there", "these",
		"they", "this", "those", "through", "to", "too", "under", "until",
		"up", "very", "was", "we", "were", "what", "when", "where",
		"which", "while", "who", "whom", "why", "will", "with", "you",
		"your", "yours", "yourself", "yourselves",
		// Hindi / Hinglish function words (Devanagari entries are already FoldIndic-normal).
		"hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "aur",
		"है", "हैं", "का", "की", "के", "को", "से", "में", "और",
	}

	stopMap := make(map[string]struct{}, len(stopList))
	for _, s := range stopList {
		stopMap[s] = struct{}{}
	}

	return &StandardAnalyzer{
		stopWords: stopMap,
	}
}

// SetFST wires the FST dictionary into the analyzer.
// Call this after engine.Load() or after the first FST.Build() completes.
// Thread-safe write — engine must not be serving queries during this call.
func (a *StandardAnalyzer) SetFST(fst *FSTDictionary) {
	a.fst = fst
}

// stem calls kljensen/snowball's English Porter2 stemmer.
// Returns the original word unchanged on error.
func stem(word string) string {
	s, err := snowball.Stem(word, "english", true)
	if err != nil {
		return word
	}
	return s
}

// resolveTerm applies FST-assisted term resolution on a stemmed token.
// If the FST is not built, returns the token unchanged.
// If the exact stem is found in the FST, returns it unchanged.
// If not found, attempts prefix search to find the closest indexed term.
// Falls back to the original stem if prefix search yields nothing.
func (a *StandardAnalyzer) resolveTerm(stemmed string) string {
	if a.fst == nil || !a.fst.IsBuilt() {
		return stemmed
	}
	if a.fst.Contains(stemmed) {
		return stemmed
	}
	// Prefix search: find the first indexed term that starts with this stem.
	// This handles query stems that are shorter than their indexed counterparts.
	matches, _ := a.fst.PrefixSearch(stemmed, 1)
	if len(matches) > 0 {
		return matches[0]
	}
	return stemmed
}

// analyseSpans is the single analysis path: segment → lowercase → (ASCII: stop, stem) /
// (other scripts: FoldIndic, stop) → optional FST resolution. Pos numbers the kept tokens.
func (a *StandardAnalyzer) analyseSpans(text string, resolve bool) []Span {
	raw := Segment(text)
	out := make([]Span, 0, len(raw))
	for _, sp := range raw {
		var term string
		if isASCII(sp.Term) {
			term = strings.ToLower(sp.Term)
			if _, stop := a.stopWords[term]; stop {
				continue
			}
			term = stem(term) // Porter2 is English-only
		} else {
			term = FoldIndic(strings.ToLower(sp.Term))
			if _, stop := a.stopWords[term]; stop {
				continue
			}
		}
		if term == "" {
			continue
		}
		if resolve {
			term = a.resolveTerm(term)
		}
		sp.Term, sp.Pos = term, len(out)
		out = append(out, sp)
	}
	return out
}

func spanTerms(spans []Span) []string {
	if len(spans) == 0 {
		return nil
	}
	out := make([]string, len(spans))
	for i, s := range spans {
		out[i] = s.Term
	}
	return out
}

// Tokenize returns stemmed, filtered string tokens from text, with FST prefix resolution when the
// FST is built. Used at index time and by the BM25/TF-IDF scorers.
func (a *StandardAnalyzer) Tokenize(text string) []string {
	return spanTerms(a.analyseSpans(text, true))
}

// TokenizeExact is Tokenize without FST prefix resolution: the exact analysed terms. Explain uses it
// so a query term never silently becomes a longer indexed term ("fire" → "firework").
func (a *StandardAnalyzer) TokenizeExact(text string) []string {
	return spanTerms(a.analyseSpans(text, false))
}

// AnalyzeSpans returns the exact analysed terms (no FST resolution) with rune offsets into text.
// On a namespace indexed in one batch (FST not yet built at index time) these terms equal the
// indexed terms, so a TermMatch.Matched value can be located in the original document.
func (a *StandardAnalyzer) AnalyzeSpans(text string) []Span {
	return a.analyseSpans(text, false)
}

// Analyze implements the Analyzer interface.
// Returns structured Tokens with position and type metadata.
// Does NOT expand synonyms — use AnalyzeQuery for query-time expansion.
func (a *StandardAnalyzer) Analyze(text string) []Token {
	terms := a.Tokenize(text)
	tokens := make([]Token, len(terms))
	for i, t := range terms {
		tokens[i] = Token{
			Term:     t,
			Position: i,
			Type:     WORD,
		}
	}
	return tokens
}

// AnalyzeQuery is the query-time counterpart of Analyze.
// It applies the full pipeline plus synonym expansion.
// Never call this during indexing — only for incoming search queries.
//
// Pipeline: tokenise → lowercase → stop-word filter → stem → FST resolve → synonym expand
func (a *StandardAnalyzer) AnalyzeQuery(query string) []Token {
	baseTokens := a.Tokenize(query)
	expanded := ExpandWithSynonyms(baseTokens)

	tokens := make([]Token, 0, len(expanded))
	baseSet := make(map[string]struct{}, len(baseTokens))
	for _, t := range baseTokens {
		baseSet[t] = struct{}{}
	}

	pos := 0
	for _, term := range expanded {
		typ := WORD
		if _, isBase := baseSet[term]; !isBase {
			typ = SYNONYM
		}
		tokens = append(tokens, Token{
			Term:     term,
			Position: pos,
			Type:     typ,
		})
		pos++
	}
	return tokens
}
