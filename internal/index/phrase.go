package index

import (
	"encoding/json"
	"sort"
	"strings"
)

// Phrase queries.
//
// A double-quoted part of a query — `"machine learning"` — is a phrase: every
// result must contain the phrase's analysed terms at consecutive positions, in
// order. A phrase is a required clause (as in web search engines and Lucene's
// "+"): documents without it are dropped from the lexical, vector and
// neural-expansion candidate sets alike. The rest of the query, and the
// phrase's own words, still score and rank results exactly as an unquoted
// query would; the phrase only decides which documents may appear.
//
// Positions are the indices of the analyser's output tokens (Token.Position
// for StandardAnalyzer, which numbers kept tokens 0..n-1), so a phrase is
// compared after the same lowercasing, stop-word removal and stemming as the
// documents: `"Machine-Learning"` matches "machine learning" and "machines
// learned", and a removed stop word holds no position (`"art of war"` matches
// "art of war" and also "art, war").
//
// Quotes pair up left to right; an unpaired final quote is ordinary text (the
// analyser drops punctuation anyway). A phrase that analyses to no terms — only
// stop words or punctuation — constrains nothing and is ignored.

// extractPhrases returns the text between each pair of double quotes in query.
func extractPhrases(query string) []string {
	var out []string
	for {
		i := strings.IndexByte(query, '"')
		if i < 0 {
			return out
		}
		j := strings.IndexByte(query[i+1:], '"')
		if j < 0 {
			return out // unpaired quote: plain text
		}
		if p := query[i+1 : i+1+j]; strings.TrimSpace(p) != "" {
			out = append(out, p)
		}
		query = query[i+1+j+1:]
	}
}

// analyzePhrases returns the analysed terms of each phrase in query, in
// position order, using the indexing analysis (no synonyms, no FST
// resolution) so they are exactly the terms a matching document holds. nil
// when the query has no phrase that constrains anything.
func (e *Engine) analyzePhrases(query string) [][]string {
	raw := extractPhrases(query)
	if len(raw) == 0 {
		return nil
	}
	var out [][]string
	for _, p := range raw {
		if terms := e.analyzedTerms(p); len(terms) > 0 {
			out = append(out, terms)
		}
	}
	return out
}

// analyzedTerms is the term sequence the analyser produces for text; a term's
// index is its position.
func (e *Engine) analyzedTerms(text string) []string {
	toks := e.analyzer.Analyze(text)
	terms := make([]string, len(toks))
	for i, t := range toks {
		terms[i] = t.Term
	}
	return terms
}

// phraseSignature canonically encodes a query's phrase constraints ("" when
// there are none). The query-result cache mixes it into the bucket key so the
// semantic near-duplicate scan never answers a phrase query with the results of
// a query that has different (or no) phrase constraints — their embeddings are
// nearly identical, their result sets are not.
func (e *Engine) phraseSignature(query string) string {
	phrases := e.analyzePhrases(query)
	if len(phrases) == 0 {
		return ""
	}
	b, _ := json.Marshal(phrases) // [][]string cannot fail to marshal
	return string(b)
}

// containsPhrase reports whether phrase occurs in seq at consecutive positions.
func containsPhrase(seq, phrase []string) bool {
	if len(phrase) == 0 {
		return true
	}
outer:
	for i := 0; i+len(phrase) <= len(seq); i++ {
		for k, t := range phrase {
			if seq[i+k] != t {
				continue outer
			}
		}
		return true
	}
	return false
}

// phraseDocs returns the live documents that contain every phrase and that
// pred accepts (nil accepts all). Engine.mu held (read or write); the caller
// must not hold e.inverted's lock.
func (e *Engine) phraseDocs(phrases [][]string, pred Predicate) map[uint64]struct{} {
	var out map[uint64]struct{}
	for _, ph := range phrases {
		out = e.docsWithPhrase(ph, pred, out)
		if len(out) == 0 {
			return map[uint64]struct{}{}
		}
	}
	return out
}

// docsWithPhrase returns the documents (within the set `within`, when it is
// non-nil) that pred accepts and that contain phrase.
func (e *Engine) docsWithPhrase(phrase []string, pred Predicate, within map[uint64]struct{}) map[uint64]struct{} {
	cands := e.docsWithAllTerms(phrase, pred, within)
	if len(phrase) == 1 {
		return cands // a one-term phrase is just a required exact term
	}
	for id := range cands {
		text, ok := e.textOf(id)
		if !ok || !containsPhrase(e.analyzedTerms(text), phrase) {
			delete(cands, id)
		}
	}
	return cands
}

// docsWithAllTerms returns the live documents containing every one of terms
// exactly (the BM25 vocabulary, not n-gram prefixes), within `within` when it
// is non-nil and accepted by pred. It walks the rarest term's postings first,
// so the candidate set only ever shrinks.
func (e *Engine) docsWithAllTerms(terms []string, pred Predicate, within map[uint64]struct{}) map[uint64]struct{} {
	distinct := dedupe(terms)
	back := segBacking{e}
	df := make(map[string]int, len(distinct))
	for _, t := range distinct {
		df[t] = e.bm25.LocalDocFreq(t) + back.DocFreq(t)
	}
	sort.SliceStable(distinct, func(i, j int) bool { return df[distinct[i]] < df[distinct[j]] })

	set := make(map[uint64]struct{})
	e.eachTermDoc(distinct[0], func(id uint64) {
		if within != nil {
			if _, ok := within[id]; !ok {
				return
			}
		}
		if pred != nil && !pred(e.attrs[id]) {
			return
		}
		set[id] = struct{}{}
	})
	for _, t := range distinct[1:] {
		if len(set) == 0 {
			break
		}
		next := make(map[uint64]struct{}, len(set))
		e.eachTermDoc(t, func(id uint64) {
			if _, ok := set[id]; ok {
				next[id] = struct{}{}
			}
		})
		set = next
	}
	return set
}

// eachTermDoc calls fn with every live document whose BM25 term list contains
// term, across the delta, the frozen layer and every segment.
func (e *Engine) eachTermDoc(term string, fn func(id uint64)) {
	e.bm25.EachLocalPosting(term, func(id uint64, _, _ int) { fn(id) })
	segBacking{e}.EachPosting(term, func(id uint64, _, _ int) { fn(id) })
}

// keepOnly deletes every entry of scores whose document is not in allowed.
func keepOnly(scores map[uint64]float64, allowed map[uint64]struct{}) {
	for id := range scores {
		if _, ok := allowed[id]; !ok {
			delete(scores, id)
		}
	}
}
