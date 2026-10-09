package index

import "strings"

// DefaultSuggestLimit and MaxSuggestLimit bound Suggest's result count.
const (
	DefaultSuggestLimit = 10
	MaxSuggestLimit     = 1000
)

// Suggest returns up to n indexed terms starting with prefix, for
// autocomplete. n <= 0 means DefaultSuggestLimit; n is capped at
// MaxSuggestLimit. An empty (or all-space) prefix returns nothing.
//
// This is an honestly-scoped v1, read from the engine's FST vocabulary:
//   - terms are the *analysed* vocabulary — lowercased, stop words removed
//     and Porter2-stemmed — so "running" is suggested as "run", and a prefix
//     that runs past a stem ("runn") finds nothing. The prefix itself is only
//     lowercased and trimmed, never stemmed (stemming a partial word would
//     mangle it).
//   - order is lexicographic, not by frequency: the FST stores no per-term
//     counts today.
//   - the FST is rebuilt lazily after writes, so a term added or removed a
//     moment ago may not be reflected yet.
func (e *Engine) Suggest(prefix string, n int) ([]string, error) {
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	if prefix == "" {
		return []string{}, nil
	}
	if n <= 0 {
		n = DefaultSuggestLimit
	}
	if n > MaxSuggestLimit {
		n = MaxSuggestLimit
	}
	terms, err := e.FSTPrefixSearch(prefix, n)
	if err != nil {
		return nil, err
	}
	if terms == nil {
		terms = []string{}
	}
	return terms, nil
}
