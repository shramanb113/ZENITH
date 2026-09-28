package index

import (
	"bufio"
	"hash/fnv"
	"os"
	"strings"
	"testing"
)

// Shared MS MARCO cache readers for the diagnostics and the real-corpus
// lexical benchmarks. The dataset lives in bench/.cache (see bench/cmd/fetch).

func internalIDOf(originalID string) uint64 {
	h := fnv.New64a()
	h.Write([]byte(originalID))
	return h.Sum64()
}

type diagQuery struct {
	id   string
	text string
}

// loadMSMARCO mirrors bench/internal/corpus.Load for the base corpus (first
// nDocs passages, qrels as map[qid]=pid, last wins) and additionally collects
// the ground-truth passages that fall outside the first nDocs so every qrel
// query becomes evaluable.
func loadMSMARCO(t *testing.T, dir string, nDocs int) (map[string]string, map[string]string, []diagQuery, map[string]string) {
	t.Helper()
	qrels := make(map[string]string)
	gtSet := make(map[string]struct{})
	scanFile(t, dir+`\qrels.dev.small.tsv`, func(parts []string) bool {
		if len(parts) >= 3 {
			qrels[parts[0]] = parts[2]
			gtSet[parts[2]] = struct{}{}
		}
		return true
	})

	passages := make(map[string]string, nDocs)
	augmented := make(map[string]string, len(gtSet))
	found := 0
	scanFile(t, dir+`\collection.tsv`, func(parts []string) bool {
		if len(parts) >= 2 {
			if len(passages) < nDocs {
				passages[parts[0]] = parts[1]
				if _, isGT := gtSet[parts[0]]; isGT {
					found++
				}
			} else if _, isGT := gtSet[parts[0]]; isGT {
				augmented[parts[0]] = parts[1]
				found++
			}
		}
		return found < len(gtSet)
	})

	var queries []diagQuery
	scanFile(t, dir+`\queries.dev.small.tsv`, func(parts []string) bool {
		if len(parts) >= 2 {
			if pid, ok := qrels[parts[0]]; ok {
				_, inBase := passages[pid]
				_, inAug := augmented[pid]
				if inBase || inAug {
					queries = append(queries, diagQuery{id: parts[0], text: parts[1]})
				}
			}
		}
		return true
	})
	return passages, augmented, queries, qrels
}

func scanFile(t *testing.T, path string, fn func(parts []string) bool) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Skipf("corpus cache missing: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		if !fn(strings.Split(sc.Text(), "\t")) {
			break
		}
	}
}
