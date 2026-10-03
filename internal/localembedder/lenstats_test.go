//go:build cgo

package localembedder

import (
	"bufio"
	"os"
	"sort"
	"strings"
	"testing"
)

// TestMSMARCOLengthStats reports the WordPiece token-length distribution of
// the first 100k MS MARCO passages, and the average padded length for
// unsorted vs length-sorted batches of 64. Skipped unless the benchmark
// corpus cache is present. Diagnostic, not a correctness test.
func TestMSMARCOLengthStats(t *testing.T) {
	f, err := os.Open(`..\..\bench\.cache\collection.tsv`)
	if err != nil {
		t.Skipf("no corpus cache: %v", err)
	}
	defer f.Close()

	tok, err := newTokenizerFromBytes(vocabBytes, false)
	if err != nil {
		t.Fatal(err)
	}

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	lens := make([]int, 0, 100_000)
	for sc.Scan() && len(lens) < 100_000 {
		parts := strings.SplitN(sc.Text(), "\t", 2)
		if len(parts) != 2 {
			continue
		}
		n := len(tok.encodeIDs(parts[1], maxLen))
		lens = append(lens, seqLenFor(n))
	}

	batchAvg := func(ls []int, batch int) float64 {
		var total int
		for i := 0; i < len(ls); i += batch {
			end := min(i+batch, len(ls))
			longest := 0
			for _, l := range ls[i:end] {
				longest = max(longest, l)
			}
			total += longest * (end - i)
		}
		return float64(total) / float64(len(ls))
	}

	unsorted := batchAvg(lens, 64)
	sorted := append([]int(nil), lens...)
	sort.Ints(sorted)
	sortedAvg := batchAvg(sorted, 64)

	sum := 0
	for _, l := range lens {
		sum += l
	}
	p := func(q float64) int { return sorted[int(q*float64(len(sorted)-1))] }
	t.Logf("passages=%d meanSeq=%.1f p50=%d p95=%d p99=%d max=%d", len(lens), float64(sum)/float64(len(lens)), p(0.5), p(0.95), p(0.99), sorted[len(sorted)-1])
	t.Logf("avg padded len: unsorted batches=%.1f  length-sorted batches=%.1f  (ratio %.2fx)", unsorted, sortedAvg, unsorted/sortedAvg)
}
