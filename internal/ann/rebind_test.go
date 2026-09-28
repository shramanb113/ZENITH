package ann

import (
	"testing"
)

// The engine moves vectors between memory (heap -> a new mmap on flush, old mmap
// -> merged mmap on compaction). Rebinding must keep search results identical,
// must point nodes at the new copies (so old mappings can be released), and
// must fail loudly rather than leave a node without a vector.
func TestRebind_KeepsResultsAndAdoptsNewCopies(t *testing.T) {
	c := buildCorpus(2000, 32, 7)
	qs := queries(c, 30, 8)
	before := make([][]Hit, len(qs))
	for i, q := range qs {
		before[i] = c.idx.Search(q, 10, 64, nil)
	}

	// New homes: independent copies of every vector.
	moved := make(map[uint64][]uint16, len(c.vecs))
	for id, v := range c.vecs {
		moved[id] = append([]uint16(nil), v...)
	}
	if err := c.idx.Rebind(func(id uint64) []uint16 { return moved[id] }); err != nil {
		t.Fatal(err)
	}
	// Prove the graph now reads the new copies: corrupt the ORIGINALS and search.
	for _, v := range c.vecs {
		for i := range v {
			v[i] = 0
		}
	}
	for i, q := range qs {
		got := c.idx.Search(q, 10, 64, nil)
		if len(got) != len(before[i]) {
			t.Fatalf("query %d: %d hits after rebind, %d before", i, len(got), len(before[i]))
		}
		for j := range got {
			if got[j] != before[i][j] {
				t.Fatalf("query %d rank %d: %+v after rebind, %+v before — the graph still reads the old memory", i, j, got[j], before[i][j])
			}
		}
	}
}

func TestRebindSome_OnlyTouchesTheGivenIDs(t *testing.T) {
	c := buildCorpus(500, 32, 3)
	ids := []uint64{}
	for id := range c.vecs {
		ids = append(ids, id)
		if len(ids) == 10 {
			break
		}
	}
	replaced := map[uint64][]uint16{}
	for _, id := range ids {
		replaced[id] = append([]uint16(nil), c.vecs[id]...)
	}
	if err := c.idx.RebindSome(append(ids, 1<<60 /* not in the graph: ignored */), func(id uint64) []uint16 { return replaced[id] }); err != nil {
		t.Fatal(err)
	}
	for _, id := range ids {
		n := c.idx.idx[id]
		if &c.idx.vecs[n][0] != &replaced[id][0] {
			t.Fatalf("node for id %d was not repointed", id)
		}
	}
	// A node outside the set keeps its original slice.
	for id, n := range c.idx.idx {
		if _, ok := replaced[id]; ok {
			continue
		}
		if &c.idx.vecs[n][0] != &c.vecs[id][0] {
			t.Fatalf("node for id %d was repointed but was not requested", id)
		}
		break
	}
	// A requested ID with no vector is an error (the engine then rebuilds).
	if err := c.idx.RebindSome(ids[:1], func(uint64) []uint16 { return nil }); err == nil {
		t.Fatal("RebindSome accepted a missing vector")
	}
	if err := c.idx.Rebind(func(uint64) []uint16 { return nil }); err == nil {
		t.Fatal("Rebind accepted a missing vector")
	}
}
