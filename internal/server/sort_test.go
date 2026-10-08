package server_test

import (
	"context"
	"testing"

	"github.com/shramanb113/ZENITH/gen/go/zenithproto"
)

// SortRequest's sort_field replaces score ordering entirely: en2020 (2020),
// fr2022 (2022) and en2024 (2024) all match "kubernetes networking" equally
// well lexically, so without sort_field the order is whatever the hybrid
// ranker's tiebreak gives; with it, it's exactly the requested year order,
// "plain" (no year attr) sorting last regardless of direction.
func TestGRPC_SearchSortFieldReplacesScoreOrder(t *testing.T) {
	s := newSrv(t)
	seed(t, s)

	asc, err := s.Search(context.Background(), &zenithproto.SearchRequest{
		Query: "kubernetes networking", Limit: 50, SortField: "year", SortDesc: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantAsc := []string{"en2020", "fr2022", "en2024", "plain"}
	if got := idsInOrder(asc); !idsEqual(got, wantAsc) {
		t.Fatalf("sort_field=year asc: got %v, want %v", got, wantAsc)
	}

	desc, err := s.Search(context.Background(), &zenithproto.SearchRequest{
		Query: "kubernetes networking", Limit: 50, SortField: "year", SortDesc: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	wantDesc := []string{"en2024", "fr2022", "en2020", "plain"}
	if got := idsInOrder(desc); !idsEqual(got, wantDesc) {
		t.Fatalf("sort_field=year desc: got %v, want %v", got, wantDesc)
	}
}

func idsInOrder(r *zenithproto.SearchResponse) []string {
	out := make([]string, len(r.GetResults()))
	for i, x := range r.GetResults() {
		out[i] = x.GetId()
	}
	return out
}

func idsEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
