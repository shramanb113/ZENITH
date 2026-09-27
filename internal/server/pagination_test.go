package server

import (
	"testing"

	"github.com/shramanb113/ZENITH/internal/index"
)

func makeResults(n int) []index.SearchResponse {
	results := make([]index.SearchResponse, n)
	for i := range results {
		results[i] = index.SearchResponse{ID: string(rune('a' + i))}
	}
	return results
}

func TestPaginate_DefaultLimitWhenUnset(t *testing.T) {
	got := paginate(makeResults(40), 0, 0)
	if len(got) != defaultSearchLimit {
		t.Fatalf("got %d results, want default limit %d", len(got), defaultSearchLimit)
	}
}

func TestPaginate_LimitAboveDefault(t *testing.T) {
	got := paginate(makeResults(40), 0, 30)
	if len(got) != 30 {
		t.Fatalf("got %d results, want 30", len(got))
	}
}

func TestPaginate_LimitLargerThanResultSet(t *testing.T) {
	got := paginate(makeResults(5), 0, 30)
	if len(got) != 5 {
		t.Fatalf("got %d results, want 5 (all available)", len(got))
	}
}

func TestPaginate_Offset(t *testing.T) {
	all := makeResults(40)
	got := paginate(all, 10, 10)
	if len(got) != 10 {
		t.Fatalf("got %d results, want 10", len(got))
	}
	if got[0].ID != all[10].ID {
		t.Fatalf("offset 10 should start at %q, got %q", all[10].ID, got[0].ID)
	}
}

func TestPaginate_OffsetPastEnd(t *testing.T) {
	got := paginate(makeResults(5), 10, 10)
	if len(got) != 0 {
		t.Fatalf("got %d results, want 0 for offset past end", len(got))
	}
}

func TestPaginate_NegativeOffsetTreatedAsZero(t *testing.T) {
	all := makeResults(5)
	got := paginate(all, -3, 10)
	if len(got) != 5 || got[0].ID != all[0].ID {
		t.Fatalf("negative offset should behave like 0, got %d results starting %q", len(got), got[0].ID)
	}
}
