package wal

import (
	"context"
	"errors"
	"fmt"
	"syscall"
	"testing"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// A refused append (full disk) must not be acknowledged, must leave no torn
// bytes behind, and must not poison the log for later appends.
func TestAppend_DiskFullIsCutBackAndLogKeepsWorking(t *testing.T) {
	df := fsx.NewDiskFull(-1)
	restore := fsx.SetHooks(df)
	defer restore()
	w, path := openTestWAL(t)
	appendKeys(t, w, "a", "b")
	sizeBefore := w.Size()

	for _, budget := range []int64{0, 5, 30} {
		df.SetBudget(budget)
		_, err := w.Append(context.Background(), &Record{Op: OpTypePut, Key: []byte("refused"), Value: []byte("xxxxxxxxxxxxxxxxxxxxxxxx")})
		if err == nil || !errors.Is(err, syscall.ENOSPC) {
			t.Fatalf("budget %d: Append error = %v, want ENOSPC", budget, err)
		}
		recs := []*Record{
			{Op: OpTypePut, Key: []byte("r1"), Value: []byte("v")},
			{Op: OpTypePut, Key: []byte("r2"), Value: []byte("v")},
		}
		if _, err := w.AppendBatch(context.Background(), recs); err == nil || !errors.Is(err, syscall.ENOSPC) {
			t.Fatalf("budget %d: AppendBatch error = %v, want ENOSPC", budget, err)
		}
		if w.Size() != sizeBefore {
			t.Fatalf("budget %d: log size %d after refused appends, want %d", budget, w.Size(), sizeBefore)
		}
	}

	df.SetBudget(-1)
	appendKeys(t, w, "c")
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	w2, recs, err := OpenWAL(path, WALConfig{SyncMode: SyncAlways})
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()
	if got := fmt.Sprint(keysOf(recs)); got != "[a b c]" {
		t.Fatalf("recovered %s, want [a b c]", got)
	}
}
