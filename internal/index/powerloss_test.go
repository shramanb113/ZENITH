package index

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// Power-loss and disk-full tests for the flush/compaction commit protocol.
//
// kill -9 (crash_test.go) cannot lose data the OS already holds in its page
// cache. A power failure can: whatever was not fsynced, or whose directory entry
// was not fsynced, may be gone or torn. These tests record every mutating
// filesystem operation of a scenario (internal/fsx), then rebuild the directory
// as it could look after a power cut at every point of the log — with only
// fsynced state, with everything, and with random torn subsets — and check that
// the index opens, passes checksum verification, and holds a state a correct
// engine could have acknowledged.

const plDoc = "pl.db"

func plText(i, ver int) string {
	return fmt.Sprintf("powerloss document%d version%d payload token%d shared", i, ver, i%5)
}

// plState is the model: id -> text.
type plState map[string]string

func (s plState) clone() plState {
	c := make(plState, len(s))
	for k, v := range s {
		c[k] = v
	}
	return c
}

// checkRecovered opens the index at dir and requires its documents to equal
// exactly one of the allowed states. A missing index equals the empty state.
func checkRecovered(t *testing.T, what, dir string, allowed ...plState) {
	t.Helper()
	path := filepath.Join(dir, plDoc)
	e := diffEngine()
	defer e.Close()
	got := plState{}
	if _, err := os.Stat(path); err == nil {
		if err := e.Load(path); err != nil {
			t.Fatalf("%s: index does not open: %v", what, err)
		}
		if err := Verify(path); err != nil {
			t.Fatalf("%s: checksum verification failed: %v", what, err)
		}
		for _, s := range allowed {
			for id := range s {
				if txt, ok := e.GetText(id); ok {
					got[id] = txt
				}
			}
		}
		// Anything else present would be a resurrected or invented document.
		if n := e.Count(); n != len(got) {
			t.Fatalf("%s: index holds %d documents, %d of them known to the model", what, n, len(got))
		}
	}
	for _, s := range allowed {
		if statesEqual(got, s) {
			return
		}
	}
	t.Fatalf("%s: recovered %d documents that match no acknowledged state (allowed: %d states, sizes %v)\n  recovered: %v",
		what, len(got), len(allowed), sizesOf(allowed), summarise(got))
}

func statesEqual(a, b plState) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}

func sizesOf(ss []plState) []int {
	out := make([]int, len(ss))
	for i, s := range ss {
		out[i] = len(s)
	}
	return out
}

func summarise(s plState) string {
	if len(s) > 12 {
		return fmt.Sprintf("%d docs", len(s))
	}
	var b strings.Builder
	for k := range s {
		b.WriteString(k + " ")
	}
	return b.String()
}

// crashPoints yields the operation counts k worth crashing at: after every
// operation. (A crash "between" two operations is a crash after the first.)
func crashPoints(n int) []int {
	ks := make([]int, 0, n+1)
	for k := 0; k <= n; k++ {
		ks = append(ks, k)
	}
	return ks
}

func ackCount(rec *fsx.Recorder, k int, prefix string) int {
	n := 0
	for _, m := range rec.Marks(k) {
		if strings.HasPrefix(m, prefix) {
			n++
		}
	}
	return n
}

// TestPowerLoss_FlushAndCompaction: every prefix of a scenario with flushes,
// deletes, replacements and compactions, under every persistence model.
func TestPowerLoss_FlushAndCompaction(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	rec := fsx.NewRecorder(root)
	restore := fsx.SetHooks(rec)
	defer restore()

	path := filepath.Join(root, plDoc)
	e := diffEngine()
	defer e.Close()

	state := plState{}
	states := []plState{{}} // states[i] = contents after the i-th acknowledged Save
	add := func(i, ver int) {
		id := fmt.Sprintf("d%d", i)
		if err := e.Add(ctx, id, plText(i, ver)); err != nil {
			t.Fatal(err)
		}
		state[id] = plText(i, ver)
	}
	remove := func(i int) {
		id := fmt.Sprintf("d%d", i)
		if err := e.Remove(ctx, id); err != nil {
			t.Fatal(err)
		}
		delete(state, id)
	}
	save := func() {
		t.Helper()
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
		states = append(states, state.clone())
		rec.Mark("ack-save")
	}
	// Both the state a Save was about to publish and the previous one are legal
	// after a crash inside it; record the "about to" state before calling.
	for i := 0; i < 30; i++ {
		add(i, 1)
	}
	save()
	for i := 30; i < 60; i++ {
		add(i, 1)
	}
	for i := 0; i < 10; i++ {
		remove(i)
	}
	save()
	for i := 10; i < 15; i++ {
		add(i, 2) // replace documents that live in the first segment
	}
	for i := 60; i < 70; i++ {
		add(i, 1)
	}
	save()
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	rec.Mark("ack-compact")
	for i := 70; i < 80; i++ {
		add(i, 1)
	}
	for i := 60; i < 65; i++ {
		remove(i)
	}
	save()
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	rec.Mark("ack-compact")
	restore()

	ops := rec.Ops()
	t.Logf("scenario recorded %d operations, %d acknowledged saves", len(ops), len(states)-1)
	if len(ops) < 30 {
		t.Fatalf("only %d operations recorded: the fsx hooks are not seeing the writes", len(ops))
	}

	checked := 0
	for _, k := range crashPoints(len(ops)) {
		acked := ackCount(rec, k, "ack-save")
		allowed := []plState{states[acked]}
		if acked+1 < len(states) {
			allowed = append(allowed, states[acked+1]) // the Save in flight may have committed
		}
		for _, mode := range []struct {
			name string
			p    fsx.Persistence
			seed int64
		}{
			{"nothing-unsynced-survives", fsx.Nothing, 0},
			{"everything-survives", fsx.Everything, 0},
			{"torn-1", fsx.Torn, int64(k)*7 + 1},
			{"torn-2", fsx.Torn, int64(k)*7 + 2},
		} {
			dst := filepath.Join(t.TempDir(), "crash")
			if err := rec.CrashState(k, mode.p, mode.seed, dst); err != nil {
				t.Fatal(err)
			}
			checkRecovered(t, fmt.Sprintf("crash after op %d/%d (%s)", k, len(ops), mode.name), dst, allowed...)
			checked++
		}
	}
	t.Logf("checked %d simulated power cuts", checked)
}

// A directory entry that was never fsynced can vanish even though the manifest
// rename that names it did not: the segment writer therefore syncs the directory
// before the manifest commit. This asserts that ordering directly on the log.
func TestPowerLoss_SegmentDirEntryIsSyncedBeforeManifestRename(t *testing.T) {
	root := t.TempDir()
	rec := fsx.NewRecorder(root)
	restore := fsx.SetHooks(rec)
	defer restore()
	e := diffEngine()
	defer e.Close()
	for i := 0; i < 10; i++ {
		e.Add(context.Background(), fmt.Sprintf("d%d", i), plText(i, 1))
	}
	if err := e.Save(filepath.Join(root, plDoc)); err != nil {
		t.Fatal(err)
	}
	restore()
	var segSync, dirSyncAfterSeg, manifestRename = -1, -1, -1
	for i, op := range rec.Ops() {
		switch {
		case op.Kind == fsx.OpSync && strings.Contains(op.Path, ".seg-"):
			segSync = i
		case op.Kind == fsx.OpSyncDir && segSync >= 0 && dirSyncAfterSeg < 0:
			dirSyncAfterSeg = i
		case op.Kind == fsx.OpRename && op.Path2 == plDoc:
			manifestRename = i
		}
	}
	if !(segSync >= 0 && segSync < dirSyncAfterSeg && dirSyncAfterSeg < manifestRename) {
		t.Fatalf("order must be: segment fsync (%d) < directory fsync (%d) < manifest rename (%d)", segSync, dirSyncAfterSeg, manifestRename)
	}
}

// ---- disk full ----

func isNoSpace(err error) bool { return errors.Is(err, syscall.ENOSPC) }

// A flush that hits a full disk at any byte must fail cleanly: an error, no
// partial segment left behind, the previous on-disk state intact and openable,
// the engine still serving its (unflushed) documents, and — once space is back —
// the next Save succeeding and losing nothing.
func TestDiskFull_FlushFailsCleanlyAtEveryBudget(t *testing.T) {
	ctx := context.Background()
	for _, budget := range []int64{0, 1, 100, 1000, 5000, 20000} {
		t.Run(strconv.FormatInt(budget, 10)+"B", func(t *testing.T) {
			root := t.TempDir()
			df := fsx.NewDiskFull(-1)
			restore := fsx.SetHooks(df)
			defer restore()
			path := filepath.Join(root, plDoc)
			ref, e := diffEngine(), diffEngine()
			defer ref.Close()
			defer e.Close()
			both := func(f func(*Engine) error) {
				t.Helper()
				if err := f(ref); err != nil {
					t.Fatal(err)
				}
				if err := f(e); err != nil {
					t.Fatal(err)
				}
			}
			base := plState{}
			for i := 0; i < 20; i++ {
				i := i
				both(func(x *Engine) error { return x.Add(ctx, fmt.Sprintf("d%d", i), plText(i, 1)) })
				base[fmt.Sprintf("d%d", i)] = plText(i, 1)
			}
			if err := e.Save(path); err != nil {
				t.Fatal(err)
			}
			for i := 20; i < 60; i++ {
				i := i
				both(func(x *Engine) error { return x.Add(ctx, fmt.Sprintf("d%d", i), plText(i, 1)) })
			}
			both(func(x *Engine) error { return x.Remove(ctx, "d3") })

			df.SetBudget(budget)
			err := e.Save(path)
			if err == nil {
				t.Fatalf("Save succeeded with only %d bytes of space", budget)
			}
			if !isNoSpace(err) {
				t.Fatalf("Save error %v does not report the full disk", err)
			}
			// Nothing half-written may be left where the index would read it.
			if segs := segFilesIn(t, root); len(segs) != 1 {
				t.Fatalf("segment files after a failed flush: %v (want only the committed one)", segs)
			}
			// The engine still answers correctly from memory.
			assertSame(t, "engine after failed flush", ref, e)
			// The disk still holds exactly the previous state.
			checkRecovered(t, "disk after failed flush", root, base)

			df.SetBudget(-1)
			if err := e.Save(path); err != nil {
				t.Fatalf("retry after space returned: %v", err)
			}
			assertSame(t, "engine after retry", ref, e)
			assertSame(t, "reload after retry", ref, mustLoad(t, path))
		})
	}
}

func segFilesIn(t *testing.T, dir string) []string {
	t.Helper()
	m, _ := filepath.Glob(filepath.Join(dir, plDoc+".seg-*"))
	return m
}

// A merge that runs out of space leaves the live index untouched.
func TestDiskFull_CompactionFailsCleanly(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	df := fsx.NewDiskFull(-1)
	restore := fsx.SetHooks(df)
	defer restore()
	path := filepath.Join(root, plDoc)
	ref, e := diffEngine(), diffEngine()
	defer ref.Close()
	defer e.Close()
	for round := 0; round < 3; round++ {
		for i := 0; i < 30; i++ {
			id := fmt.Sprintf("r%d-%d", round, i)
			for _, x := range []*Engine{ref, e} {
				if err := x.Add(ctx, id, plText(i, round)); err != nil {
					t.Fatal(err)
				}
			}
		}
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
	}
	before := e.SegmentCount()
	df.SetBudget(2000)
	if err := e.Compact(); err == nil || !isNoSpace(err) {
		t.Fatalf("Compact error = %v, want a no-space error", err)
	}
	if e.SegmentCount() != before {
		t.Fatalf("segments %d -> %d after a failed compaction", before, e.SegmentCount())
	}
	assertSame(t, "engine after failed compaction", ref, e)
	assertSame(t, "disk after failed compaction", ref, mustLoad(t, path))
	df.SetBudget(-1)
	if err := e.Compact(); err != nil {
		t.Fatal(err)
	}
	assertSame(t, "after compaction with space", ref, e)
}
