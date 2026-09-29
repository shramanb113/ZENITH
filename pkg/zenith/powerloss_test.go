package zenith

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/shramanb113/ZENITH/internal/fsx"
)

// Power-loss and disk-full tests at the DB level: the WAL, its rotation at a
// checkpoint, the engine's flush and the archive cleanup together. See
// internal/index/powerloss_test.go for the model; here the acknowledgement point
// is a returned Add/Delete (the WAL fsync), and a checkpoint is interleaved
// with writes exactly as the background checkpoint interleaves them.

const plName = "pl.db"

type dbState map[string]string

func (s dbState) clone() dbState {
	c := make(dbState, len(s))
	for k, v := range s {
		c[k] = v
	}
	return c
}

func plDocText(i, ver int) string {
	return fmt.Sprintf("powerloss doc%d version%d payload shared token%d", i, ver, i%4)
}

// recoverAndCompare opens the crash image and requires its contents to equal one
// of the allowed states.
func recoverAndCompare(t *testing.T, what, dir string, universe []string, allowed ...dbState) {
	t.Helper()
	db, err := Open(filepath.Join(dir, plName), WithBM25Only())
	if err != nil {
		t.Fatalf("%s: Open failed: %v", what, err)
	}
	defer db.Close()
	got := dbState{}
	for _, id := range universe {
		if txt, ok, _ := db.Get(id); ok {
			got[id] = txt
		}
	}
	if n := db.engine.Count(); n != len(got) {
		t.Fatalf("%s: index holds %d documents, %d of them known to the model", what, n, len(got))
	}
	for _, s := range allowed {
		if len(s) != len(got) {
			continue
		}
		same := true
		for k, v := range s {
			if got[k] != v {
				same = false
				break
			}
		}
		if same {
			return
		}
	}
	sizes := make([]int, len(allowed))
	for i, s := range allowed {
		sizes[i] = len(s)
	}
	t.Fatalf("%s: recovered %d documents matching none of the %d allowed states (sizes %v)", what, len(got), len(allowed), sizes)
}

func TestPowerLoss_DBWithWALAndCheckpoints(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	rec := fsx.NewRecorder(root)
	restore := fsx.SetHooks(rec)
	defer restore()

	db, err := Open(filepath.Join(root, plName), WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}

	state := dbState{}
	states := []dbState{{}} // states[n] = contents after n acknowledged mutations
	var universe []string
	seen := map[string]bool{}
	ack := func() {
		states = append(states, state.clone())
		rec.Mark("ack")
	}
	put := func(i, ver int) {
		t.Helper()
		id := fmt.Sprintf("d%d", i)
		if err := db.Add(ctx, id, plDocText(i, ver)); err != nil {
			t.Fatal(err)
		}
		state[id] = plDocText(i, ver)
		if !seen[id] {
			seen[id] = true
			universe = append(universe, id)
		}
		ack()
	}
	del := func(i int) {
		t.Helper()
		id := fmt.Sprintf("d%d", i)
		if err := db.Delete(ctx, id); err != nil {
			t.Fatal(err)
		}
		delete(state, id)
		ack()
	}
	begin := func() *checkpointJob {
		t.Helper()
		db.mu.Lock()
		job, err := db.beginCheckpointLocked()
		db.mu.Unlock()
		if err != nil {
			t.Fatal(err)
		}
		return job
	}
	finish := func(job *checkpointJob) {
		t.Helper()
		if job == nil {
			return
		}
		if err := db.finishCheckpoint(job); err != nil {
			t.Fatal(err)
		}
	}

	for i := 0; i < 25; i++ {
		put(i, 1)
	}
	job := begin() // freeze + cut the WAL
	for i := 25; i < 35; i++ {
		put(i, 1) // written while the checkpoint is in flight
	}
	del(3)    // a frozen document
	put(4, 2) // replace a frozen document
	finish(job)
	for i := 35; i < 45; i++ {
		put(i, 1)
	}
	del(30) // a document flushed by the first checkpoint
	job = begin()
	put(45, 1)
	finish(job)
	put(46, 1)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	restore()

	ops := rec.Ops()
	t.Logf("scenario recorded %d operations, %d acknowledged mutations", len(ops), len(states)-1)
	if len(ops) < 50 {
		t.Fatalf("only %d operations recorded: fsx hooks are not seeing the WAL/segment writes", len(ops))
	}

	checked := 0
	for k := 0; k <= len(ops); k++ {
		n := 0
		for _, m := range rec.Marks(k) {
			if m == "ack" {
				n++
			}
		}
		allowed := []dbState{states[n]}
		if n+1 < len(states) {
			allowed = append(allowed, states[n+1]) // the mutation in flight may already be durable
		}
		for _, mode := range []struct {
			name string
			p    fsx.Persistence
			seed int64
		}{
			{"nothing-unsynced-survives", fsx.Nothing, 0},
			{"everything-survives", fsx.Everything, 0},
			{"torn", fsx.Torn, int64(k)*13 + 5},
		} {
			dst := filepath.Join(t.TempDir(), "crash")
			if err := rec.CrashState(k, mode.p, mode.seed, dst); err != nil {
				t.Fatal(err)
			}
			recoverAndCompare(t, fmt.Sprintf("power cut after op %d/%d (%s), %d acked", k, len(ops), mode.name, n), dst, universe, allowed...)
			checked++
		}
	}
	t.Logf("checked %d simulated power cuts", checked)
}

// ---- disk full ----

// A write-ahead-log append that hits a full disk must fail without
// acknowledging, must not leave a torn record that swallows later ones, and the
// DB must keep working once space returns.
func TestDiskFull_WALAppendFailsCleanlyAndRecovers(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	df := fsx.NewDiskFull(-1)
	restore := fsx.SetHooks(df)
	defer restore()

	path := filepath.Join(root, plName)
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	want := dbState{}
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("d%d", i)
		if err := db.Add(ctx, id, plDocText(i, 1)); err != nil {
			t.Fatal(err)
		}
		want[id] = plDocText(i, 1)
	}

	// Budgets small enough to cut the next record at different offsets.
	for _, budget := range []int64{0, 3, 40} {
		df.SetBudget(budget)
		err := db.Add(ctx, "refused", plDocText(99, 1))
		if err == nil {
			t.Fatalf("budget %d: Add succeeded on a full disk", budget)
		}
		if !errors.Is(err, syscall.ENOSPC) {
			t.Fatalf("budget %d: Add error %v does not report the full disk", budget, err)
		}
		if _, ok, _ := db.Get("refused"); ok {
			t.Fatalf("budget %d: a refused Add is visible in the index", budget)
		}
		if err := db.Delete(ctx, "d1"); err == nil {
			t.Fatalf("budget %d: Delete succeeded on a full disk", budget)
		}
		if _, ok, _ := db.Get("d1"); !ok {
			t.Fatalf("budget %d: a refused Delete removed the document", budget)
		}
	}

	df.SetBudget(-1)
	for i := 10; i < 15; i++ {
		id := fmt.Sprintf("d%d", i)
		if err := db.Add(ctx, id, plDocText(i, 1)); err != nil {
			t.Fatalf("Add after space returned: %v", err)
		}
		want[id] = plDocText(i, 1)
	}
	if err := db.Delete(ctx, "d2"); err != nil {
		t.Fatal(err)
	}
	delete(want, "d2")

	// Crash right now (copy the directory: the WAL is fsynced) and recover.
	image := copyDir(t, root)
	universe := []string{"refused"}
	for i := 0; i < 15; i++ {
		universe = append(universe, fmt.Sprintf("d%d", i))
	}
	recoverAndCompare(t, "crash image after the disk-full episode", image, universe, want)
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	recoverAndCompare(t, "clean reopen after the disk-full episode", root, universe, want)
	if b, _ := os.ReadFile(path + ".wal"); strings.Contains(string(b), "refused") {
		t.Fatal("a refused record is present in the WAL")
	}
}

// A checkpoint that cannot write its segment fails without losing anything: the
// frozen documents stay searchable, the WAL archive stays until a checkpoint
// succeeds, and a crash at that point recovers everything.
func TestDiskFull_CheckpointFailureKeepsWALArchive(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	df := fsx.NewDiskFull(-1)
	restore := fsx.SetHooks(df)
	defer restore()

	path := filepath.Join(root, plName)
	db, err := Open(path, WithBM25Only())
	if err != nil {
		t.Fatal(err)
	}
	want := dbState{}
	universe := []string{}
	for i := 0; i < 40; i++ {
		id := fmt.Sprintf("d%d", i)
		universe = append(universe, id)
		if err := db.Add(ctx, id, plDocText(i, 1)); err != nil {
			t.Fatal(err)
		}
		want[id] = plDocText(i, 1)
	}

	db.mu.Lock()
	job, err := db.beginCheckpointLocked()
	db.mu.Unlock()
	if err != nil || job == nil {
		t.Fatalf("begin: %v %v", job, err)
	}
	df.SetBudget(50)
	if err := db.finishCheckpoint(job); err == nil || !errors.Is(err, syscall.ENOSPC) {
		t.Fatalf("finishCheckpoint error = %v, want a no-space error", err)
	}
	if archives, _ := walArchives(path + ".wal"); len(archives) != 1 {
		t.Fatalf("WAL archive must survive a failed checkpoint, have %v", archives)
	}
	for i := 0; i < 40; i++ { // still served from the frozen layer
		if _, ok, _ := db.Get(fmt.Sprintf("d%d", i)); !ok {
			t.Fatalf("document d%d vanished after a failed checkpoint", i)
		}
	}
	recoverAndCompare(t, "crash after a failed checkpoint", copyDir(t, root), universe, want)

	df.SetBudget(-1)
	db.mu.Lock()
	job2, err := db.beginCheckpointLocked()
	db.mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if job2 == nil {
		t.Fatal("the frozen layer left by the failed checkpoint is not pending")
	}
	if err := db.finishCheckpoint(job2); err != nil {
		t.Fatalf("retry after space returned: %v", err)
	}
	if archives, _ := walArchives(path + ".wal"); len(archives) != 0 {
		t.Fatalf("archives left after a successful checkpoint: %v", archives)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	recoverAndCompare(t, "reopen after the retry", root, universe, want)
}
