package index

import (
	"bufio"
	"context"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Crash-recovery tests. A child process (this test binary re-executed) builds
// an index and dies — either at a named failpoint inside flush/compaction
// (ZENITH_FAILPOINT, deterministic) or by being hard-killed by the parent at a
// random moment (TerminateProcess / SIGKILL, the `kill -9` case). The parent
// then opens the directory and checks the durability contract:
//
//   - the index always opens (no torn manifest, no dangling segment reference);
//   - every document whose Save had returned is present with its exact text;
//   - no document that was removed before a returned Save reappears;
//   - orphaned segment files from the interrupted commit are cleaned up.

const crashEnv = "ZENITH_CRASH_CHILD"

func crashText(i int) string { return fmt.Sprintf("crashdoc%d payload token%d shared", i, i%7) }
func crashID(i int) string   { return fmt.Sprintf("c%d", i) }

// TestCrashChild is the child half; it is skipped in normal runs.
func TestCrashChild(t *testing.T) {
	dir := os.Getenv(crashEnv)
	if dir == "" {
		t.Skip("child process only")
	}
	path := filepath.Join(dir, "crash.db")
	ctx := context.Background()
	e := diffEngine()
	if _, err := os.Stat(path); err == nil {
		if err := e.Load(path); err != nil {
			fmt.Println("LOADFAIL", err)
			os.Exit(2)
		}
	}
	fp := os.Getenv("ZENITH_CRASH_FAILPOINT") // armed only after the baseline is durable
	ack := func(format string, a ...any) { fmt.Printf(format+"\n", a...); os.Stdout.Sync() }

	switch os.Getenv("ZENITH_CRASH_MODE") {
	case "flush", "compact":
		add := func(lo, hi int) {
			for i := lo; i < hi; i++ {
				if err := e.Add(ctx, crashID(i), crashText(i)); err != nil {
					t.Fatal(err)
				}
			}
		}
		add(0, 40)
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
		ack("SAVED 40")
		add(40, 80)
		for i := 0; i < 10; i++ { // deletes of segment docs
			if err := e.Remove(ctx, crashID(i)); err != nil {
				t.Fatal(err)
			}
		}
		if os.Getenv("ZENITH_CRASH_MODE") == "flush" {
			os.Setenv("ZENITH_FAILPOINT", fp)
			if err := e.Save(path); err != nil {
				t.Fatal(err)
			}
			ack("SAVED 80")
			return
		}
		if err := e.Save(path); err != nil {
			t.Fatal(err)
		}
		ack("SAVED 80")
		os.Setenv("ZENITH_FAILPOINT", fp)
		if err := e.Compact(); err != nil {
			t.Fatal(err)
		}
		ack("COMPACTED")

	case "loop":
		// Runs until killed. After every Save the acknowledged high-water mark
		// and the removed set are printed; the parent trusts only those lines.
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		next := 0
		removed := map[int]bool{}
		for {
			n := 5 + r.Intn(20)
			for i := 0; i < n; i++ {
				if err := e.Add(ctx, crashID(next), crashText(next)); err != nil {
					t.Fatal(err)
				}
				next++
			}
			for i := 0; i < 3 && next > 10; i++ {
				v := r.Intn(next)
				if !removed[v] {
					ack("RM %d", v) // announced first: it may become durable before the ack below
					if err := e.Remove(ctx, crashID(v)); err != nil {
						t.Fatal(err)
					}
					removed[v] = true
				}
			}
			if err := e.Save(path); err != nil {
				t.Fatal(err)
			}
			// Everything up to `next`, and every removal announced so far, was
			// durable when this line prints.
			ack("SAVED %d", next)
			if r.Intn(6) == 0 {
				if err := e.Compact(); err != nil {
					t.Fatal(err)
				}
				ack("COMPACTED")
			}
		}
	}
}

func runChild(t *testing.T, dir, mode, failpoint string) []string {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestCrashChild$", "-test.v")
	cmd.Env = append(os.Environ(), crashEnv+"="+dir, "ZENITH_CRASH_MODE="+mode, "ZENITH_CRASH_FAILPOINT="+failpoint, "ZENITH_FAILPOINT=")
	out, err := cmd.CombinedOutput()
	if failpoint != "" {
		ee, ok := err.(*exec.ExitError)
		if !ok || ee.ExitCode() != 137 {
			t.Fatalf("child was expected to die at failpoint %q with 137, got err=%v\n%s", failpoint, err, out)
		}
	} else if err != nil {
		t.Fatalf("child failed: %v\n%s", err, out)
	}
	var acks []string
	for _, l := range strings.Split(string(out), "\n") {
		if l = strings.TrimSpace(l); strings.HasPrefix(l, "SAVED") || strings.HasPrefix(l, "COMPACTED") {
			acks = append(acks, l)
		}
	}
	return acks
}

// expectDocs asserts the loaded index contains exactly the wanted document set.
func expectDocs(t *testing.T, e *Engine, present func(i int) bool, upto int) {
	t.Helper()
	want := 0
	for i := 0; i < upto; i++ {
		text, ok := e.GetText(crashID(i))
		if present(i) {
			want++
			if !ok || text != crashText(i) {
				t.Fatalf("document %d: present=%v text=%q, want %q", i, ok, text, crashText(i))
			}
		} else if ok {
			t.Fatalf("document %d should not exist but does", i)
		}
	}
	if e.Count() != want {
		t.Fatalf("Count = %d, want %d", e.Count(), want)
	}
	res, err := e.Search(context.Background(), "crashdoc45 payload")
	if err != nil {
		t.Fatal(err)
	}
	if want > 45 && len(res) == 0 {
		t.Fatal("search returns nothing on a non-empty recovered index")
	}
}

func segFiles(t *testing.T, dir string) []string {
	t.Helper()
	m, _ := filepath.Glob(filepath.Join(dir, "crash.db.seg-*"))
	return m
}

func TestCrash_AtFailpoints(t *testing.T) {
	if os.Getenv(crashEnv) != "" {
		t.Skip("child")
	}
	// present(i) is the visible state before the interrupted step (`before`) or after it.
	base := func(i int) bool { return i < 40 }
	full := func(i int) bool { return i >= 10 && i < 80 }

	cases := []struct {
		name, mode, fp string
		want           func(i int) bool // state a correct recovery must show
		segsAfterOpen  int
	}{
		// Segment fully written, manifest not yet: the flush never happened.
		{"flush dies before manifest commit", "flush", "flush-after-segment", base, 1},
		// Manifest renamed, process dies before cleanup/in-memory reset: flush is durable.
		{"flush dies after manifest commit", "flush", "flush-after-manifest", full, 2},
		// Merge written but not committed: old segments still authoritative.
		{"compaction dies before manifest commit", "compact", "compact-after-segment", full, 2},
		// Merge committed; old segment files not yet deleted: must be GC'd on open.
		{"compaction dies after manifest commit", "compact", "compact-after-manifest", full, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			runChild(t, dir, tc.mode, tc.fp)

			e := diffEngine()
			t.Cleanup(func() { e.Close() })
			if err := e.Load(filepath.Join(dir, "crash.db")); err != nil {
				t.Fatalf("recovery failed to open the index: %v", err)
			}
			expectDocs(t, e, tc.want, 80)
			if n := len(segFiles(t, dir)); n != tc.segsAfterOpen {
				t.Fatalf("%d segment files on disk after recovery, want %d (orphans not collected?): %v", n, tc.segsAfterOpen, segFiles(t, dir))
			}

			// The recovered index must accept writes and flush again.
			if err := e.Add(context.Background(), "post", "post recovery document"); err != nil {
				t.Fatal(err)
			}
			if err := e.Save(filepath.Join(dir, "crash.db")); err != nil {
				t.Fatalf("Save after recovery: %v", err)
			}
		})
	}
}

// TestCrash_HardKill kills a busy writer at random moments (no failpoint: the
// process is terminated wherever it happens to be — mid-write, mid-fsync,
// mid-rename, mid-merge) and checks the durability contract each time.
func TestCrash_HardKill(t *testing.T) {
	if os.Getenv(crashEnv) != "" {
		t.Skip("child")
	}
	if testing.Short() {
		t.Skip("-short")
	}
	rounds := 8
	if v, err := strconv.Atoi(os.Getenv("ZENITH_CRASH_ROUNDS")); err == nil {
		rounds = v
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "crash.db")
	r := rand.New(rand.NewSource(7))

	for round := 0; round < rounds; round++ {
		cmd := exec.Command(os.Args[0], "-test.run=^TestCrashChild$", "-test.v")
		cmd.Env = append(os.Environ(), crashEnv+"="+dir, "ZENITH_CRASH_MODE=loop", "ZENITH_FAILPOINT=")
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		lines := make(chan string, 4096)
		go func() {
			sc := bufio.NewScanner(stdout)
			for sc.Scan() {
				lines <- sc.Text()
			}
			close(lines)
		}()

		// acked: durable when the child printed SAVED. maybe: announced removals
		// not yet covered by a SAVED — they may or may not have become durable.
		kill := time.After(time.Duration(300+r.Intn(1200)) * time.Millisecond)
		ackedNext := 0
		ackedRemoved := map[int]bool{}
		var pendingRM []int
	loop:
		for {
			select {
			case l, ok := <-lines:
				if !ok {
					break loop
				}
				f := strings.Fields(l)
				switch {
				case len(f) == 2 && f[0] == "RM":
					v, _ := strconv.Atoi(f[1])
					pendingRM = append(pendingRM, v)
				case len(f) == 2 && f[0] == "SAVED":
					ackedNext, _ = strconv.Atoi(f[1])
					for _, v := range pendingRM {
						ackedRemoved[v] = true
					}
					pendingRM = nil
				}
			case <-kill:
				cmd.Process.Kill() // TerminateProcess / SIGKILL: no cleanup runs
				break loop
			}
		}
		cmd.Wait()
		for range lines { // drain
		}
		maybe := map[int]bool{}
		for _, v := range pendingRM {
			maybe[v] = true
		}

		e := diffEngine()
		if err := e.Load(path); err != nil {
			e.Close()
			// Killed before the very first Save committed: there is legitimately
			// no index yet. That is only acceptable if nothing was acknowledged.
			if os.IsNotExist(err) && ackedNext == 0 {
				t.Logf("round %d: killed before the first commit, nothing acknowledged — no index expected", round)
				continue
			}
			t.Fatalf("round %d: index does not open after kill (acked through %d): %v", round, ackedNext, err)
		}
		for i := 0; i < ackedNext; i++ {
			if ackedRemoved[i] || maybe[i] {
				continue
			}
			text, ok := e.GetText(crashID(i))
			if !ok || text != crashText(i) {
				e.Close()
				t.Fatalf("round %d: acknowledged doc %d lost or corrupted (present=%v %q); acked through %d", round, i, ok, text, ackedNext)
			}
		}
		for v := range ackedRemoved {
			if _, ok := e.GetText(crashID(v)); ok {
				e.Close()
				t.Fatalf("round %d: acknowledged deletion of doc %d was lost", round, v)
			}
		}
		if _, err := e.Search(context.Background(), "crashdoc1 payload"); err != nil {
			e.Close()
			t.Fatalf("round %d: search after recovery: %v", round, err)
		}
		// The recovered index must keep working: write and flush again.
		if err := e.Add(context.Background(), "after-crash", "written after recovery"); err != nil {
			e.Close()
			t.Fatal(err)
		}
		if err := e.Save(path); err != nil {
			e.Close()
			t.Fatalf("round %d: Save after recovery: %v", round, err)
		}
		e.Close()
		t.Logf("round %d: killed with %d docs acked, %d removals acked, %d in flight — recovered", round, ackedNext, len(ackedRemoved), len(pendingRM))

		// Start the next round from an empty directory so its acks are self-contained.
		files, _ := filepath.Glob(path + "*")
		for _, f := range files {
			os.Remove(f)
		}
	}
}
