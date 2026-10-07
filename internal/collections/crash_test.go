package collections

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// Hard-kill durability: a child process (this test binary re-executed)
// creates a collection and upserts documents one at a time, acknowledging
// each on stdout only after Upsert has returned nil. The parent hard-kills
// it mid-stream — no CloseAll, no graceful shutdown — then reopens the root
// and checks that every acknowledged document survived with its exact text.

const crashEnv = "ZENITH_COLLECTIONS_CRASH_CHILD"

func crashDocText(i int) string { return fmt.Sprintf("crash doc %d payload", i) }
func crashDocID(i int) string   { return fmt.Sprintf("d%d", i) }

// TestCollectionsCrashChild is the child half; it is skipped in normal runs.
func TestCollectionsCrashChild(t *testing.T) {
	root := os.Getenv(crashEnv)
	if root == "" {
		t.Skip("child process only")
	}
	m, err := New(Config{Root: root, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		fmt.Println("NEWFAIL", err)
		os.Exit(2)
	}
	if _, _, err := m.Create("k", CreateOptions{}); err != nil {
		if _, statErr := m.Stat("k"); statErr != nil {
			fmt.Println("CREATEFAIL", err)
			os.Exit(2)
		}
	}
	ctx := context.Background()
	for i := 0; ; i++ {
		id := crashDocID(i)
		if _, err := m.Upsert(ctx, "k", map[string]string{id: crashDocText(i)}, nil); err != nil {
			fmt.Println("UPSERTFAIL", err)
			os.Exit(2)
		}
		fmt.Println("ACK", id)
		os.Stdout.Sync()
	}
}

func TestCollections_SurvivesHardKill(t *testing.T) {
	if os.Getenv(crashEnv) != "" {
		t.Skip("child")
	}
	if testing.Short() {
		t.Skip("-short")
	}
	root := t.TempDir()

	cmd := exec.Command(os.Args[0], "-test.run=^TestCollectionsCrashChild$", "-test.v")
	cmd.Env = append(os.Environ(), crashEnv+"="+root)
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

	var acked []string
	for len(acked) < 30 {
		l, ok := <-lines
		if !ok {
			t.Fatalf("child exited early after %d acks", len(acked))
		}
		f := strings.Fields(l)
		if len(f) == 2 && f[0] == "ACK" {
			acked = append(acked, f[1])
		}
	}
	if err := cmd.Process.Kill(); err != nil {
		t.Fatalf("kill: %v", err)
	}
	_ = cmd.Wait()
	for range lines {
	}

	m, err := New(Config{Root: root, Embedder: embedding.NewDeterministicEmbedder(384)})
	if err != nil {
		t.Fatalf("New after kill: %v", err)
	}
	t.Cleanup(func() { _ = m.CloseAll() })

	ctx := context.Background()
	for i, id := range acked {
		got, err := m.GetDoc(ctx, "k", id)
		if err != nil {
			t.Fatalf("GetDoc(%s) after kill (acked #%d): %v", id, i, err)
		}
		want := crashDocText(i)
		if got != want {
			t.Fatalf("GetDoc(%s) = %q, want %q", id, got, want)
		}
	}
	info, err := m.Stat("k")
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.DocCount < len(acked) {
		t.Fatalf("DocCount = %d, want >= %d acked", info.DocCount, len(acked))
	}
}
