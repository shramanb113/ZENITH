package index

import (
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"testing"
)

// TestMemoryPerDoc reports retained heap per document on the synthetic Zipf
// corpus, and (with ZENITH_MEMPROFILE=path) writes an in-use heap profile so
// the biggest structures can be ranked with `go tool pprof -sample_index=inuse_space`.
// It is a measurement, not an assertion, and is skipped unless ZENITH_MEMDOCS is set.
func TestMemoryPerDoc(t *testing.T) {
	v := os.Getenv("ZENITH_MEMDOCS")
	if v == "" {
		t.Skip("set ZENITH_MEMDOCS=<n> to run")
	}
	n, _ := strconv.Atoi(v)

	runtime.GC()
	var before runtime.MemStats
	runtime.ReadMemStats(&before)

	eng, _ := buildLexEngine(t, n, 30000)

	runtime.GC()
	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	t.Logf("docs=%d retained heap=%.1f MB  (%.2f KB/doc)", n,
		float64(after.HeapAlloc-before.HeapAlloc)/1e6,
		float64(after.HeapAlloc-before.HeapAlloc)/1024/float64(n))

	if p := os.Getenv("ZENITH_MEMPROFILE"); p != "" {
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := pprof.Lookup("heap").WriteTo(f, 0); err != nil {
			t.Fatal(err)
		}
	}
	runtime.KeepAlive(eng)
}
