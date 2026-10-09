//go:build cgo

package localembedder

import (
	"testing"
	"time"
)

// These tests need the real onnxruntime shared library extracted (via
// extractToTemp(ortLibBytes, ortLibFilename), the same helper newEmbedder
// already uses) plus a real small ONNX model — which this task does not
// have without network access to the real clip-vit-base-patch32 files. They
// are therefore structural/safety tests only (pool sizing, plus a
// no-real-session safety check that close() doesn't deadlock on a pool with
// no started workers — see TestVisionPool_CloseOnEmptyPoolDoesNotDeadlock
// below), not real-inference tests — Task 8's opt-in smoke test (gated on
// the real model actually being pulled) covers real inference.

func TestNewVisionPool_PoolSizeAtLeastOne(t *testing.T) {
	// A pool constructed with poolSize 0 or negative must not panic from
	// SetIntraOpNumThreads(0) or divide-by-zero — it must clamp to 1.
	// This only exercises the sizing math (via the exported threadsPerSession
	// helper), not a real session, since no real model bytes are available
	// in this unit test.
	for _, poolSize := range []int{-1, 0, 1, 2, 999} {
		got := clipThreadsPerSession(poolSize, 4) // pretend 4 CPUs
		if got < 1 {
			t.Errorf("clipThreadsPerSession(%d, 4) = %d, want >= 1", poolSize, got)
		}
	}
}

func TestClipPoolSize_AtLeastOne(t *testing.T) {
	for _, numCPU := range []int{1, 2, 4, 16} {
		got := clipPoolSize(numCPU)
		if got < 1 {
			t.Errorf("clipPoolSize(%d) = %d, want >= 1", numCPU, got)
		}
	}
}

// TestVisionPool_CloseOnEmptyPoolDoesNotDeadlock exercises close()'s own
// documented claim (see visionPool.close()'s doc comment) that closing a
// pool with no started worker goroutines is safe and returns immediately:
// closing an unclosed channel with no readers is a no-op wait, and a
// zero-value (or partially incremented) sync.WaitGroup with no pending Add
// calls outstanding returns immediately from Wait. This needs no real ONNX
// session — p.workers is left empty, so close()'s session-destroy loop never
// runs.
func TestVisionPool_CloseOnEmptyPoolDoesNotDeadlock(t *testing.T) {
	p := &visionPool{jobs: make(chan visionJob)}

	done := make(chan struct{})
	go func() {
		p.close()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("visionPool.close() deadlocked on a pool with no started workers")
	}
}
