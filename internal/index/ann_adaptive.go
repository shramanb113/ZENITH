package index

import "sync"

// ewma is a small exponentially-weighted rolling average, used to track
// recent per-path search latency (exact scan vs. ANN graph) without storing
// a history. alpha weights the newest observation; 1-alpha decays the rest.
// Zero value is "no observations yet" (value() returns 0 until the first
// observe call, which seeds it directly rather than blending against 0).
// Guarded by mu: concurrent searches call observe() on the same Engine's
// annLatency/exactLatency fields, and shouldUseANN reads them concurrently
// with that — both the write and the read must be synchronized.
type ewma struct {
	mu  sync.Mutex
	v   float64
	set bool
}

// alpha close to 0.3 reacts within a handful of observations without being
// so twitchy that one slow outlier flips the adaptive decision immediately.
const ewmaAlpha = 0.3

func (e *ewma) observe(v float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.set {
		e.v = v
		e.set = true
		return
	}
	e.v = ewmaAlpha*v + (1-ewmaAlpha)*e.v
}

func (e *ewma) value() float64 {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.v
}

// isSet reports whether observe has been called at least once. Synchronized
// the same way value() is, since shouldUseANN reads it concurrently with
// observe() writing it.
func (e *ewma) isSet() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.set
}

// shouldUseANN decides whether vectorPass should use the ANN graph for a
// corpus of this size. Outside Config.ANNThresholdBandPct's band around
// e.annMinDocs (or when the band is disabled, ANNThresholdBandPct <= 0), the
// static threshold alone decides — exactly as before this feature existed.
// Inside the band, whichever path's rolling-average latency is currently
// lower wins; with no observations yet for one or both paths, the static
// threshold still decides (no measurement to trust yet).
func (e *Engine) shouldUseANN(corpusSize int) bool {
	static := corpusSize >= e.annMinDocs
	band := e.config.ANNThresholdBandPct
	if band <= 0 || e.annMinDocs <= 0 {
		return static
	}
	lo := float64(e.annMinDocs) * (1 - band)
	hi := float64(e.annMinDocs) * (1 + band)
	if float64(corpusSize) < lo || float64(corpusSize) > hi {
		return static
	}
	if !e.annLatency.isSet() || !e.exactLatency.isSet() {
		return static
	}
	return e.annLatency.value() < e.exactLatency.value()
}
