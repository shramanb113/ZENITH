package index

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// newProcessEpoch returns a fresh random identifier for one Engine
// instance's lifetime, used by bucketKey to keep the optional L2 cache
// correct across restarts and replicas (see bucketKey's doc comment). Not a
// security token — it never leaves the process except baked into an opaque
// cache-key hash — so crypto/rand is used only for good uniqueness, not for
// any confidentiality property.
func newProcessEpoch() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// crypto/rand failing is effectively unreachable on every supported
		// platform; falling back to all-zeros would still be correct within
		// a single process (every bucketKey call in this process gets the
		// same epoch either way) and only loses the cross-process isolation
		// this field exists for, in an already-exceptional situation.
		return "00000000000000000000000000000000"
	}
	return hex.EncodeToString(b[:])
}

// bucketKey hashes everything that determines a query-result cache entry's
// correctness *except the query text itself*: namespace, writeGen, the
// filter's structured spec (as JSON), the per-call weight overrides, and the
// engine's process epoch (see Engine.processEpoch). Two calls differing only
// in query text share a bucket — exactly the grouping the semantic/near-
// duplicate scan needs (it must never compare vectors across different
// filters or weights; see semantic_cache.go).
//
// epoch exists because writeGen is only meaningful within one process's
// lifetime: it starts at 0 on every restart and is unrelated to which actual
// documents are indexed. Without epoch in the key, an L2 (Redis) entry
// written by a process at generation N could be read back by a different
// process — after a restart, or a different replica, or a recreated
// collection reusing the same id — that is also at generation N but holds
// entirely different data. Mixing a random-per-process epoch into the key
// means a restart (or any other new Engine instance) can never read an
// older process's L2 entries, trading away intentional cross-replica cache
// sharing for correctness; sharing a warm cache across replicas serving the
// exact same immutable snapshot is a deliberately unsupported use case (see
// DECISIONS.md).
//
// Each variable-length field is length-prefixed before hashing so, e.g.,
// specJSON=`"ab"` + query="c" can never hash identically to specJSON=`"a"`
// + query="bc" — see fullCacheKey, which appends the query the same way.
func bucketKey(namespace string, writeGen uint64, specJSON []byte, w Weights, epoch string) string {
	h := sha256.New()
	writeField(h, []byte(namespace))
	var genBuf [8]byte
	binary.BigEndian.PutUint64(genBuf[:], writeGen)
	h.Write(genBuf[:])
	writeField(h, specJSON)
	var wBuf [24]byte
	binary.BigEndian.PutUint64(wBuf[0:8], math.Float64bits(w.Vector))
	binary.BigEndian.PutUint64(wBuf[8:16], math.Float64bits(w.Phonetic))
	binary.BigEndian.PutUint64(wBuf[16:24], math.Float64bits(w.RRF))
	h.Write(wBuf[:])
	writeField(h, []byte(epoch))
	return hex.EncodeToString(h.Sum(nil))
}

// fullCacheKey extends a bucket with the query text, producing the actual
// cache/singleflight key used for the exact-match path.
func fullCacheKey(bucket, query string) string {
	h := sha256.New()
	writeField(h, []byte(bucket))
	writeField(h, []byte(query))
	return hex.EncodeToString(h.Sum(nil))
}

type hashWriter interface {
	Write(p []byte) (int, error)
}

func writeField(h hashWriter, b []byte) {
	var lenBuf [8]byte
	binary.BigEndian.PutUint64(lenBuf[:], uint64(len(b)))
	h.Write(lenBuf[:])
	h.Write(b)
}
