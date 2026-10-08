package index

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
)

// bucketKey hashes everything that determines a query-result cache entry's
// correctness *except the query text itself*: namespace, writeGen, the
// filter's structured spec (as JSON), and the per-call weight overrides.
// Two calls differing only in query text share a bucket — exactly the
// grouping the semantic/near-duplicate scan needs (it must never compare
// vectors across different filters or weights; see semantic_cache.go).
//
// Each variable-length field is length-prefixed before hashing so, e.g.,
// specJSON=`"ab"` + query="c" can never hash identically to specJSON=`"a"`
// + query="bc" — see fullCacheKey, which appends the query the same way.
func bucketKey(namespace string, writeGen uint64, specJSON []byte, w Weights) string {
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
