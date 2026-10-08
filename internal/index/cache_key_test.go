package index

import "testing"

func TestBucketKey_DifferentNamespacesProduceDifferentKeys(t *testing.T) {
	a := bucketKey("tenant-a", 1, nil, Weights{})
	b := bucketKey("tenant-b", 1, nil, Weights{})
	if a == b {
		t.Fatalf("bucketKey identical across namespaces: %q", a)
	}
}

func TestBucketKey_DifferentWriteGenProducesDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, nil, Weights{})
	b := bucketKey("ns", 2, nil, Weights{})
	if a == b {
		t.Fatalf("bucketKey identical across writeGen: %q", a)
	}
}

func TestBucketKey_DifferentSpecJSONProducesDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"a"}`), Weights{})
	b := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"b"}`), Weights{})
	if a == b {
		t.Fatalf("bucketKey identical across different filter specs: %q", a)
	}
}

func TestBucketKey_DifferentWeightsProduceDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, nil, Weights{Vector: 2.0})
	b := bucketKey("ns", 1, nil, Weights{Vector: 1.0})
	if a == b {
		t.Fatalf("bucketKey identical across different weights: %q", a)
	}
}

func TestBucketKey_Deterministic(t *testing.T) {
	a := bucketKey("ns", 1, []byte(`{"op":"eq"}`), Weights{Vector: 1.5, Phonetic: 0.3, RRF: 20})
	b := bucketKey("ns", 1, []byte(`{"op":"eq"}`), Weights{Vector: 1.5, Phonetic: 0.3, RRF: 20})
	if a != b {
		t.Fatalf("bucketKey nondeterministic for identical inputs: %q vs %q", a, b)
	}
}

func TestFullCacheKey_DifferentQueriesProduceDifferentKeys(t *testing.T) {
	bucket := bucketKey("ns", 1, nil, Weights{})
	a := fullCacheKey(bucket, "weather in SF")
	b := fullCacheKey(bucket, "SF weather")
	if a == b {
		t.Fatalf("fullCacheKey identical across different query text: %q", a)
	}
}

func TestFullCacheKey_SameBucketAndQueryIsDeterministic(t *testing.T) {
	bucket := bucketKey("ns", 1, nil, Weights{})
	a := fullCacheKey(bucket, "same query")
	b := fullCacheKey(bucket, "same query")
	if a != b {
		t.Fatalf("fullCacheKey nondeterministic: %q vs %q", a, b)
	}
}

// Gap #5 from QUERYCACHE.md, directly: plain queries, WithWeights overrides,
// and WithFilter calls must never share a cache key with each other.
func TestFullCacheKey_WeightsAndFilterNeverCollideWithPlainQuery(t *testing.T) {
	plainBucket := bucketKey("ns", 1, nil, Weights{})
	weightedBucket := bucketKey("ns", 1, nil, Weights{Vector: 5})
	filteredBucket := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"a"}`), Weights{})

	plain := fullCacheKey(plainBucket, "q")
	weighted := fullCacheKey(weightedBucket, "q")
	filtered := fullCacheKey(filteredBucket, "q")

	if plain == weighted || plain == filtered || weighted == filtered {
		t.Fatalf("cache keys collided: plain=%q weighted=%q filtered=%q", plain, weighted, filtered)
	}
}
