package index

import "testing"

func TestBucketKey_DifferentNamespacesProduceDifferentKeys(t *testing.T) {
	a := bucketKey("tenant-a", 1, nil, Weights{}, "epoch")
	b := bucketKey("tenant-b", 1, nil, Weights{}, "epoch")
	if a == b {
		t.Fatalf("bucketKey identical across namespaces: %q", a)
	}
}

func TestBucketKey_DifferentWriteGenProducesDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, nil, Weights{}, "epoch")
	b := bucketKey("ns", 2, nil, Weights{}, "epoch")
	if a == b {
		t.Fatalf("bucketKey identical across writeGen: %q", a)
	}
}

func TestBucketKey_DifferentSpecJSONProducesDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"a"}`), Weights{}, "epoch")
	b := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"b"}`), Weights{}, "epoch")
	if a == b {
		t.Fatalf("bucketKey identical across different filter specs: %q", a)
	}
}

func TestBucketKey_DifferentWeightsProduceDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, nil, Weights{Vector: 2.0}, "epoch")
	b := bucketKey("ns", 1, nil, Weights{Vector: 1.0}, "epoch")
	if a == b {
		t.Fatalf("bucketKey identical across different weights: %q", a)
	}
}

// Review Focus (code review, 2026-10-08): writeGen alone is only meaningful
// within one process's lifetime — without epoch in the key, a different
// process (after a restart, a different replica, or a recreated collection
// reusing the same id) at the same (namespace, writeGen) could read back
// another process's L2 entries for entirely different data.
func TestBucketKey_DifferentEpochProducesDifferentKey(t *testing.T) {
	a := bucketKey("ns", 1, nil, Weights{}, "epoch-a")
	b := bucketKey("ns", 1, nil, Weights{}, "epoch-b")
	if a == b {
		t.Fatalf("bucketKey identical across different process epochs: %q", a)
	}
}

func TestBucketKey_Deterministic(t *testing.T) {
	a := bucketKey("ns", 1, []byte(`{"op":"eq"}`), Weights{Vector: 1.5, Phonetic: 0.3, RRF: 20}, "epoch")
	b := bucketKey("ns", 1, []byte(`{"op":"eq"}`), Weights{Vector: 1.5, Phonetic: 0.3, RRF: 20}, "epoch")
	if a != b {
		t.Fatalf("bucketKey nondeterministic for identical inputs: %q vs %q", a, b)
	}
}

func TestFullCacheKey_DifferentQueriesProduceDifferentKeys(t *testing.T) {
	bucket := bucketKey("ns", 1, nil, Weights{}, "epoch")
	a := fullCacheKey(bucket, "weather in SF")
	b := fullCacheKey(bucket, "SF weather")
	if a == b {
		t.Fatalf("fullCacheKey identical across different query text: %q", a)
	}
}

func TestFullCacheKey_SameBucketAndQueryIsDeterministic(t *testing.T) {
	bucket := bucketKey("ns", 1, nil, Weights{}, "epoch")
	a := fullCacheKey(bucket, "same query")
	b := fullCacheKey(bucket, "same query")
	if a != b {
		t.Fatalf("fullCacheKey nondeterministic: %q vs %q", a, b)
	}
}

// Gap #5 from QUERYCACHE.md, directly: plain queries, WithWeights overrides,
// and WithFilter calls must never share a cache key with each other.
func TestFullCacheKey_WeightsAndFilterNeverCollideWithPlainQuery(t *testing.T) {
	plainBucket := bucketKey("ns", 1, nil, Weights{}, "epoch")
	weightedBucket := bucketKey("ns", 1, nil, Weights{Vector: 5}, "epoch")
	filteredBucket := bucketKey("ns", 1, []byte(`{"op":"eq","field":"x","value":"a"}`), Weights{}, "epoch")

	plain := fullCacheKey(plainBucket, "q")
	weighted := fullCacheKey(weightedBucket, "q")
	filtered := fullCacheKey(filteredBucket, "q")

	if plain == weighted || plain == filtered || weighted == filtered {
		t.Fatalf("cache keys collided: plain=%q weighted=%q filtered=%q", plain, weighted, filtered)
	}
}

func TestNewProcessEpoch_ProducesDifferentValuesEachCall(t *testing.T) {
	a := newProcessEpoch()
	b := newProcessEpoch()
	if a == b {
		t.Fatalf("newProcessEpoch returned identical values on two calls: %q", a)
	}
	if a == "" || b == "" {
		t.Fatal("newProcessEpoch returned an empty string")
	}
}
