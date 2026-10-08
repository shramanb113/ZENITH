package querycache

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

func TestBytesCache_SetThenGet(t *testing.T) {
	mr := miniredis.RunT(t)
	bc := NewBytesCache(mr.Addr(), time.Minute)
	defer bc.Close()

	ctx := context.Background()
	bc.Set(ctx, "a", []byte("hello"))
	v, ok := bc.Get(ctx, "a")
	if !ok || string(v) != "hello" {
		t.Fatalf("Get(a) = %q, %v; want \"hello\", true", v, ok)
	}
}

func TestBytesCache_MissOnUnknownKey(t *testing.T) {
	mr := miniredis.RunT(t)
	bc := NewBytesCache(mr.Addr(), time.Minute)
	defer bc.Close()

	if _, ok := bc.Get(context.Background(), "missing"); ok {
		t.Fatal("Get(missing) = _, true; want false")
	}
}

func TestBytesCache_TTLExpires(t *testing.T) {
	mr := miniredis.RunT(t)
	bc := NewBytesCache(mr.Addr(), time.Minute)
	defer bc.Close()

	ctx := context.Background()
	bc.Set(ctx, "a", []byte("hello"))
	mr.FastForward(2 * time.Minute)
	if _, ok := bc.Get(ctx, "a"); ok {
		t.Fatal("Get(a) = _, true after TTL expiry; want false")
	}
}

// Unreachable Redis must degrade to a miss, never panic or return an error
// to the caller — the whole point of the "degrade gracefully" contract
// (QUERYCACHE.md "Error handling"). An address nothing listens on stands in
// for a real outage.
func TestBytesCache_UnreachableRedisIsATreatedAsMiss(t *testing.T) {
	bc := NewBytesCache("127.0.0.1:1", time.Minute) // nothing listens on port 1
	defer bc.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, ok := bc.Get(ctx, "a"); ok {
		t.Fatal("Get against unreachable Redis returned a hit; want a miss")
	}
	// Set must not panic either.
	bc.Set(ctx, "a", []byte("hello"))
}
