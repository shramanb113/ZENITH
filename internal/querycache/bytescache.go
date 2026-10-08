package querycache

import (
	"context"
	"log/slog"
	"time"

	"github.com/redis/go-redis/v9"
)

// BytesCache is the L2 tier: Redis fundamentally stores bytes, so this is
// the only tier that deals in []byte rather than a native Go value — Tiered
// supplies the encode/decode pair that lets Engine use it as []SearchResponse.
//
// Any Redis error (unreachable, timeout, ...) is logged at Warn and treated
// as a miss on Get, or silently dropped on Set — never surfaced to the
// caller as an error. A query-result cache is strictly an optimization; a
// Redis outage must degrade search to "slower", never "broken".
type BytesCache struct {
	client *redis.Client
	ttl    time.Duration
}

// Short, fixed Redis client timeouts: a query-result cache is strictly an
// optimization (see the type doc above), so a Redis outage or partition
// must turn into cache misses within tens of milliseconds, never into every
// search silently blocking for go-redis's multi-second defaults
// (DialTimeout 5s, Read/WriteTimeout 3s) while retrying. MaxRetries: 0 for
// the same reason — a retry only adds latency to an already-degraded path
// that is about to be treated as a miss anyway.
const (
	redisDialTimeout  = 100 * time.Millisecond
	redisReadTimeout  = 50 * time.Millisecond
	redisWriteTimeout = 50 * time.Millisecond
)

// NewBytesCache connects (lazily — go-redis dials on first command) to the
// Redis instance at addr. ttl is applied to every Set; 0 means "no expiry",
// which QUERYCACHE.md deliberately avoids as a default (callers should pass
// Config.QueryCacheTTL, whose zero value is translated to a 5-minute default
// before it ever reaches here — see internal/index's cache construction).
func NewBytesCache(addr string, ttl time.Duration) *BytesCache {
	return &BytesCache{
		client: redis.NewClient(&redis.Options{
			Addr:         addr,
			DialTimeout:  redisDialTimeout,
			ReadTimeout:  redisReadTimeout,
			WriteTimeout: redisWriteTimeout,
			MaxRetries:   0,
		}),
		ttl: ttl,
	}
}

func (b *BytesCache) Get(ctx context.Context, key string) ([]byte, bool) {
	v, err := b.client.Get(ctx, key).Bytes()
	if err != nil {
		if err != redis.Nil {
			slog.Warn("querycache: L2 get failed, treating as miss", "error", err)
		}
		return nil, false
	}
	return v, true
}

func (b *BytesCache) Set(ctx context.Context, key string, val []byte) {
	if err := b.client.Set(ctx, key, val, b.ttl).Err(); err != nil {
		slog.Warn("querycache: L2 set failed, entry not cached", "error", err)
	}
}

func (b *BytesCache) Close() error {
	return b.client.Close()
}
