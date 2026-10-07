package server

import (
	"context"
	"crypto/subtle"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// KeyMetadata is the gRPC metadata key carrying the shared secret, the
// metadata-call equivalent of the HTTP sidecar's X-Zenith-Key header.
const KeyMetadata = "x-zenith-key"

func checkKey(ctx context.Context, key string) error {
	if key == "" {
		return nil
	}
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return status.Error(codes.Unauthenticated, "unauthorized")
	}
	v := md.Get(KeyMetadata)
	if len(v) != 1 || subtle.ConstantTimeCompare([]byte(v[0]), []byte(key)) != 1 {
		return status.Error(codes.Unauthenticated, "unauthorized")
	}
	return nil
}

// KeyAuthUnary rejects unary calls missing a matching x-zenith-key metadata
// value. An empty key is a pass-through (auth disabled).
func KeyAuthUnary(key string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if err := checkKey(ctx, key); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// KeyAuthStream rejects streaming calls missing a matching x-zenith-key
// metadata value. An empty key is a pass-through (auth disabled).
func KeyAuthStream(key string) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := checkKey(ss.Context(), key); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}
