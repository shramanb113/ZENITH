package server

import (
	"context"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

func ctxWithKey(values ...string) context.Context {
	if values == nil {
		return context.Background()
	}
	md := metadata.MD{}
	for _, v := range values {
		md.Append(KeyMetadata, v)
	}
	return metadata.NewIncomingContext(context.Background(), md)
}

func TestKeyAuthUnary(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		ctx     context.Context
		wantErr bool
	}{
		{"no server key, no metadata", "", ctxWithKey(), false},
		{"no server key, with metadata", "", ctxWithKey("anything"), false},
		{"correct key", "k", ctxWithKey("k"), false},
		{"wrong key", "k", ctxWithKey("nope"), true},
		{"missing metadata entirely", "k", context.Background(), true},
		{"missing value", "k", ctxWithKey(), true},
		{"two values", "k", ctxWithKey("k", "k"), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			handler := func(ctx context.Context, req any) (any, error) {
				called = true
				return "ok", nil
			}
			interceptor := KeyAuthUnary(tt.key)
			resp, err := interceptor(tt.ctx, nil, &grpc.UnaryServerInfo{}, handler)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected an error, got nil")
				}
				if status.Code(err) != codes.Unauthenticated {
					t.Fatalf("code = %v, want Unauthenticated", status.Code(err))
				}
				if called {
					t.Fatal("handler must not run when auth fails")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !called || resp != "ok" {
				t.Fatal("handler should have run and returned its response")
			}
		})
	}
}

// fakeServerStream is the minimum grpc.ServerStream needed to exercise
// KeyAuthStream: only Context() is read by the interceptor.
type fakeServerStream struct {
	grpc.ServerStream
	ctx context.Context
}

func (f *fakeServerStream) Context() context.Context { return f.ctx }

func TestKeyAuthStream(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		ctx     context.Context
		wantErr bool
	}{
		{"no server key", "", ctxWithKey(), false},
		{"correct key", "k", ctxWithKey("k"), false},
		{"wrong key", "k", ctxWithKey("nope"), true},
		{"missing", "k", context.Background(), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			called := false
			handler := func(srv any, stream grpc.ServerStream) error {
				called = true
				return nil
			}
			interceptor := KeyAuthStream(tt.key)
			err := interceptor(nil, &fakeServerStream{ctx: tt.ctx}, &grpc.StreamServerInfo{}, handler)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected an error, got nil")
				}
				if status.Code(err) != codes.Unauthenticated {
					t.Fatalf("code = %v, want Unauthenticated", status.Code(err))
				}
				if called {
					t.Fatal("handler must not run when auth fails")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !called {
				t.Fatal("handler should have run")
			}
		})
	}
}
