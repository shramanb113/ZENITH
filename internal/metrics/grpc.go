package metrics

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// UnaryServerInterceptor counts and times every unary RPC under its full
// method name and the gRPC status code name returned (an error without a
// *status.Status, e.g. a plain Go error, counts as "Unknown" — status.Code
// already does that). Non-OK codes are also counted in errors_total.
// Chain it before auth (grpc.ChainUnaryInterceptor(metrics..., auth...)) so
// a rejected call is still counted.
func UnaryServerInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		var resp any
		var err error
		observeDuration(GRPCRequestDuration.WithLabelValues(info.FullMethod), func() {
			resp, err = handler(ctx, req)
		})
		code := status.Code(err)
		GRPCRequestsTotal.WithLabelValues(info.FullMethod, code.String()).Inc()
		if err != nil {
			ErrorsTotal.WithLabelValues("grpc", code.String()).Inc()
		}
		return resp, err
	}
}

// StreamServerInterceptor is UnaryServerInterceptor's streaming counterpart.
func StreamServerInterceptor() grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		var err error
		observeDuration(GRPCRequestDuration.WithLabelValues(info.FullMethod), func() {
			err = handler(srv, ss)
		})
		code := status.Code(err)
		GRPCRequestsTotal.WithLabelValues(info.FullMethod, code.String()).Inc()
		if err != nil {
			ErrorsTotal.WithLabelValues("grpc", code.String()).Inc()
		}
		return err
	}
}
