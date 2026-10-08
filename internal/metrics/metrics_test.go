package metrics

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/shramanb113/ZENITH/internal/embedding"
)

// Route records the handler's actual status code, not always 200.
func TestRoute_RecordsActualStatusCode(t *testing.T) {
	h := Route("GET /x", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/x", nil))

	if got := testutil.ToFloat64(HTTPRequestsTotal.WithLabelValues("GET /x", "401")); got != 1 {
		t.Fatalf("requests_total{route=\"GET /x\",code=\"401\"} = %v, want 1", got)
	}
}

// Route defaults to 200 when the handler never calls WriteHeader.
func TestRoute_DefaultsTo200(t *testing.T) {
	h := Route("GET /y", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok"))
	}))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest("GET", "/y", nil))

	if got := testutil.ToFloat64(HTTPRequestsTotal.WithLabelValues("GET /y", "200")); got != 1 {
		t.Fatalf("requests_total{route=\"GET /y\",code=\"200\"} = %v, want 1", got)
	}
}

type namedStub struct{ name string }

func (namedStub) Embed(context.Context, string) ([]float32, error)          { return []float32{1}, nil }
func (namedStub) EmbedBatch(context.Context, []string) ([][]float32, error) { return nil, nil }
func (namedStub) Dimensions() int                                           { return 1 }
func (n namedStub) Name() string                                            { return n.name }

type unnamedStub struct{}

func (unnamedStub) Embed(context.Context, string) ([]float32, error)          { return []float32{1}, nil }
func (unnamedStub) EmbedBatch(context.Context, []string) ([][]float32, error) { return nil, nil }
func (unnamedStub) Dimensions() int                                           { return 1 }

// InstrumentEmbedder's Name() forwards to the wrapped embedder's Name() when
// it implements embedding.Named, exactly like CachingEmbedder already does —
// otherwise a wrapped index would hit ErrEmbedderMismatch.
func TestInstrumentEmbedder_NameForwards(t *testing.T) {
	w := InstrumentEmbedder(namedStub{name: "onnx:gte-small"})
	named, ok := w.(embedding.Named)
	if !ok {
		t.Fatal("InstrumentEmbedder's result must implement embedding.Named")
	}
	if got := named.Name(); got != "onnx:gte-small" {
		t.Fatalf("Name() = %q, want %q", got, "onnx:gte-small")
	}
}

func TestInstrumentEmbedder_UnnamedReportsUnknown(t *testing.T) {
	w := InstrumentEmbedder(unnamedStub{})
	named, ok := w.(embedding.Named)
	if !ok {
		t.Fatal("InstrumentEmbedder's result must implement embedding.Named")
	}
	if got := named.Name(); got != "unknown" {
		t.Fatalf("Name() = %q, want \"unknown\"", got)
	}
}

func TestInstrumentEmbedder_EmbedQueryCountsAsOpQuery(t *testing.T) {
	w := InstrumentEmbedder(namedStub{name: "x"})
	before := testutil.ToFloat64(EmbeddingTextsTotal.WithLabelValues("query"))
	if _, err := w.(embedding.QueryEmbedder).EmbedQuery(context.Background(), "hello"); err != nil {
		t.Fatal(err)
	}
	after := testutil.ToFloat64(EmbeddingTextsTotal.WithLabelValues("query"))
	if after != before+1 {
		t.Fatalf("texts_total{op=query} went from %v to %v, want +1", before, after)
	}
}

// gRPC unary interceptor: a handler returning a non-OK status is counted
// under that status's name on both requests_total and errors_total.
func TestUnaryServerInterceptor_CountsStatusAndError(t *testing.T) {
	interceptor := UnaryServerInterceptor()
	handler := func(ctx context.Context, req any) (any, error) {
		return nil, status.Error(codes.NotFound, "nope")
	}
	info := &grpc.UnaryServerInfo{FullMethod: "/zenith.SearchService/Search"}

	_, _ = interceptor(context.Background(), nil, info, handler)

	if got := testutil.ToFloat64(GRPCRequestsTotal.WithLabelValues("/zenith.SearchService/Search", "NotFound")); got != 1 {
		t.Fatalf("grpc_requests_total{method=...,code=NotFound} = %v, want 1", got)
	}
	if got := testutil.ToFloat64(ErrorsTotal.WithLabelValues("grpc", "NotFound")); got != 1 {
		t.Fatalf("errors_total{surface=grpc,code=NotFound} = %v, want 1", got)
	}
}

func TestUnaryServerInterceptor_OKIsNotAnError(t *testing.T) {
	interceptor := UnaryServerInterceptor()
	handler := func(ctx context.Context, req any) (any, error) { return "fine", nil }
	info := &grpc.UnaryServerInfo{FullMethod: "/zenith.SearchService/GetDocument"}

	_, err := interceptor(context.Background(), nil, info, handler)
	if err != nil {
		t.Fatal(err)
	}
	if got := testutil.ToFloat64(GRPCRequestsTotal.WithLabelValues("/zenith.SearchService/GetDocument", "OK")); got != 1 {
		t.Fatalf("grpc_requests_total{method=...,code=OK} = %v, want 1", got)
	}
}

// End to end: the registry's /metrics output carries build info and the Go
// runtime collector's series.
func TestHandler_ExposesBuildInfoAndGoCollector(t *testing.T) {
	BuildInfo.WithLabelValues("test-version", "test-model").Set(1)

	rec := httptest.NewRecorder()
	Handler().ServeHTTP(rec, httptest.NewRequest("GET", "/metrics", nil))

	body := rec.Body.String()
	if !contains(body, "zenith_build_info") {
		t.Fatal("/metrics missing zenith_build_info")
	}
	if !contains(body, "go_goroutines") {
		t.Fatal("/metrics missing go_goroutines (Go collector)")
	}
}

func contains(haystack, needle string) bool {
	return len(haystack) >= len(needle) && (func() bool {
		for i := 0; i+len(needle) <= len(haystack); i++ {
			if haystack[i:i+len(needle)] == needle {
				return true
			}
		}
		return false
	})()
}
