package metrics

import (
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Route wraps h so every request through it is counted and timed under
// pattern (the exact mux pattern string, e.g. "POST /v1/collections/{id}/search" —
// callers must pass the same pattern string to both mux.Handle and Route, so
// the label matches the route actually registered, not the literal request
// path, which would blow up cardinality with real document/collection ids).
func Route(pattern string, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
		observeDuration(HTTPRequestDuration.WithLabelValues(pattern), func() {
			h.ServeHTTP(sw, r)
		})
		HTTPRequestsTotal.WithLabelValues(pattern, strconv.Itoa(sw.status)).Inc()
	})
}

// statusWriter captures the status code a handler sends, defaulting to 200
// (http.ResponseWriter's own documented default when WriteHeader is never
// called) since Go's ResponseWriter has no way to read it back otherwise.
type statusWriter struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (w *statusWriter) WriteHeader(code int) {
	if !w.wroteHeader {
		w.status = code
		w.wroteHeader = true
	}
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Write(b []byte) (int, error) {
	w.wroteHeader = true
	return w.ResponseWriter.Write(b)
}

// Handler returns the /metrics endpoint for Registry.
func Handler() http.Handler {
	return promhttp.HandlerFor(Registry, promhttp.HandlerOpts{})
}
