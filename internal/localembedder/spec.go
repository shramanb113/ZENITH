package localembedder

import "github.com/shramanb113/ZENITH/internal/modelspec"

// The model registry lives in internal/modelspec so tooling can read it without
// compiling this package (whose go:embed needs the model files to exist). These
// aliases keep the names callers already use.
type (
	Spec    = modelspec.Spec
	Pooling = modelspec.Pooling
)

const (
	PoolCLS = modelspec.PoolCLS
)

// Models returns every registered model, sorted by ID.
func Models() []Spec { return modelspec.Models() }

// Lookup finds a model by ID (case-insensitive).
func Lookup(id string) (Spec, error) { return modelspec.Lookup(id) }

// RerankerSpec describes a cross-encoder reranker (see internal/modelspec).
type RerankerSpec = modelspec.RerankerSpec

// RerankerModels returns every registered reranker, sorted by ID.
func RerankerModels() []RerankerSpec { return modelspec.RerankerModels() }

// LookupReranker finds a reranker by ID (case-insensitive).
func LookupReranker(id string) (RerankerSpec, error) { return modelspec.LookupReranker(id) }
