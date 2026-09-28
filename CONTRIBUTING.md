# Contributing to ZENITH

Thanks for looking. ZENITH is a small, single-maintainer project, so the goal of
this file is to make it cheap to get productive and cheap to get a change merged.

## Get running

```bash
git clone https://github.com/shramanb113/ZENITH && cd ZENITH
go build ./...
go test ./...                      # lexical mode, no C toolchain needed
```

Semantic search needs the ONNX model and runtime bundled at build time and a C
compiler (`CGO_ENABLED=1`; MinGW-w64 on Windows, gcc/clang elsewhere):

```bash
go run scripts/download_assets.go            # fetches model + onnxruntime for your OS
CGO_ENABLED=1 go test ./...
CGO_ENABLED=1 go test -race ./internal/index/ ./internal/ann/ ./internal/segment/
```

Both legs matter and both run in CI: **the lexical-only (no CGo) build is a
supported, first-class mode**, not a fallback to be left broken.

## Where things live

Read `CLAUDE.md` (architecture in one page) and `DECISIONS.md` (why). Briefly:

| Path | What |
|------|------|
| `pkg/zenith` | the public embedded API |
| `internal/index` | the search engine: lexical + vector passes, rank fusion, persistence |
| `internal/segment` | the immutable memory-mapped segment file format |
| `internal/ann` | HNSW vector index |
| `internal/analysis`, `internal/ranking` | tokenising, fuzzy/phonetic, BM25, RRF |
| `internal/localembedder`, `internal/modelspec` | in-process ONNX embedder and the model registry |
| `cmd/zenith`, `cmd/server` | CLI and gRPC server |
| `FORMAT.md` | the on-disk format and compatibility policy |

## Before you open a PR

1. `gofmt` your changes and run `go vet ./...`.
2. Add a test. Bug fixes need a test that fails without the fix. Prefer
   behaviour tests over tests that mirror the implementation.
3. Run the suite in both modes (`CGO_ENABLED=0` and `=1`).
4. If you touch the on-disk format, read `FORMAT.md` first: format changes
   need a migration from the previous version and a golden-file test.
5. If you touch ranking or the lexical pass, say what happened to Recall@10 and
   latency. `ZENITH_MSMARCO_LEX=1 go test ./internal/index -run TestMSMARCOLexicalLatency -v`
   measures both on real MS MARCO (`cd bench && go run ./cmd/fetch` downloads it
   once). "Faster" without a recall number is not an accepted claim here.

## Good first issues

Pulled from `ROADMAP.md`; each is self-contained:

- **Filter surface area**: expose attribute filters on the gRPC/HTTP/CLI surfaces
  (the library has them). `internal/index/attrs.go` has the predicate model.
- **A model for another language**: the registry (`internal/modelspec`) only lists
  models that share the BERT-uncased vocabulary. A SentencePiece tokenizer would
  unlock multilingual models.
- **Windows/Linux arm64 release builds**: `release.yml` builds amd64 and darwin
  arm64 with the model; add linux-arm64 (needs an onnxruntime arm64 entry in
  `scripts/download_assets.go`).
- **Benchmarks**: human-authored code-mixed (e.g. Hinglish) query sets — the
  current `codemixed` set is a synthetic proxy.

## Ground rules

- No telemetry, no phoning home, no license-gated core features (see the
  monetization guardrails in `ROADMAP.md`). Diagnostics are opt-in
  (`zenith doctor --json`) and contain no index contents.
- The embedded library stays zero-config: operational features (auth, quotas,
  metrics) belong in `cmd/`, not `pkg/zenith`.
- Data safety beats speed. A change that could overwrite or misread a user's
  index must fail loudly instead.
