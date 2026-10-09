# Sources: https://qdrant.tech/ and https://qdrant.tech/pricing/ — scraped 2026-10-09 (quick scan)

## Homepage
Tagline: "High-Performance Vector Search at Scale — Qdrant helps you build the AI retrieval you want. Ship high performance, full-feature vector search at any scale and with any deployment model."
Stats: "30k+ GitHub Stars" (self-reported, stale vs live 35.0k) / "60k+ Community Members" / "Rust Powered" / "SOC2 & HIPAA compliant"
Case studies linked: TripAdvisor (2-3x revenue), OpenTable, Deutsche Telekom (2M+ AI conversations), HubSpot/Breeze AI, Dust (5000+ data sources), Lyzr (90% latency reduction), Canva (quoted).

Feature highlights:
- "Native Hybrid Search (Dense + Sparse)" — "Blend keyword and vector search in one query – use dense or sparse vectors. Supports BM25, SPLADE++, and miniCOIL."
- "Built-in Multivector" — multiple vectors per object
- "Efficient, One-Stage Filtering" — "Filters are applied during HNSW traversal — no pre- or post-filtering."
- "Full-Spectrum Reranking" — score boosting, late interaction (ColBERT), MMR diversification
- Quantization: asymmetric/scalar/binary, "up to 64x" memory reduction
- Custom storage engine "Gridstore," built entirely in Rust with SIMD
- Native Cloud Inference (text/image embeddings in-cluster)
- Deployment modes: Qdrant Cloud (managed, AWS/GCP/Azure), Hybrid Cloud (BYO k8s), Private Cloud (air-gapped), Qdrant Edge (beta, "lightweight, low-latency vector search close to where data is generated")

## Pricing (qdrant.tech/pricing/)
- Free Tier: forever free, single-node, 0.5 vCPU / 1GB RAM / 4GB disk, free cloud inference w/ selected models
- Standard Tier: usage-based, dedicated resources, flexible vertical/horizontal scaling, HA setups, backup/DR, 99.5% uptime SLA
- Premium Tier: minimum spend required, SSO, private VPC links, 99.9% uptime SLA, extra support
- Hybrid Cloud: managed control plane, your infra/network/storage — for data residency / regulated workloads
- Private Cloud: dedicated isolated deployment, air-gapped
Billing: by vCPU + memory (GB) + storage (GB) + backup storage (GB) + paid inference tokens, hourly.
No embedded/in-process/library mode offered anywhere — Qdrant is always a running server (self-hosted binary/Docker, or a managed cluster).
