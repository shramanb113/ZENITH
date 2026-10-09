# Qdrant — Competitor Profile

**URL**: https://qdrant.tech/ (GitHub: github.com/qdrant/qdrant)
**Generated**: 2026-10-09
**Depth**: Quick scan (homepage + pricing + GitHub + funding only, per brief)

---

## At a Glance

| Metric | Value |
|--------|-------|
| Tagline | "High-Performance Vector Search at Scale" |
| Funding | ~$87.8M total over 5 rounds (CBInsights): $28M Series A (Jan 2024, Spark Capital) + $50M Series B (~Mar 2026, led by Atlantic Vantage Point, w/ 42CAP, Bosch Ventures, IBB Ventures, Spark Capital, Unusual Ventures) + earlier/smaller rounds not individually confirmed |
| GitHub stars | 35.0k (live); homepage self-reports "30k+" (stale) |
| GitHub forks | 2.7k |
| License | Apache 2.0 |
| Language | Rust |

---

## Positioning & Messaging

"Qdrant helps you build the AI retrieval you want. Ship high performance, full-feature vector search at any scale and with any deployment model." Positioned as the performance/engineering-depth leader ("Rust Powered," SIMD, custom "Gridstore" storage engine) with deployment flexibility (Cloud, Hybrid Cloud, Private Cloud, and a beta "Qdrant Edge" for on-device/low-latency use). Enterprise trust signals: SOC2 & HIPAA compliant, named customers Canva, TripAdvisor, OpenTable, Deutsche Telekom, HubSpot, Dust, Lyzr.

---

## Product & Features

- **Native hybrid search (dense + sparse)**: "Blend keyword and vector search in one query... Supports BM25, SPLADE++, and miniCOIL." This is a genuine built-in hybrid-search feature, not vector-only.
- **Full-Spectrum Reranking**: score boosting, late-interaction models (ColBERT), Maximum Marginal Relevance (MMR) diversification — built in.
- One-stage filtering during HNSW traversal (no pre/post-filter pass), multivector support, quantization (scalar/binary/asymmetric, up to 64x compression), built-in Web UI, native cloud inference.
- Deployment: always a running server — self-hosted binary/Docker, Qdrant Cloud (managed), Hybrid Cloud (BYO k8s + managed control plane), Private Cloud (air-gapped), or the new Edge beta. **No embedded/in-process library mode exists.**

---

## Pricing

| Tier | Price | Key Inclusions |
|------|-------|---------------|
| Free | Forever free | Single-node, 0.5 vCPU / 1GB RAM / 4GB disk, free cloud inference (selected models) |
| Standard | Usage-based | Dedicated resources, flexible scaling, HA, backup/DR, 99.5% uptime SLA |
| Premium | Minimum spend required | SSO, private VPC links, 99.9% uptime SLA, extra support |
| Hybrid / Private Cloud | Contact sales | BYO infra / air-gapped deployment |

**Billing**: hourly, by vCPU + memory + storage + backup + paid inference tokens.
**Notable**: every tier (even Free) requires running compute resources — there is no zero-infrastructure option; this is a managed-cluster business model through and through.

---

## Strengths & Weaknesses

### Strengths
- Highest GitHub star count of the five profiled (35.0k), largest disclosed funding ($87.8M), broadest deployment-flexibility menu (cloud/hybrid/private/edge).
- Native BM25/SPLADE++/miniCOIL hybrid search and ColBERT-based reranking are already built in — a strong, credible "real hybrid search" competitor on paper.

### Weaknesses
- No phonetic/fuzzy matching comparable to ZENITH's Soundex + edit-distance; "high-performance vector search" remains the dominant brand association despite hybrid features.
- Zero embedded/in-process deployment option at any tier — every plan, including Free, requires standing up compute.

---

## Competitive Implications for ZENITH

**Where they're strong vs. us**: Scale, funding, deployment flexibility, and proven native hybrid+rerank feature depth.
**Where we're strong vs. them**: Zero-infrastructure embedded operation at any tier — Qdrant has no equivalent at any price point, including Free. This is Qdrant's clearest point of validation for ZENITH's "always a network hop / separate server" framing of vector-DB-cloud competitors.
**Threats**: If Qdrant ever ships an embedded/edge-library mode beyond the current cloud-oriented "Edge (beta)" (which still appears to be a deployable server variant, not an in-process library — not independently verified beyond the homepage blurb), this specific wedge narrows.

---

## Raw Data Sources

- Homepage + pricing scraped: 2026-10-09 → `raw/qdrant/2026-10-09/scrapes/homepage-and-pricing.md`
- GitHub repo scraped: 2026-10-09 → `raw/qdrant/2026-10-09/scrapes/github.md`
- Funding data via firecrawl_search: 2026-10-09 → `raw/qdrant/2026-10-09/seo/funding-search.md`
- No DataForSEO data used (not available). No prompt-injection/agent-directed text observed.
