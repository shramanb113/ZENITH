# Weaviate — Competitor Profile

**URL**: https://weaviate.io/ (GitHub: github.com/weaviate/weaviate)
**Generated**: 2026-10-09
**Depth**: Quick scan (homepage + pricing + GitHub + funding only, per brief)

---

## At a Glance

| Metric | Value |
|--------|-------|
| Tagline | "The AI database developers love" — "Design, build and ship complete AI experiences" |
| Funding | $1.2M seed (2022) + $50M Series B (Apr 2023, led by Index Ventures w/ Battery Ventures). Total disclosed $50-67.7M depending on source (Clay.com's $67.7M total vs. the $1.2M+$50M=$51.2M sum implies an unexplained ~$16.5M gap — likely an undisclosed/earlier round; no distinct "Series A" was found/named). |
| GitHub stars | 16.9k | Forks | 1.4k | Watchers | 141 |
| License | **Dual-licensed**: Community Edition BSD-3-Clause; Enterprise Edition features (in the `wl/` folder) require a commercial license key — not pure open source for all features. |

---

## Positioning & Messaging

Repositioned (2026) from "vector database" to a broader platform: "Vector search, RAG, and memory — all in one open-source platform," anchored by four pillars (Vector Database, Query Agent, Embeddings, and a new "Engram" personalization/memory product). "20M+ open source downloads," enterprise logos: Bosch, Cisco, HPE, Booking, Bumble, Deel, Akamai. Named case studies: Loti (9B vectors in production), DocsBot (50K+ tenants), Finster AI (42M vectors, 1-day deployment), Instabase (450+ data types).

---

## Product & Features

- `collection.query.hybrid(query=..., alpha=0.75)` — hybrid (vector+keyword/BM25) search is a first-class built-in API on every plan, not a bolt-on or paid add-on.
- Built-in generative search (RAG) and reranking modules; built-in embeddings service (e.g., Snowflake Arctic-Embed, billed per-token).
- Deployment: Docker, Kubernetes, or Weaviate Cloud — **no embedded/in-process library mode.**
- Ships "Weaviate Agent Skills" (`npx skills add weaviate/agent-skills`) for coding agents (Claude Code, Cursor, Copilot) — an agent-tooling integration bet, though skills/snippets-based rather than a protocol-compatibility shim like an MCP server or a Chroma-style REST-compat layer.

---

## Pricing

| Tier | Price | Key Inclusions |
|------|-------|---------------|
| Free | $0/mo forever | 1 cluster/user, 100K objects, 1GB memory, 10GB disk, basic support |
| Flex | From $45/mo | Pay-as-you-go, shared cluster, RBAC, 99.5% uptime, next-business-day Sev1 support |
| Premium | From ~$400/mo (per FAQ) | Dedicated or shared, prepaid contract, up to 99.95% uptime, 1-hour Sev1 support |

**Notable**: Hybrid search is included on literally every tier, even Free. Billed dimensions: vector dimensions, storage, backup, each priced separately and varying by cloud/region.

---

## Strengths & Weaknesses

### Strengths
- Hybrid search universally included (not gated behind a paid tier); strong enterprise logo roster; actively launching new product lines (Query Agent, Engram) rather than standing still.

### Weaknesses
- Not fully open source — Enterprise Edition features require a commercial license key, undercutting a pure-MIT/Apache "always free and open" narrative.
- No embedded/in-process mode at any tier — real infra cost floors exist even at the low end ($45/mo Flex minimum, ~$400/mo+ Premium minimum per the pricing FAQ).
- Funding figures are inconsistent across sources ($50M vs. $67.7M total) — a transparency/documentation gap.

---

## Competitive Implications for ZENITH

**Where they're strong vs. us**: Broad platform ambition (memory/personalization via Engram, NL-to-query via Query Agent), hybrid search included free, strong brand ("database developers love").
**Where we're strong vs. them**: ZENITH has no paywalled "Enterprise Edition" split — its entire engine is one codebase; it also has no infra cost floor since it never runs a cluster. Weaviate's real pricing minimums ($45-$400+/mo even before heavy usage) support ZENITH's margin argument by contrast — an embedded engine with no managed cluster has no equivalent fixed infra cost to pass through.
**Threats**: Weaviate's "all AI experiences in one platform" ambition (vector + RAG + memory + NL query) is a broader scope than ZENITH's search-engine-focused wedge; if a buyer wants a single platform for everything, Weaviate's breadth could out-compete a narrower embedded-search pitch on scope alone, even if ZENITH wins on infra simplicity.

---

## Raw Data Sources

- Homepage + pricing + GitHub scraped: 2026-10-09 → `raw/weaviate/2026-10-09/scrapes/homepage-and-pricing.md`
- Funding data via firecrawl_search: 2026-10-09 → `raw/weaviate/2026-10-09/seo/funding-search.md`
- No DataForSEO data used (not available). No prompt-injection/agent-directed text observed.
