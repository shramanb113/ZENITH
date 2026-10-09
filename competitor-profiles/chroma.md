# Chroma — Competitor Profile

**URL**: https://www.trychroma.com/ (GitHub: github.com/chroma-core/chroma)
**Generated**: 2026-10-09
**Depth**: Deep profile

---

## At a Glance

| Metric | Value |
|--------|-------|
| Tagline | "Open-source search infrastructure for AI" |
| Founded | 2022 (pre-seed May 2022); SF-based |
| Headquarters | San Francisco, CA |
| Team size | Not disclosed (not found on site) |
| Funding | $18M seed (Apr 2023, Quiet Capital) + undisclosed pre-seed (May 2022). No later round found. |
| GitHub stars | 29.5k (live, 2026-10-09) |
| GitHub forks | 2.6k |
| Self-reported downloads | "15M+ monthly downloads" (homepage stat) vs "11M times a month" (same homepage, OSS section) — inconsistent |
| License | Apache 2.0 |

---

## Positioning & Messaging

**Primary value proposition**: "Open-source search infrastructure for AI. Fast, serverless, and scalable infrastructure supporting vector, full-text, regex, and metadata search. Built on object storage and trusted by millions of developers."

**Target audience**: AI/RAG application developers, from solo builders (free local start) to enterprises needing BYOC/compliance (Capital One, UnitedHealthcare named as customers).

**Positioning angle**: "Serverless, object-storage-native search infra" — explicitly pitches itself as a database built on S3/GCS with automatic tiering, not a managed cluster. Messaging has shifted from "embedding database" (2023 framing) to "search infrastructure for AI" (2026), reflecting the Oct 2025 addition of first-class sparse/lexical (BM25, SPLADE) search alongside vector.

**Key messaging themes**:
- Cost: "Vectors are large: 1GB text → 15GB of vectors... Memory is expensive: $5/GB/mo... Object storage is not: $0.02/GB/mo" (homepage)
- Zero-ops: "Chroma is a database you'll want to be on-call for... Auto-scales with usage, no manual tuning, serverless pricing"
- Enterprise trust: BYOC in customer VPC, multi-region replication, point-in-time recovery, SOC 2 Type II (trychroma.com/enterprise)

---

## Product & Features

### Core capabilities
- Vector (semantic) search
- Sparse vector / lexical search — BM25, SPLADE (added as "first class support," Oct 2025 changelog)
- Full-text search — trigram and regex search
- Metadata search — filtering and faceted search
- Forking — dataset versioning, A/B testing, copy-on-write collection duplication (Aug 2025)
- CLI tooling; clients in Python, TypeScript, Rust

### Notable differentiators
- Object-storage-native architecture (S3/GCS) with automatic hot/warm/cold tiering, pitched as 10x cheaper than memory-resident vector DBs at scale
- "Package Search MCP" (Sep 2025) — Chroma's own shipped MCP server, used to query thousands of open-source repos through MCP (a code/package search use case, not a general collections-API MCP server)
- wal3 — custom write-ahead log built directly on object storage (engineering blog, Sep 2025)
- Rust-based "push-based, morsel-driven" query execution engine (Aug 2025)

### Integrations
Not separately enumerated on a dedicated integrations page found; product surface implies LangChain/LlamaIndex-style framework usage is standard (not independently verified in this pass).

### Product direction signals (changelog, Jun 2025 – Mar 2026)
Heavy 2025-2026 investment in: Cloud productization (Cloud Sync, Web Sync, GitHub repo sync), enterprise controls (CMEK, private networking/PrivateLink, read-level consistency), and lexical/sparse search parity with vector. Direction = becoming a full managed "search infra" platform, not just a local embedding store.

---

## Pricing

| Tier | Price | Key Inclusions |
|------|-------|---------------|
| Starter | $0/mo + usage ($5 free credits) | 10 databases, 10 team members, community Slack |
| Team | $250/mo + usage ($100 included credits) | 100 databases, 30 team members, Slack support, SOC II, volume discounts |
| Enterprise | Custom | Unlimited DBs/members, dedicated support, single-tenant clusters, BYOC clusters, SLAs |

**Usage rates**: Write $2.50/GiB · Storage $0.33/GiB/mo · Query $0.0075/TiB · Network egress $0.09/GiB.
**Billing**: usage-based, monthly, pro-rated on plan change.
**Free trial**: Yes — $5 free credits, no card required to start locally/OSS.
**Notable**: Chroma publishes a live pricing calculator with realistic example math ($79/mo for ~1M docs written, 6M stored, 10M queries at ~4qps) — unusually transparent vs. the other four competitors profiled, three of which (LanceDB, parts of Qdrant/Weaviate Premium) require "contact sales."

---

## Customers & Social Proof

**Named customers**: Capital One, UnitedHealthcare, Weights & Biases, Mintlify (case study), Propel AI (case study), Conduit, Cofounder, Medwise.
**Industries**: Enterprise/finance (Capital One), healthcare (UHC), developer tools (Mintlify, W&B).
**Case study themes**: Scaling RAG/search infra without managing clusters.
**Review ratings**: Not independently pulled in this pass (G2/Capterra not scraped — out of scope for allotted depth; Discord community is the primary support surface, 10K+ members per homepage).

---

## SEO & Content Strategy

**Note on methodology**: DataForSEO was not available for this research. Substituted free signals: GitHub stars/forks/watchers (live GitHub scrape), self-reported download/usage stats from the company's own homepage, and funding data via web search (Crunchbase/Business Insider/Tracxn/company blog) rather than backlink/keyword tooling. No independent traffic, keyword-ranking, or backlink-profile data is available for Chroma in this profile.

**Content strategy signals**: Active changelog (near-monthly cadence through 2025-2026), a dedicated "Research" content track (Context-1, Context Rot, Generative Benchmarking, Chunking Strategies, Embedding Adapters) positioning Chroma as a technical thought leader in retrieval/context-engineering, not just a product vendor. Heavy YouTube presence (dozens of linked talks/demos on reranking, lexical search, context engineering).

---

## Strengths & Weaknesses

### Strengths
- Real hybrid search in one engine as of late 2025: vector + BM25/SPLADE sparse + full-text regex + metadata filters (homepage, Oct 2025 changelog) — not vector-only.
- Transparent, self-serve usage-based pricing with a public calculator (rare among profiled competitors).
- Already shipped a production MCP server (Package Search MCP), demonstrating MCP-surface investment ahead of ZENITH.
- Named enterprise customers (Capital One, UnitedHealthcare) suggest real production traction beyond hobbyist use.

### Weaknesses
- Internally inconsistent self-reported metrics on its own homepage (27k vs 26k GitHub stars; 15M vs 11M downloads/month, both stated on the same page) — a minor but real credibility/freshness gap.
- No publicly disclosed funding round since the April 2023 $18M seed (~3.5 years) — either quiet/undisclosed growth or a slower capital trajectory than Qdrant/Weaviate/LanceDB, all of which have disclosed later rounds within the last 1-2 years.
- True in-process/embedded mode exists only for Python's ephemeral `chromadb.Client()`; Chroma's own docs state this "starts a Chroma server in-memory" (their words), and the JS/TS and Rust clients have **no** embedded mode at all — getting-started docs explicitly instruct users to run `chroma run --path ...` or a Docker container first, then connect over HTTP.
- No apparent built-in phonetic/fuzzy matching comparable to ZENITH's Soundex + edit-distance fuzzy.

---

## Competitive Implications for ZENITH

**Where they're strong vs. us**: Already shipped hybrid lexical+vector+full-text in one engine with a transparent, generous self-serve pricing model, a thought-leadership content program, and a live MCP server product. Enterprise BYOC/VPC story is already built out.

**Where we're strong vs. them**: True in-process embedding for every language ZENITH supports (Go binary, no sidecar) vs. Chroma's client-server requirement for JS/TS/Rust and for any Python workload needing persistence. ZENITH also has phonetic (Soundex) + edit-distance fuzzy matching that Chroma's stack does not appear to offer.

**Opportunities**: Chroma's own docs use the word "server" for its own in-memory Python mode — this is a legitimate wedge to contrast against ("even Chroma's docs call it a server; ZENITH never starts one").

**Threats**: Chroma's sparse-vector (BM25/SPLADE) + full-text + metadata feature set closed much of the "real hybrid search" gap in the last year; a reader evaluating "does ZENITH do something Chroma can't" needs to lean on the embedded/no-sidecar and phonetic-fuzzy angles specifically, not "hybrid search" generally.

---

## Raw Data Sources

- Homepage scraped: 2026-10-09 → `raw/chroma/2026-10-09/scrapes/homepage.md`
- Pricing page scraped: 2026-10-09 → `raw/chroma/2026-10-09/scrapes/pricing.md`
- Docs getting-started scraped: 2026-10-09 → `raw/chroma/2026-10-09/scrapes/docs-getting-started.md`
- GitHub repo scraped: 2026-10-09 → `raw/chroma/2026-10-09/scrapes/github.md`
- Funding data pulled via firecrawl_search (web): 2026-10-09 → `raw/chroma/2026-10-09/seo/funding-search.md`
- No DataForSEO data used (not available) — see "SEO & Content Strategy" note above for substitutes.
- No prompt-injection or agent-directed text observed in any scraped Chroma page.
