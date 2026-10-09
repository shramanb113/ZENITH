# Source: https://www.trychroma.com/ — scraped 2026-10-09 (firecrawl_scrape, maxAge=0)
# NOTE: condensed to key excerpts (full scrape ~55K chars); long image/video/changelog lists trimmed.

## Hero
"Open-source search infrastructure for AI"
"Fast, serverless, and scalable infrastructure supporting vector, full-text, regex, and metadata search. Built on object storage and trusted by millions of developers. Open-source Apache 2.0."
CTA: "Start free on Cloud" / "get started locally"

## Social proof logos shown
Capital One, Mintlify (case study), UnitedHealthcare, Conduit, Propel (case study), Cofounder, Weights & Biases, Medwise

## Stats widget
"15M+ monthly downloads" / "Apache 2.0" / "27k Github stars"
(Elsewhere on same page, OSS community section states: "Chroma has over 26k GitHub stars and is used in over 90k other open-source codebases on GitHub. It is downloaded over 11M times a month." — NOTE: internally inconsistent numbers across the same homepage: 27k vs 26k stars, 15M vs 11M downloads/month.)

## Features grid
- Sparse vector search — Lexical search (BM25, SPLADE)
- Vector search — Semantic similarity search
- Full-text search — Trigram and regex search
- Metadata search — Filtering and faceted search
- Forking — Dataset versioning, A/B testing, roll-outs
- CLI — command-line tools
Clients: TypeScript, Python, Rust

## Performance / architecture
Latency @384 dim, 100k vectors: Warm p50 20ms/p90 27ms/p99 57ms; Cold p50 650ms/p90 1.2s/p99 1.5s
Technical specs: Write throughput 30MB/s (2000+ QPS) per collection; concurrent reads 10 (200+ QPS); 1M collections/db; 5M records/collection; recall 90-100%
Architecture diagram: Query Layer (memory cache hot, SSD cache warm) <-> Storage Layer (S3/GCS cold: all vectors/metadata/indexes)
"Unlike legacy search systems, Chroma is a database you'll want to be on-call for." Auto-scales, no manual tuning, serverless pricing.
Cost framing: "Vectors are large: 1GB text -> 15GB of vectors"; "Memory is expensive: $5/GB/mo"; "Object storage is not: $0.02/GB/mo"

## Enterprise
"Chroma brings the security, compliance, education and operational model enterprises need with our Apache 2.0 architecture. BYOC in your VPC, multi-cloud/multi-region replication, point-in-time-recovery..."
ASCII diagram shows "YOUR VPC" (data plane, your data/your cloud) vs "CHROMA VPC" (control plane, managed by Chroma: monitoring, backups, ops). Explicitly: "BYOC in your VPC / Multi-region replication / 0-ops management"

## Support tiers
Open-source: 10K-person Discord community
All plans: "Helpful support direct from engineers on the Chroma team"
Pro plan: "Direct Slack communication"
Enterprise: "Customized SLAs ensure your team gets 24/7 assistance"

## Changelog highlights (most recent first, dates as shown)
- Chroma Cloud Sync (Mar 2026) — serverless data ingestion for Chroma Cloud
- Metadata Arrays (Feb 2026)
- Indexing Status (Jan 2026)
- Read Level (Jan 2026) — index-only vs full read consistency modes
- Private Networking (Jan 2026) — AWS PrivateLink
- GroupBy (Jan 2026)
- Customer-Managed Encryption Keys (Dec 2025)
- Chroma Web Sync (Nov 2025) — auto crawl/scrape/chunk/embed web pages
- Sparse Vector Search (Oct 2025) — "First class support for BM25 and SPLADE vectors"
- Introducing Chroma Sync (Oct 2025) — auto chunk/embed/index GitHub repos
- wal3: Chroma's Write-Ahead Log (Sep 2025) — WAL built on object storage
- Package Search MCP (Sep 2025) — "Query thousands of open-source repos through MCP" — Chroma's own MCP server product
- Collection Forking (Aug 2025)
- Introducing Chroma Cloud (Aug 2025) — GA
- Designing a query execution engine (Aug 2025) — push-based morsel-driven engine in Rust
- Regex Search Support (Jun 2025)
- JavaScript Client V3 (Jun 2025)

## Quick start
pip install chromadb / npm install chromadb
