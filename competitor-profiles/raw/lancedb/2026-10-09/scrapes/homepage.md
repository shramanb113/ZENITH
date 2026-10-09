# Source: https://www.lancedb.com/ — scraped 2026-10-09 (firecrawl_scrape, maxAge=0)
# NOTE: condensed; full scrape ~60K chars, mostly a long blog-post roll (trimmed here).

## Page title / meta (as of 2026-10-09)
"LanceDB | Multimodal Lakehouse for AI"
"The multimodal lakehouse for AI, accelerating large-scale data curation and feature engineering so teams can build better models faster."
-> Current primary positioning is NOT "vector database" or "embedded search" — it is an ML training-data lakehouse.

## Hero / core pitch section
"One table for all your training data. Performant for any workload."
"Developing the right dataset is critical for model quality. Feeding that dataset to the GPU efficiently is essential for cost-effective training at scale."
Stat callouts: "70% MFU" (Model FLOPS Utilization) vs "10%" for others; "0K+ Queries per second" / "0B+ Rows in a single table" (animated counters, values not captured in static scrape)

## Four product pillars (homepage)
1. Curation — "Find the optimal distribution. Deduplicate hundreds of billions of rows. Identify edge cases for labeling."
2. Feature Engineering — "Write Python UDFs locally. Plug and run at petabyte scale with automatic updates... Built for agents and humans." (Geneva)
3. Training — "Accelerated training from the same table you explored and curated. Get up to 70% MFU with no egress bottleneck."
4. Search & Retrieval — "Vector/semantic, full-text, and hybrid search combined with SQL filters against one table. The unified layer for production-grade agentic retrieval." [CONFIRMS: native full-text + hybrid search exists, not vector-only]

## Customer quotes on homepage
- Rohit Khanna, VP Eng, CodeRabbit: "...other vector databases hit cost and performance walls, LanceDB scales effortlessly... sub-second response times across millions of code reviews."
- Noah Shpak, Character.ai: "Migrating full-text search from ElasticSearch to LanceDB reduced our p90 latency by over 90%."
- Nadia Ali, CFO, Midjourney: "Vector search is critical infrastructure... LanceDB was the only one that could meet the high-traffic and large scale requirements."
- Kamil Sindi, Head of Eng, Runway: "The ability to append columns without rewriting entire datasets, combined with fast random access and multimodal support..."
- Keunhong Park, World Labs: "...a dramatic step up from legacy formats like WebDataset and Parquet."

## Funding blog post referenced in blog roll
"LanceDB Raises $30M Series A to Build the Multimodal Lakehouse" (blog/series-a-funding) — "closed another funding round to accelerate development of the Multimodal Lakehouse... As of June 2025, Lance remains the fastest growing format..."

## Notable blog-roll items confirming product history/claims
- "LanceDB WikiSearch: Native Full-Text Search on 41M Wikipedia Docs" — "No more Tantivy! We stress-tested native full-text search..." (LanceDB originally used Tantivy for FTS, later built native FTS)
- "OpenSearch vs LanceDB for Vector Search: Query Cost and Infrastructure" — "Choosing a vector database usually comes down to a tradeoff between a full search service and an in-process library." [LanceDB's own framing, nearly identical to ZENITH's wedge]
- "AnythingLLM's Competitive Edge: LanceDB for Seamless RAG and Agent Workflows" — "leveraged LanceDB's serverless architecture to eliminate vector database setup complexity, enabling seamless cross-platform RAG and agent workflows with zero configuration required."
- "The Future of AI-Native Development is Local: Inside Continue's LanceDB-Powered Evolution" — "LanceDB's embedded TypeScript library, enabling lightning-fast semantic code search while maintaining complete developer privacy and offline capability."
- "Why LanceDB Is the Most Natural Memory Layer for OpenClaw" — "LanceDB fits that role with embedded deployment, filesystem-native storage, and multimodal retrieval."
- "Why CrewAI Rebuilt Agent Memory on LanceDB" — "ships by default across 12 million monthly downloads and 2B+ agent executions."
- Case studies referenced: Netflix (Media Data Lake), Harvey (enterprise RAG), Dosu, Cognee, WeRide (autonomous driving, "90x improvement in ML developer productivity"), ByteDance Volcano Engine (100K+ QPS agent memory), Second Dinner/Marvel Snap, Uber (multi-base layout).
