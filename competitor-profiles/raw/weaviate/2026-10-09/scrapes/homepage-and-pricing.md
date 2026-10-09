# Sources: https://weaviate.io/ , https://weaviate.io/pricing , https://github.com/weaviate/weaviate — scraped 2026-10-09 (quick scan)

## Homepage (repositioned 2026)
Title: "The AI database developers love | Weaviate"
Hero: "Design, build and ship complete AI experiences — Vector search, RAG, and memory - all in one open-source platform."
New product banner: "Weaviate Engram — Personalized AI experiences" (memory/personalization product)
"Platform Services" = 4 pillars: Vector Database, Query Agent (NL-to-query), Embeddings (built-in vectorization), Engram (personalization/memory)
Social proof: "20M+ open source downloads," customer logos incl. Bosch, Cisco, HPE, Booking, Bumble, Deel, Akamai; named case studies: Loti (9B vectors in production), DocsBot (50K+ tenants, 6.1M+ questions answered), Finster AI (42M vectors, 1-day enterprise deployment), MarvelX (banking security), Instabase (450+ data types, 50K+ tenants)
Code sample shows `collection.query.hybrid(query=..., alpha=0.75, limit=5)` — hybrid search (vector+keyword) is a first-class built-in API, not a bolt-on.
"Why Weaviate": AI-first features under one roof, billion-scale architecture, RBAC/SOC2/HIPAA, production-ready for enterprise.

## GitHub (weaviate/weaviate)
Stars: 16.9k | Forks: 1.4k | Watchers: 141
License: dual — "Most of this repository is available under the BSD 3-Clause License. Files in our wl/ folder are only available under a commercial license with a license key... Enterprise features require a license key." (Community Edition BSD vs Enterprise Edition commercial, same Docker binary, EE features gated by license key.)
Description: "Weaviate is an open-source vector database that stores both objects and vectors... combination of vector search with structured filtering with the fault tolerance and scalability of a cloud-native database."
Deployment: Docker, Kubernetes, Weaviate Cloud — no embedded/in-process library mode; also ships community "Weaviate Agent Skills" for coding agents (npx skills add weaviate/agent-skills) — an agent-tooling integration play, somewhat analogous to an MCP-style integration bet but client-libraries/skills based rather than a protocol-compatibility shim.

## Pricing (weaviate.io/pricing)
- Free: $0/mo forever, 1 cluster/user, 100K objects, 1GB memory, 10GB disk, 1 collection/3 tenants, Embeddings (2000 req/day) + Query Agent (1000 req/mo), basic support
- Flex: starts $45/mo, pay-as-you-go, shared cluster, RBAC, 99.5% uptime, standard support (next-business-day Sev1)
- Premium: dedicated or shared, prepaid contract, up to 99.95% uptime, 1-hour Sev1 enterprise support, custom price (FAQ states minimum ~$400/mo on Premium)
Hybrid search is included on EVERY tier (per comparison table: "Hybrid search ✓" across Free/Flex/Premium-shared/Premium-dedicated).
Billed dimensions: vector dimensions (from $0.00465/1M on Flex down to $0.002718/1M on dedicated Premium), storage (from $0.12/GiB), backup (from $0.03036/GiB). Embeddings billed separately per-token (e.g., Snowflake Arctic-Embed-M $0.025/1M tokens).
