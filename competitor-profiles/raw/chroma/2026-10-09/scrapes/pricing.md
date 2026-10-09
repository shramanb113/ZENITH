# Source: https://www.trychroma.com/pricing — scraped 2026-10-09

"Simple, usage-based pricing. Just like the rest of Chroma Cloud."
"Chroma Cloud enables fast, scalable, & serverless vector, full-text, and metadata search across terabytes of data."

Usage rates:
- Write: $2.50 / GiB
- Storage: $0.33 / GiB / month
- Query: $0.0075 / TiB queried
- Network: $0.09 / GiB returned

Example calc shown on page (500 collections, vector dim 1536, doc 8192 bytes, metadata 64 bytes):
- Written: 1.0M docs (13GiB) = $34
- Stored: 6.0M docs (80GiB) = $27
- Queried: 10.0M queries (4qps) = $19
- Usage total/month: $79

Plans:
- Starter: $0/mo + usage. 10 databases, 10 team members, Community Slack. Includes $5 free credits.
- Team: $250/mo + usage. 100 databases, 30 team members, Slack support, SOC II, volume discounts. $100 included credits.
- Enterprise: Custom. Unlimited DBs/members, dedicated support, single-tenant clusters, BYOC clusters, SLAs.

FAQ highlights:
- Credits roll over (except Team's $100 included usage)
- Can export data on leaving — "Chroma is an open-source product and egressing your data is easy."
