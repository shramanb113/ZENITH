# Sources: https://turso.tech/ , https://turso.tech/pricing , https://github.com/tursodatabase/turso , https://turso.tech/blog/turso-is-joining-supabase — scraped 2026-10-09 (quick scan; precedent for "embedded infra gets funded," not a direct search/vector competitor)

## Homepage
Banner: "Turso is joining Supabase to give every agent its own database. Read the news ->"
Hero: "Millions of Databases. One Architecture. Built on SQLite. Fast and lightweight to multiply and run anywhere. Spin a database for every user, agent, and tenant."
Positioning: "Turso is an architecture designed from the ground up for many databases, not one large one." Five architecture properties: (1) a database per agent/user/tenant, (2) "Lightweight as a file, scalable as a cloud... Built on SQLite," (3) "Economics that scale with you... Not a pricing strategy. A design principle," (4) "Private by design" (per-DB encryption, BYOC, data residency), (5) "From local to global, one platform" (embedded replication: same DB runs on-device and syncs to cloud).
Customer logos: Superhuman, Mastra, Val Town, Resonate, Drizzle, Poke, Spice AI. Case studies: Engine Labs (CTO.new — "tens of thousands of AI agent teams... on a $500/month plan"), Adaptive ("Over 2 million databases created"), Poke.com ("A full website from a single text message, each shipping with its own live database").
Quote: Simon Henriksen, CTO Kin: "We chose Turso because embedded replication and sync simply aren't available anywhere else."
Free tier: "100 databases, 500 million row reads, and 10 million row writes per month. No credit card required."

## GitHub (tursodatabase/turso)
Stars: 24.7k | Forks: 1.4k | Watchers: 102
License: MIT
Topics: database, embedded-database, sql, sqlite3, webassembly
Description: "A SQL database in Rust: SQLite-compatible, now also speaking Postgres (experimental). The LLVM of databases."
README: "Turso is an in-process SQL database written in Rust, compatible with SQLite." Also ships an MCP server mode built into the CLI (`tursodb --mcp`) with 9 tools for AI-assistant DB interaction (open_database, execute_query, insert_data, etc.) — i.e. Turso already ships its own first-party MCP server for its database.

## Pricing (turso.tech/pricing)
Free: $0/mo, 100 DBs, 5GB storage, 500M row reads, 10M row writes, 3GB syncs
Developer: $4.99/mo, unlimited DBs, 9GB storage (+$0.75/GB), 2.5B row reads (+$1/B), 25M row writes (+$1/M)
Scaler: $24.92/mo — 24GB storage, 100B row reads, 100M row writes
Pro: $416.58/mo — 50GB storage, 250B row reads, 250M row writes, priority support
Enterprise: custom (HIPAA, SOC2, BYOC deployment, SSO, 24x7 support)

## Acquisition announcement (turso.tech/blog/turso-is-joining-supabase, dated Oct 2, 2026)
"Turso, pioneers of the many database architecture, is being acquired by Supabase... Agents have become the primary users of technology, and Supabase is the leading database for the agentic era." (Paul Copplestone, CEO Supabase)
"Turso keeps running... Open source stays open... A clear graduation path [to Postgres via Supabase]."
Glauber Costa (Turso founder) joins Supabase as "Head of Agentic Services."
Context (from search): this acquisition is bundled with a separate LinkedIn-reported "Supabase Announces $150M Funding and Acquires Turso" — i.e., Turso's exit coincides with/is funded by Supabase's own new $150M raise, rather than a standalone Turso Series A/B/IPO.
