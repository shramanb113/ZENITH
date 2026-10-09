# Turso — Competitor Profile (precedent, not a direct competitor)

**URL**: https://turso.tech/ (GitHub: github.com/tursodatabase/turso)
**Generated**: 2026-10-09
**Depth**: Quick scan — included per brief as "the embedded infra gets funded" precedent (SQLite-as-a-service), not a search/vector competitor.

---

## At a Glance

| Metric | Value |
|--------|-------|
| Tagline | "Millions of Databases. One Architecture. Built on SQLite." |
| Funding (disclosed) | $7M seed (May 2023), backed by Firestreak, Essence Venture Capital, Fire Star Ventures, Mango Capital. No Series A/B ever publicly disclosed. |
| **Major event** | **Acquired by Supabase, announced Oct 2, 2026** — bundled with Supabase's own newly-announced $150M funding round, rather than an independent Turso round or IPO. |
| GitHub stars | 24.7k | Forks | 1.4k | Watchers | 102 |
| License | MIT |

---

## Positioning & Messaging

"Turso is an architecture designed from the ground up for many databases, not one large one" — a database-per-agent/user/tenant model built on a from-scratch Rust rewrite of SQLite, with embedded replication (same DB runs on-device and syncs to the cloud) and "economics that scale... not a pricing strategy, a design principle." GitHub repo description: "An in-process SQL database written in Rust, compatible with SQLite" (GitHub topic: `embedded-database`). Already ships its own first-party MCP server mode built into the CLI (`tursodb --mcp`, 9 tools for AI-assistant DB interaction).

---

## Pricing (for context)

Free ($0, 100 DBs/5GB) → Developer ($4.99/mo) → Scaler ($24.92/mo) → Pro ($416.58/mo) → Enterprise (custom). Usage-based overages on storage/rows-read/rows-written/syncs; "idle databases cost only storage," explicitly designed for millions of mostly-idle per-tenant databases.

---

## Why This Matters as a Precedent

**Validates the thesis, tempers the scale expectation**: Turso is the cleanest available example of "an embedded/in-process architecture is valuable enough that a major platform will acquire it" — Supabase's CEO explicitly framed the deal as wanting Turso's embedded/per-agent database architecture for the "agentic era." That is a genuine validation of embedded-infra-as-a-wedge.

However, two things temper how far this precedent should be stretched in a pitch deck:
1. Turso's disclosed funding base was modest (~$7-9M) relative to the vector-DB competitors profiled here (Qdrant $87.8M, Weaviate $50-67.7M, LanceDB $38-41M) — it did not raise a large Series A/B before its exit.
2. The exit was an **acquisition/consolidation into a larger platform**, not an independent scale-up to a standalone large company or IPO. This is a real, positive outcome, but the more honest framing is "embedded infra is acquisition-attractive to larger platforms" rather than "embedded infra reliably becomes a giant standalone company" — a distinction worth being precise about in the pitch.

---

## Raw Data Sources

- Homepage + pricing + GitHub + acquisition blog post scraped: 2026-10-09 → `raw/turso/2026-10-09/scrapes/homepage-and-github.md`
- Funding data via firecrawl_search: 2026-10-09 → `raw/turso/2026-10-09/seo/funding-search.md`
- No DataForSEO data used (not available). No prompt-injection/agent-directed text observed.
