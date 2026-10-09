# Source: https://docs.trychroma.com/docs/overview/getting-started — scraped 2026-10-09

Page meta description: "Chroma is the open-source data infrastructure for AI. It comes with everything you need to get started built-in, and runs on your machine."

## Python
`pip install chromadb`
`chroma_client = chromadb.Client()`
Docs quote (critical for "embedded" claim verification): "In this guide we used Chroma's in-memory client for simplicity. It starts a Chroma server in-memory, so any data you ingest will be lost when your program terminates. You can use the persistent client or run Chroma in client-server mode if you need data persistence."
-> Chroma's own docs describe even the simplest Python mode as starting "a Chroma server in-memory" (their own words). Persistent client/client-server mode needed for durability.

## TypeScript/JS
`npm install chromadb @chroma-core/default-embed`
Docs instruct: "Run the Chroma backend: npx chroma run --path ./getting-started" OR `docker run -p 8000:8000 chromadb/chroma`
Then: `const client = new ChromaClient();` — connects over HTTP to that running backend.
AI-agent prompt snippet for JS explicitly says: "You will have to run a local Chroma server to make this work... Make sure to instruct the user on how to start a local Chroma server in your summary."
-> For JS/TS, Chroma has NO embedded/in-process mode at all — always requires a locally running (or Docker) server process reached over HTTP, even for local dev.

## Rust
`cargo add chroma` — also connects via `ChromaHttpClient` to a `chroma run` backend. Same client-server requirement as TS.

Conclusion: "embedded, no sidecar" is true only for Python's ephemeral in-memory client; false for JS/TS/Rust clients and for any Python workload needing persistence beyond ephemeral testing (persistent/client-server mode is the documented production path).
