# 008. Embedding Provider: Google Gemini text-embedding-005
Status: Accepted; the model is now `gemini-embedding-001` (`text-embedding-005` is a Vertex AI model, not served by the Gemini API), and the Redis embedding cache described here was built and then removed: its hit rate was zero (2026-10-23) | Date: 2026-05-14

## Context
The RAG pipeline needs a text embedding model. Key constraints: must produce fixed-dimensional vectors compatible with pgvector, must be cost-effective at portfolio scale, and must be swappable if requirements change.

## Decision
Default to Google Gemini `text-embedding-005` (768 dimensions) with raw `httpx` for API calls and a Redis cache keyed by `{model}:{dims}:{sha256(text)}`.

**Why Gemini over alternatives.** Gemini `text-embedding-005` scores competitively on MTEB benchmarks and costs roughly 30× less than Voyage AI (the next obvious choice) at equivalent query volume. At portfolio scale (thousands of documents, not millions) neither latency SLA nor cost is a constraint — but demonstrating cost-awareness is. OpenAI `text-embedding-3-small` is comparable in price and quality but introduces a second vendor dependency given we already use Anthropic; Gemini keeps AI providers to two.

**Why raw httpx over a provider SDK.** Provider SDKs bundle connection pooling, retry, and batching in opinionated ways. Using raw `httpx` means: explicit control over Gemini's 100-request `batchEmbedContents` limit (OpenAI's is 2048 — a silent correctness difference); explicit exponential backoff on 429 with no hidden surprises; and trivial provider switching — changing URL, request body shape, and batch limit is three localized edits in `EmbeddingService`, not a dependency swap.

**Why the cache key includes model and dimensions.** `EMBEDDING_DIMENSIONS` is a `Settings` field (default 768) shared by four components: the `Vector(DIMS)` SQLAlchemy column, the Alembic migration, the `EmbeddingService`, and the Redis cache key. If a dimension change is deployed, the new key automatically misses all old cached embeddings from the previous dimension, preventing wrong-dimension vectors from being returned. Re-embedding every document (re-running `embed_chunks`, which replaces a document's whole chunk set in one transaction) then fills the cache with correctly-sized vectors.

**Why EMBEDDING_DIMENSIONS in Settings rather than hardcoded.** A hardcoded value would require touching four separate files to switch providers. The Settings singleton makes it one change with a single source of truth. The SQLAlchemy model reads it at import time via a module-level constant (`EMBEDDING_DIMS = get_settings().EMBEDDING_DIMENSIONS`) — this is the only way to provide a fixed column size to SQLAlchemy's declarative base without deferred column definitions.

## Consequences
Switching to a different embedding provider (e.g., OpenAI, Voyage, Cohere) requires: updating `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, and the relevant API key in Settings; running a new Alembic migration to resize the `Vector` column; and re-embedding all documents. Old- and new-dimension vectors can't share one `Vector(N)` column, so the resize migration must drop the old embeddings, and each document stays unsearchable until its re-embedding swaps in a new set. *(Revised 2026-10-24: a per-chunk `version` tag used to separate generations; it was dropped, see the plan's step 5.)* Rejected alternative: a vendor-managed embedding API like Pinecone or Weaviate's hosted embeddings would remove the caching concern but introduce a second hosted data store, conflicting with the architectural goal of PostgreSQL as the single source of truth.
