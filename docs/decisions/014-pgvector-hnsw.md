# 014. pgvector HNSW over IVFFlat for Embedding Search
Status: Accepted | Date: 2026-05-25

## Context
The `document_chunks` table stores 768-dimensional embeddings (Gemini `text-embedding-005`) and is queried via cosine similarity for every RAG retrieval. pgvector offers two approximate nearest-neighbour index types: HNSW (Hierarchical Navigable Small World) and IVFFlat (Inverted File with Flat quantisation). The index type determines recall accuracy, query latency, and the operational complexity of index maintenance.

## Decision
Use HNSW (`USING hnsw (embedding vector_cosine_ops)`) with default `m=16` and `ef_construction=64`. The migration creates the index at table creation time: `CREATE INDEX ix_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)`. The query uses cosine distance (`<=>`) which matches the index operator class.

## Consequences
HNSW delivers better recall at low query latency without a training phase. New documents can be inserted and immediately appear in ANN search results; the index updates incrementally at write time. IVFFlat requires building cluster centroids over a representative dataset before the index becomes useful — on a freshly seeded workspace, IVFFlat would return poor results until enough documents exist. The tradeoff is higher memory usage per index page (HNSW stores the graph structure); at 768 dims and typical portfolio data volumes this is negligible.

Rejected alternative: IVFFlat with `lists=100`. While IVFFlat has lower memory overhead at scale, it requires a `VACUUM ANALYZE` + `SET ivfflat.probes` tuning step and degrades recall significantly when the table is sparsely populated. For a portfolio project where workspaces start empty, HNSW's no-training-required property is the deciding factor.
