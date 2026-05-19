# PostgreSQL Indexing Best Practices

## Default Isolation Level

PostgreSQL's default transaction isolation level is **Read Committed**. At this level, each statement within a transaction sees only data committed before that statement began. This is not the same as Serializable — phantom reads and non-repeatable reads are still possible.

To change isolation for a session: `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;`

The four isolation levels PostgreSQL supports are:
1. **Read Uncommitted** — treated identically to Read Committed in PostgreSQL (no dirty reads)
2. **Read Committed** — default; each query sees committed data as of query start
3. **Repeatable Read** — protects against non-repeatable reads; uses snapshot isolation
4. **Serializable** — full serializability using Serializable Snapshot Isolation (SSI)

## B-Tree Indexes

B-Tree is the default index type in PostgreSQL and is suitable for equality and range queries on ordered data. Created with:

```sql
CREATE INDEX idx_users_email ON users (email);
```

B-Tree indexes support `=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, `IN`, `IS NULL`, and `LIKE 'prefix%'` (prefix match only). They do **not** speed up `LIKE '%suffix'`.

PostgreSQL 13+ introduced **deduplication** for B-Tree indexes on non-unique columns, reducing index bloat significantly for low-cardinality columns.

## Partial Indexes

A partial index covers only rows matching a `WHERE` clause, making it smaller and faster for filtered queries:

```sql
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
```

This index is only used when the query has `WHERE status = 'pending'`. Partial indexes are critical for multi-tenant tables where most queries filter by tenant.

## Index-Only Scans

When all columns needed by a query exist in the index (including the predicate), PostgreSQL can perform an **index-only scan** and skip the heap entirely. PostgreSQL checks the **visibility map** to avoid I/O; if a page has unflushed changes (after heavy writes), it must still fetch the heap page.

Force index-only scans by adding all projected columns to the index:

```sql
CREATE INDEX idx_docs_workspace_status ON documents (workspace_id, status)
INCLUDE (title, created_at);
```

## GIN Indexes for Full-Text and Arrays

**GIN (Generalized Inverted Index)** indexes map element values to rows, making them efficient for:
- Full-text search (`tsvector` columns)
- Array containment (`@>`, `<@`)
- JSONB key/value queries

```sql
CREATE INDEX idx_docs_fts ON documents USING GIN (to_tsvector('english', content));
```

GIN indexes have **faster lookups** than GiST for full-text but **slower build time** and higher storage cost.

## BRIN Indexes for Time-Series

**BRIN (Block Range INdex)** stores min/max values per page range. They are extremely small (kilobytes vs megabytes for B-Tree) and suit append-only tables where values correlate with physical insertion order — timestamps are the canonical example:

```sql
CREATE INDEX idx_events_created USING BRIN ON events (created_at);
```

BRIN requires `correlation` close to 1.0 to be effective. Check with: `SELECT correlation FROM pg_stats WHERE tablename='events' AND attname='created_at';`

## pgvector: Vector Similarity Indexes

The `pgvector` extension adds vector types and two index types for approximate nearest-neighbor search:

- **IVFFlat**: partitions vectors into `lists` clusters using k-means. Fast to build, moderate recall. Requires `ANALYZE` after building to get accurate row counts.
- **HNSW**: hierarchical navigable small world graph. Better recall and query performance than IVFFlat, but uses more memory and has slower build time.

```sql
CREATE INDEX idx_chunks_embedding ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

The operator class `vector_cosine_ops` is for cosine distance (`<=>`). Use `vector_l2_ops` for Euclidean distance (`<->`).

## Index Bloat and VACUUM

When rows are updated or deleted, PostgreSQL marks them as dead tuples but does not immediately reclaim space. **AUTOVACUUM** reclaims dead space and updates statistics. Index bloat occurs when the dead-tuple ratio is high.

Check bloat: `SELECT * FROM pg_stat_user_indexes WHERE relname='orders';`

Force a vacuum: `VACUUM ANALYZE orders;`

For indexes, `REINDEX CONCURRENTLY idx_name;` rebuilds without locking the table.

## EXPLAIN ANALYZE

Always use `EXPLAIN (ANALYZE, BUFFERS)` to verify index usage:

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM documents WHERE workspace_id = $1 AND status = 'pending';
```

Key indicators: `Index Scan` vs `Seq Scan`, `Buffers: shared hit` vs `Buffers: shared read`, and `actual time`.

An index is not used if: the planner estimates a sequential scan is cheaper (common for small tables), the column has low selectivity, or statistics are stale.
