# 005. Cursor-based pagination with (created_at, id) composite key
Status: Accepted | Date: 2026-05-14

## Context
The document list endpoint needs pagination that stays consistent as new documents are uploaded concurrently. Offset pagination drifts when inserts shift rows between pages, producing duplicates or gaps for clients mid-traversal.

## Decision
Use an opaque base64-encoded cursor carrying `(created_at, id)`. Each page query adds `(created_at, id) < (cursor_ts, cursor_id)` with `ORDER BY created_at DESC, id DESC LIMIT n+1`. The composite index on `(workspace_id, created_at, id)` makes this O(1) regardless of table size.

`created_at` alone is not sufficient as a cursor: documents uploaded within the same millisecond share a timestamp, creating ambiguous tie-breaking. Adding the UUIDv7 `id` as a secondary key breaks all ties deterministically, since UUIDv7 embeds a monotonic sub-millisecond counter.

## Consequences
Cursor pagination is stable under concurrent inserts — a new document appearing before the cursor never shifts subsequent pages. Clients cannot jump to arbitrary pages (no random access), but knowledge-platform access patterns are sequential (search → next results). Rejected alternatives: offset pagination (page drift), keyset on `created_at` alone (ties), and `id`-only cursor (loses time ordering, can't use the composite index).
