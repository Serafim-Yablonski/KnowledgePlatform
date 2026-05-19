# 009. Sliding Window Rate Limiter with Redis Sorted Sets
Status: Accepted | Date: 2026-05-14

## Context
AI search endpoints are expensive (embedding API call + vector scan). Without rate limiting, a single user can exhaust the embedding quota or degrade latency for everyone else. We need per-user rate limiting that is precise, race-condition-free, and easy to explain in API documentation.

## Decision
Sliding window counter using Redis sorted sets. Each request is recorded as a member in a sorted set keyed by `nexus:ratelimit:{prefix}:{user_id}`, with the request timestamp as score. On every request a single pipeline executes: ZADD (record request), ZREMRANGEBYSCORE (evict entries older than the window), ZCARD (count), ZRANGE (oldest entry for retry-after), EXPIRE (TTL cleanup). If the count exceeds the limit, `RateLimitError` is raised before the handler runs. The `Retry-After` header value is derived from the oldest entry's timestamp so users know the exact wait rather than a worst-case estimate.

## Consequences
Sliding window is more precise than a fixed window (no burst allowed at window boundaries) and more intuitive than token bucket (the limit "N requests per minute" maps directly to what users read in the docs). Redis sorted sets avoid race conditions that simple string counters have under concurrent writes — no MULTI/EXEC needed because each request uses a nanosecond-precision unique member key. Rejected alternatives: token bucket (harder to explain, require float arithmetic in Lua), fixed window counter (burst possible at boundary), in-process rate limiting (doesn't work across multiple API replicas). The one trade-off: rejected requests are counted in the window, meaning retrying while rate-limited delays recovery — this is intentional to prevent retry storms.
