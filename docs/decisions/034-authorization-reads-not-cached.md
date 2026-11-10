# 034. Authorization Lookups Read PostgreSQL, Not a Cache
Status: Accepted | Date: 2026-11-10 | Amends: ADR-019

## Context
Membership, workspace and API-key lookups (with the key's user) were served from Redis with a 300 s TTL (`CachedWorkspaceRepository`, `CachedApiKeyRepository`) and invalidated after the inner repository's commit. No ADR recorded why. JWT authentication read the user from PostgreSQL on every request, while the API-key path served the key and `is_active` from Redis, so the two paths disagreed. Staleness here is a security fault: a reader straddling a removal's commit could re-cache a removed membership or a revoked key for the full TTL. `invalidate_user_keys` had no caller. Under ADR-033 the caches would also need post-commit hooks in the unit of work.

Measured locally with testcontainers (pgvector:pg18, redis:8, Docker Desktop port forwarding, so the absolute values are inflated):
- **A cache hit** (Redis GET + JSON): median 0.35 ms.
- **A membership read by primary key** in its own session: 2.4 ms with `pool_pre_ping`, 1.4 ms without.
- **A cache miss** (GET, read, SET): 3.3 ms.

So a hit saves ~1–2 ms per lookup, at most three lookups per request. The requests these lookups guard cost hundreds of milliseconds (embedding) to seconds (LLM).

## Decision
**Membership, workspace and API-key lookups always read PostgreSQL,** like the JWT path. The cached repositories, their serializers, the `CACHE_TTL_*` settings, `ApiKeyService.invalidate_user_keys` / `invalidate_all_for_user`, and `ResponseCache.get/set/get_or_set/delete/delete_pattern` are deleted. `ResponseCache` keeps only `claim`.

**ADR-019's throttle stays,** but moves to `ApiKeyService`. It claims the Redis marker before opening its own unit of work for the `last_used_at` write. That write's rollback can no longer expire objects that authentication returned, so ADR-019's detached-snapshot paragraph no longer applies.

Tests:
- Removing a member or revoking a key takes effect on the next request.
- The throttle writes at most once per window and fails closed without Redis.

## Consequences
- **Each authenticated request makes up to three more short reads,** about 1–2 ms each, on a database that is on every request's path anyway.
- **Revocation and removal take effect immediately,** and both authentication paths behave the same.
- **About 340 lines and their tests go.**
- **When to reintroduce a cache:** if authorization reads come to dominate database load, or the source of authorization becomes remote (an identity provider). Such a cache needs a stated staleness bound and invalidation that runs after the commit.

Rejected alternatives:
- **Keeping the caches with post-commit callbacks in the unit of work, or with explicit cache clears in services.** That is machinery whose only job is to preserve a cache that saves ~1–2 ms.
- **SQLAlchemy's `after_commit` event.** Its handlers are synchronous and can't await Redis.
- **A consistent cache with a 60 s bound.** Defensible, but all four questions argue against it: the read is cheap, Postgres is not the bottleneck, authorization tolerates no staleness, and the gain doesn't cover the cost.
