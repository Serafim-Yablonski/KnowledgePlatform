# REST API Design Principles

## Resource Naming

REST APIs model resources as nouns, not verbs. URL paths identify resources; HTTP methods express the action.

**Do:**
- `GET /users` — list users
- `GET /users/{id}` — get one user
- `POST /users` — create a user
- `PATCH /users/{id}` — partial update
- `DELETE /users/{id}` — delete

**Don't:**
- `POST /createUser`
- `GET /getUserById?id=123`
- `POST /users/delete`

Use **plural nouns** for collections. Nest sub-resources when the relationship is intrinsic: `GET /workspaces/{id}/members`. Keep nesting to a maximum of 2 levels; deeper hierarchies become unmaintainable.

## HTTP Status Codes

| Code | Meaning | Use When |
|------|---------|----------|
| 200 | OK | Successful GET, PATCH, or DELETE |
| 201 | Created | Successful POST; include `Location` header |
| 204 | No Content | Successful DELETE with no body |
| 400 | Bad Request | Malformed request or validation error |
| 401 | Unauthorized | Missing or invalid credentials |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Duplicate key, concurrent update |
| 422 | Unprocessable Entity | Validation failed (preferred in APIs) |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Server Error | Unexpected server failure |

Always return a consistent error schema:

```json
{
  "error": "validation_error",
  "message": "title must not be empty",
  "field": "title"
}
```

## Versioning

Version APIs from day one. Common strategies:

1. **URL path**: `/v1/users` — most visible, easy to route, forces explicit migration
2. **Header**: `Accept: application/vnd.myapi.v1+json` — cleaner URLs, harder to test in browser
3. **Query parameter**: `?version=1` — least recommended; caches don't differentiate

URL path versioning is the most practical default. Increment the major version only on breaking changes (removing fields, changing types). Adding optional fields is backward-compatible and does not require a version bump.

## Pagination

**Offset pagination** (`?page=2&size=20`) is simple but unreliable for large datasets: rows inserted or deleted between pages cause skipped or duplicated results.

**Cursor-based pagination** uses an opaque cursor from the last-seen item:

```json
{
  "items": [...],
  "next_cursor": "eyJjcmVhdGVkX2F0IjoiMjAyNC0wMS0xNVQxMjowMDowMFoiLCJpZCI6IjAxOTI...",
  "has_more": true
}
```

The cursor encodes `(created_at, id)` and is base64-encoded. This approach handles concurrent writes without skips. It does not support random page access.

## Idempotency

`GET`, `PUT`, and `DELETE` should be idempotent — calling them multiple times yields the same result as once. `POST` is not idempotent by definition.

For `POST` requests that must be retried safely (e.g., payment creation), support an **idempotency key** header:

```
Idempotency-Key: 550e8400-e29b-41d4-a716-446655440000
```

Store the response keyed on `(user_id, idempotency_key)` for 24 hours; return the stored response on duplicates.

## Rate Limiting

Return `429 Too Many Requests` when limits are exceeded. Include headers:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 1704067200
Retry-After: 60
```

Common algorithms:
- **Token bucket**: allows bursts up to bucket capacity; smooth refill rate
- **Fixed window**: simple; susceptible to burst at window boundary
- **Sliding window**: accurate; higher memory cost per user

## Filtering and Sorting

Use query parameters for filtering: `GET /documents?status=ready&content_type=pdf`

For sorting: `GET /documents?sort=created_at&order=desc`

Do not use custom DSLs for simple filters. Reserve a `filter` parameter with structured query syntax only if the use case genuinely requires it (e.g., search APIs).

## HATEOAS

Hypermedia As The Engine Of Application State (HATEOAS) adds `_links` to responses so clients discover available transitions:

```json
{
  "id": "doc-123",
  "status": "pending",
  "_links": {
    "self": "/documents/doc-123",
    "cancel": "/documents/doc-123/cancel"
  }
}
```

HATEOAS is rarely implemented in practice outside public APIs with long-lived clients. For internal APIs consumed by a single frontend, it adds overhead without benefit.
