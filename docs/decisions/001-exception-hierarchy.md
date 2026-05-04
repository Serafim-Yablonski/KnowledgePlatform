# 001. Custom Exception Hierarchy over HTTPException
Status: Accepted | Date: 2026-05-04

## Context
The service layer needs to signal domain errors (not found, forbidden, conflict) to callers. The obvious shortcut is raising FastAPI's `HTTPException` directly from services, but this couples business logic to HTTP transport.

## Decision
Introduce `AppError(Exception)` with a `status_code` int and subclasses for each HTTP error category. A single FastAPI exception handler maps `AppError → JSONResponse`. Services raise `NotFoundError`, `ForbiddenError`, etc. — with no knowledge of HTTP.

## Consequences
Services, repositories, and background workers (Celery) can raise the same exception types regardless of transport. The MCP server, REST API, and any future gRPC layer all translate `AppError` locally. The tradeoff: one extra indirection layer versus using `HTTPException` everywhere. Rejected alternative: catching `HTTPException` in non-HTTP contexts (e.g., Celery tasks) would require stripping HTTP semantics before re-raising, which is worse than the extra layer.
