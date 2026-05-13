# 004. Multi-Tenancy via Workspace-Level Isolation
Status: Accepted | Date: 2026-05-13

## Context
The platform needs tenant isolation so that users' documents, members, and AI outputs are scoped to a team boundary. Three common PostgreSQL isolation strategies are workspace-level FK scoping, row-level security (RLS), and schema-per-tenant.

## Decision
Enforce isolation at the application layer: every resource table carries a `workspace_id` foreign key, and every query in the repository layer filters by it. Membership is checked in the service layer on every workspace-scoped request. The `get_current_workspace` FastAPI dependency performs the membership check once per request and attaches the caller's role to `request.state` for downstream permission checks without re-querying.

## Consequences
This approach is straightforward to implement, easy to test with Protocol stubs, and visible to static analysis (a missing `workspace_id` filter is caught in code review rather than buried in DB policy). It performs well at the scale of a portfolio project and avoids the operational complexity of schema-per-tenant (DDL migrations across N schemas) or RLS (policy debugging, Alembic limitations with `SET LOCAL` in async drivers). The trade-off is that a bug in the application layer could leak cross-tenant data — RLS provides defence-in-depth that this design omits. Both RLS and schema-per-tenant were rejected: RLS requires careful driver-level session configuration with asyncpg and makes integration tests significantly harder; schema-per-tenant adds non-trivial migration orchestration that is an ops concern, not a backend developer concern.
