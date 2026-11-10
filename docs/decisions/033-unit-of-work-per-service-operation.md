# 033. Each Service Operation Opens Its Own Unit of Work
Status: Accepted | Date: 2026-11-10 | Amends: ADR-017, ADR-018

## Context
Each REST request and each MCP call had one `AsyncSession` shared by every service it built (`Services(session, …)`), and commits were made in two places: some repositories committed (`UserRepository`, `ApiKeyRepository`, most of `WorkspaceRepository`), while `DocumentService` committed the session it was handed. Three faults came from that shared session:
- **Partial commits.** One commit writes everything pending in the session, so a repository's commit could commit another service's unfinished writes. No path does two writes today, but Track 3's research worker will.
- **Parallel tool calls fail (reproduced).** PydanticAI runs a model's parallel tool calls as concurrent tasks. Two `get_document_details` calls in one response used the shared session at once, and `ask` failed with a 500 (`concurrent operations are not permitted`). Anthropic and Gemini models send parallel calls by default.
- **Connections held while waiting on providers (measured).** A session takes a pooled connection at its first query and keeps it until commit, rollback or close. Every request reads the user first, so the connection sat `idle in transaction` through search's embedding call and through `ask`'s whole LLM loop. The pool has 30 connections per process, so a few slow answers could stall every other request for `pool_timeout` (30 s).

## Decision
**The service method owns the transaction; repositories never commit.** `Services` holds a unit-of-work factory instead of a session. Each public service method opens a unit of work: a fresh session with every repository bound to it (`UnitOfWorkProtocol` in `repositories/protocols.py`, built by the composition root):
```python
async with self._uow() as uow:
    ...
    await uow.commit()
```
- **Explicit commit.** Leaving the block without `commit()` rolls back. An exception or a cancellation rolls back and closes the session, which returns the connection. The close is shielded (`anyio.CancelScope(shield=True)`): Starlette's and the MCP SDK's task groups cancel level-triggered, so every await in a cancelled scope raises again, and an unshielded close left the connection checked out (reproduced). `expire_on_commit=False` stays.
- **One public method, one unit of work.** Private helpers take the open unit of work as an argument. A call into another service opens its own. No unit of work spans two services.
- **No nesting.** A unit of work opened inside another, or in a task spawned within one while that one is still open, raises. A `contextvar` marker set by the factory, and marked closed when the block ends, enforces this: nesting would hold one connection while waiting for another, which starves the pool under load. A fire-and-forget task that runs after its parent's block closed holds nothing beside it and may open its own.
- **One `Services` per process.** `lifespan` creates the engine, the Redis client and the HTTP client, builds `Services` from them and yields `{"services": …, "engine": …, "redis": …}` as lifespan state (`/health` probes the engine; the rate limiters use Redis). REST's `get_services` and an MCP helper both read it from the request. A worker builds its own `Services` from its own engine. No engine or client is created at import time.
- **Unchanged orderings:**
  - An upload writes its file before its commit and holds the row lock across the write, so a workspace delete waits.
  - Deletes commit before removing files.
  - `IntegrityError` → 409 is mapped inside the service, because the repository flushes the insert.

Tests (on real Postgres; service, REST and MCP tests run the production wiring with real commits and a `TRUNCATE` per test; repository tests keep the rolled-back-transaction harness, since a repository only ever runs inside its caller's transaction):
- The unit of work's commit, rollback, cancellation (an asyncio task and an anyio scope) and nesting behaviour.
- A held-connection counter (pool checkout/checkin events) that must read 0 inside the embedding client and the model during `search` and `ask`, and 1 during the upload's file write.
- Two parallel tool calls through the real `AIService`.
- The real lifespan builds one `Services` that a REST request and an MCP call both use; importing the app, in a fresh interpreter, leaves no engine, pool or client behind.
- The orderings above, with injected failures.

## Consequences
- **Connections are held for milliseconds,** for the database work only, not for the length of a request. The pool stops capping concurrent `ask`s. A parallel tool call takes one more connection.
- **More checkouts per request.** A request opens several short transactions: auth, membership, the operation. Measured locally, that adds ~4 ms per request, about half of it `pool_pre_ping`. The ping is kept, because without it the first request on each stale connection after a Postgres restart is a 500.
- **Detached results.** ORM objects returned by a service are detached once its block ends. ADR-017's rules already keep routes from needing a session.

Rejected alternatives:
- **Keeping the request's session, with each method in `session.begin()`, `autobegin=False` and `sequential=True` on every tool that touches the database.** A smaller diff, but parallel-call safety rests on a per-tool flag no lint can check (a missed flag is an intermittent, model-dependent 500), nested service calls hit "a transaction is already begun", and the worker would still need a session per step.
- **Committing once at request end in the dependency.** It removes files and maps conflicts after the fact, and fixes neither fault.
- **An ambient unit of work in a `contextvar`, joined by nested calls (Spring's REQUIRED).** `create_task` copies the context, so parallel tool calls would share one session again, and whether a method commits would be invisible at the call site.
- **Commit on clean exit (`sessions.begin()`).** An exception caught inside the block would commit partial writes.
- **A module-level `Services` getter.** Two app instances in one process would overwrite each other, and tests would monkeypatch modules.
