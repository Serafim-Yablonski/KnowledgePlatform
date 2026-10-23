# 017. ORM Objects May Cross the Service Boundary, Never the HTTP Boundary
Status: Accepted | Date: 2026-10-16

## Context
Repositories return SQLAlchemy ORM instances, services pass them on, and routes receive them (e.g. the current `User` and `Workspace` from dependencies). ORM instances are bound to a session: touching an unloaded relationship issues a query, which under async SQLAlchemy fails with `MissingGreenlet`, and the session closes when the request ends.

## Decision
ORM objects may travel through services into routes as data carriers, under three rules. Routes always return a Pydantic response schema (`response_model`, `from_attributes=True`), so internal columns never reach clients by accident. Routes never use session behaviour: no session access, no lazy relationship traversal. Any relationship a response needs is eager-loaded in the repository with `selectinload()`. Computed or aggregated results that are not entities (`WorkspaceStats`, `ResearchStatus`) use plain types from `domain/`.

## Consequences
Services keep SQLAlchemy's unit of work: mutate an entity, then `commit()` (as `DocumentService.create` does). Queries defer large columns their callers never read, with `raiseload` so a later read fails at the access site instead of lazy-loading (the document repository's `get_by_id` and `list_by_workspace` defer `Document.raw_text`, which only the workers read, through their own queries). The cost is that services are coupled to the ORM, and a schema change can ripple up to response mapping. **Rejected: a pure domain model (hexagonal architecture)**, where repositories map every ORM row to a dataclass. For a mostly-CRUD service it defines each entity three times (ORM model, domain object, API schema), adds mapping code between them, and replaces unit of work with explicit `repo.update()` calls, for isolation this project does not need.
