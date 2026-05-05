# 003. Repository Interfaces as typing.Protocol
Status: Accepted | Date: 2026-05-04

## Context
Services need a stable contract to program against without coupling to SQLAlchemy. The interface mechanism must satisfy `mypy --strict` and enable real unit tests that don't touch the database.

## Decision
Every repository is declared as a `typing.Protocol` in `src/repositories/protocols.py`. Services depend on the protocol type, never the SQLAlchemy class. Unit tests supply in-memory stub classes that implement the protocol structurally — no mocking framework required.

This works with `mypy --strict` because Protocol uses structural subtyping: any class with the matching method signatures satisfies the type checker, regardless of inheritance. The stub classes in tests are verified by mypy to be correct implementations at type-check time, not just at runtime.

## Consequences
Chosen over ABC: ABC requires explicit inheritance, which couples the SQLAlchemy implementation to the abstract class and forces the test stub to inherit it too. With Protocol, third-party or alternative implementations (e.g., a Redis-backed cache repo) automatically satisfy the contract if their signatures match — zero coupling. Rejected runtime `isinstance` checks against a base class for the same reason. Rejected no interface at all because services would then depend directly on `SQLAlchemyUserRepository`, making unit tests impossible without a live database.
