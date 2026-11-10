"""R1 (ADR-033): the service method owns the transaction; repositories never end one.

A repository that commits or rolls back on its own commits or discards its caller's
other writes mid-operation (F1) — in an upload, a row committed before its file is
written. Read from the source: no runtime test would notice, since the service's own
commit follows anyway.
"""

from __future__ import annotations

import ast
from pathlib import Path

import src.repositories

_REPOSITORIES = Path(src.repositories.__file__).parent
_ENDS_A_TRANSACTION = {"commit", "rollback"}


def _transaction_ends(path: Path) -> list[str]:
    return [
        f"{path.name}:{node.lineno} .{node.func.attr}()"
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _ENDS_A_TRANSACTION
    ]


def test_the_scan_finds_the_unit_of_works_own_commit() -> None:
    # Guards the scan itself: a wrong path or pattern would pass the test below.
    assert _transaction_ends(_REPOSITORIES / "unit_of_work.py")


def test_only_the_unit_of_work_ends_a_transaction() -> None:
    found = [
        end
        for path in sorted(_REPOSITORIES.glob("*.py"))
        if path.name != "unit_of_work.py"
        for end in _transaction_ends(path)
    ]
    assert found == []
