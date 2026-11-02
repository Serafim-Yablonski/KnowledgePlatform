"""docker-compose.yml as data: tests run what compose runs, not a copy of it."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def compose() -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(COMPOSE_FILE.read_text())
    return loaded


def service(name: str) -> dict[str, Any]:
    services: dict[str, dict[str, Any]] = compose()["services"]
    return services[name]


def default(value: object) -> str:
    """A compose value as it runs without an override: ``${VAR:-x}`` → ``x``."""
    text = str(value)
    match = re.fullmatch(r"\$\{\w+:-(.+)\}", text)
    return match[1] if match else text


def megabytes(size: str) -> int:
    """A compose byte size such as "1536m" or "2g", in MiB."""
    match = re.fullmatch(r"(\d+)([mg])", size.lower())
    assert match, f"unsupported size {size!r}"
    return int(match[1]) * (1024 if match[2] == "g" else 1)
