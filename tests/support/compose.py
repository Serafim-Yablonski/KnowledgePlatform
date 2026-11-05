"""docker-compose.yml as data: tests run what compose runs, not a copy of it."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

COMPOSE_FILE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
# What `docker compose stop` waits before SIGKILL when a service sets nothing.
DOCKER_DEFAULT_GRACE_S = 10


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


def resolved(value: object) -> str:
    """A compose value as it runs here: ``${VAR:-x}`` → VAR from the shell, else
    from the project's .env (as compose interpolates), else ``x``."""
    text = str(value)
    match = re.fullmatch(r"\$\{(\w+):-(.+)\}", text)
    if not match:
        return text
    name, fallback = match[1], match[2]
    if os.environ.get(name):
        return os.environ[name]
    dotenv = COMPOSE_FILE.parent / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            key, sep, val = line.partition("=")
            if sep and key.strip() == name and val.strip():
                return val.strip()
    return fallback


def megabytes(size: str) -> int:
    """A compose byte size such as "1536m" or "2g", in MiB."""
    match = re.fullmatch(r"(\d+)([mg])", size.lower())
    assert match, f"unsupported size {size!r}"
    return int(match[1]) * (1024 if match[2] == "g" else 1)


def grace_period_s(name: str) -> float:
    """A service's stop_grace_period, such as "30s" or "1m"; Docker's default when
    unset."""
    raw = service(name).get("stop_grace_period")
    if raw is None:
        return DOCKER_DEFAULT_GRACE_S
    match = re.fullmatch(r"(\d+)(s|m)", str(raw))
    assert match, f"unsupported duration {raw!r}"
    return int(match[1]) * (60 if match[2] == "m" else 1)
