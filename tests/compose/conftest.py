"""The compose stack as it ships (P7d): the image and the wiring — what the host
suite (tests/acceptance) can't see.

One isolated stack per module: a unique project name, an environment built from
nothing plus a generated env file (no developer key, token or override reaches it),
random loopback ports, an always-200 fake provider in the stack's own network
(tests/support/fake_gemini.py, mounted read-only), and the image's own code (the
dev bind mounts are dropped). Torn down with its volumes and the images it built
(each run's project name tags new ones; pulled images stay).
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from src.core.config import settings
from tests.support.acceptance import FAKE_API_KEY, PIPELINE_ENV, ROOT

FAKE = ROOT / "tests" / "support" / "fake_gemini.py"


@dataclass(frozen=True)
class Stack:
    command: list[str]
    env: dict[str, str]
    api_url: str

    def compose(self, *args: str, timeout: float = 600) -> str:
        done = subprocess.run(
            [*self.command, *args],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        assert done.returncode == 0, f"compose {args}: {done.stderr[-2000:]}"
        return done.stdout

    def container(self, service: str) -> str:
        return self.compose("ps", "-q", service).strip()

    def logs(self) -> str:
        return self.compose("logs", "--no-color", "--tail", "100")


def _env_file(path: Path) -> None:
    values = {
        "POSTGRES_DB": "knowledge_platform",
        "POSTGRES_USER": "user",
        "POSTGRES_PASSWORD": "acceptance-password",
        "SECRET_KEY": "acceptance-secret-key-not-for-production",
        "ENVIRONMENT": "test",
        "LOGFIRE_TOKEN": "",
        # The LLM agents are built at startup; no request reaches a provider here.
        "GOOGLE_API_KEY": "not-a-key",
        "ANTHROPIC_API_KEY": "not-a-key",
        "EMBEDDING_API_KEY": FAKE_API_KEY,
        "EMBEDDING_BASE_URL": "http://fake-gemini:8080/v1beta",
        **PIPELINE_ENV,
    }
    path.write_text("".join(f"{k}={v}\n" for k, v in values.items()))


def _override(path: Path, env_file: Path) -> None:
    path.write_text(
        f"""
services:
  postgres:
    ports: !reset []
  app:
    env_file: !override ["{env_file}"]
    ports: !override ["127.0.0.1::8000"]
    volumes: !override ["uploads_data:/data/uploads"]
  worker-embed:
    env_file: !override ["{env_file}"]
  fake-gemini:
    image: python:3.14-slim
    command:
      - python
      - /fake_gemini.py
      - --host=0.0.0.0
      - --port=8080
      - --model={settings.EMBEDDING_MODEL}
      - --dimensions={settings.EMBEDDING_DIMENSIONS}
    volumes: ["{FAKE}:/fake_gemini.py:ro"]
"""
    )


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Stack]:
    from tests.conftest import _ensure_docker_host

    _ensure_docker_host()
    tmp = tmp_path_factory.mktemp("compose")
    env_file, override = tmp / "acceptance.env", tmp / "override.yml"
    _env_file(env_file)
    _override(override, env_file)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "DOCKER_HOST") if k in os.environ}
    command = [
        "docker", "compose",
        "-p", f"kp-accept-{uuid.uuid4().hex[:8]}",
        "--project-directory", str(ROOT),
        "-f", str(ROOT / "docker-compose.yml"),
        "-f", str(override),
        "--env-file", str(env_file),
    ]  # fmt: skip
    stack = Stack(command, env, api_url="")
    try:
        stack.compose("up", "--detach", "--build", "--wait", "postgres", "redis")
        stack.compose("up", "--detach", "--build")
        stack.compose("exec", "-T", "app", "alembic", "upgrade", "head")
        host_port = stack.compose("port", "app", "8000").strip()
        stack = Stack(command, env, api_url=f"http://{host_port}")
        _wait_healthy(stack)
        yield stack
    except BaseException:
        print(stack.logs())  # noqa: T201 — the failing run's evidence
        raise
    finally:
        stack.compose(
            "down", "--volumes", "--rmi", "local", "--remove-orphans", timeout=120
        )


def _wait_healthy(stack: Stack, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            if httpx.get(f"{stack.api_url}/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline, f"the app never got healthy\n{stack.logs()}"
        time.sleep(1)
