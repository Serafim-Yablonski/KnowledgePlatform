"""Past the extract worker's memory cap the kernel kills the parse; it doesn't swap.

A Linux container (the suite's Docker) gets the cap's pairing from docker-compose.yml:
`memswap_limit` equal to `mem_limit` turns swap off; without it Docker allows as much
again in swap, and a hungry parse would crawl into its timeout instead (S8-10). Inside,
a parent runs a child that allocates past the limit, as extract_text runs the PDF
parse: the OOM killer takes the child, and the parent sees -SIGKILL, which _parse_pdf
turns into ParseKilled. The limit is scaled down; tests/unit/test_compose.py pins the
real size against what we measured.
"""

from __future__ import annotations

from typing import Any

from testcontainers.core.container import DockerContainer

from tests.support.compose import default, megabytes, service
from tests.support.docker import wait_for_port_mappings

LIMIT_MIB = 128

_PARENT = """
import subprocess, sys
hog = (
    "chunks = []\\nfor _ in range({mib}):\\n"
    "    chunks.append(b'x' * 2**20)\\nprint('survived')"
)
done = subprocess.run([sys.executable, "-c", hog], capture_output=True)
print("child returncode", done.returncode, done.stdout.decode().strip())
"""


def _limits(limit_mib: int) -> dict[str, Any]:
    """The compose cap's mem / swap pairing, at `limit_mib`."""
    spec = service("worker-extract")
    limits: dict[str, Any] = {"mem_limit": f"{limit_mib}m"}
    if "memswap_limit" in spec:
        mem = megabytes(default(spec["mem_limit"]))
        swap = megabytes(default(spec["memswap_limit"]))
        limits["memswap_limit"] = f"{limit_mib * swap // mem}m"
    return limits


def test_a_parse_past_the_cap_is_killed_not_swapped() -> None:
    from tests.conftest import _ensure_docker_host

    _ensure_docker_host()
    script = _PARENT.format(mib=LIMIT_MIB * 3 // 2)
    container = DockerContainer("python:3.14-slim", **_limits(LIMIT_MIB))
    container.with_command(["python", "-c", script])
    with wait_for_port_mappings():  # Ryuk, the reaper, starts first
        container.start()
    try:
        wrapped = container.get_wrapped_container()
        status = wrapped.wait(timeout=120)["StatusCode"]
        logs = wrapped.logs().decode()
    finally:
        container.stop()

    assert status == 0, f"the parent itself died:\n{logs}"
    assert "child returncode -9" in logs, logs
