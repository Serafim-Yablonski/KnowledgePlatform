"""What docker-compose.yml promises about the runtime (pipeline step 8)."""

import re

from tests.support.compose import compose, default, megabytes, service


def test_redis_keeps_its_data_in_a_named_volume() -> None:
    """Redis 8's image declares no volume, and an anonymous one is replaced on every
    `docker compose down` + `up`: only a named volume keeps the queue (S8-1, S8-2)."""
    mounts = service("redis").get("volumes", [])
    named = [m for m in mounts if m.endswith(":/data") and not m.startswith(("/", "."))]
    assert len(named) == 1, f"no named volume at /data: {mounts}"
    volume = named[0].split(":")[0]
    assert volume in compose()["volumes"]


def test_every_service_restarts_after_a_crash() -> None:
    """Without a policy a crashed or OOM-killed container stays down until someone
    starts it (S8-8). `unless-stopped`: a `docker compose stop` still stops it."""
    for name, spec in compose()["services"].items():
        assert spec.get("restart") == "unless-stopped", name


def _workers() -> list[str]:
    return [n for n in compose()["services"] if n.startswith("celery-worker")]


def test_workers_turn_sigterm_into_a_soft_shutdown() -> None:
    """SIGTERM (`docker compose stop`) is otherwise a warm shutdown, which waits for
    running tasks past the grace period (tests/integration/test_worker_shutdown.py)."""
    assert _workers()
    for name in _workers():
        assert service(name)["environment"].get("REMAP_SIGTERM") == "SIGQUIT", name


def test_the_soft_shutdown_ends_inside_the_grace_period() -> None:
    """After the soft timeout the worker cancels, restores its messages and exits;
    that must happen before compose's SIGKILL. The margin covers the cold shutdown
    (about a second in the step-8 probe)."""
    from src.core.config import settings
    from tests.integration.test_worker_shutdown import grace_period_s

    margin_s = 5
    for name in _workers():
        grace = grace_period_s(name)
        assert settings.CELERY_WORKER_SOFT_SHUTDOWN_TIMEOUT + margin_s <= grace, name


# Measured in the built image at the 25 MB upload cap (pipeline step 8c-0): an idle
# extract worker (main process + 2 pool children), the largest parse child seen
# (a 70 MB decoded stream, stopped by pypdf's limit) and what a pool child holds
# beyond its base during one extraction (the bytes, the text, its encoded copy).
IDLE_EXTRACT_WORKER_MIB = 332
PARSE_CHILD_PEAK_MIB = 455
POOL_CHILD_JOB_MIB = 88
MEASURED_AT_UPLOAD_CAP_MB = 25


def test_the_extract_worker_cannot_swap() -> None:
    """With mem_limit alone Docker allows as much again in swap: a hungry parse would
    swap into its timeout instead of being killed (S8-10)."""
    spec = service("celery-worker-extract")
    assert "mem_limit" in spec
    assert default(spec.get("memswap_limit")) == default(spec["mem_limit"])


def test_the_extract_memory_cap_covers_the_measured_worst_case() -> None:
    """The cap must not kill what we measured: every slot running the worst parse
    seen. Raising the upload cap or the concurrency means measuring again."""
    from src.core.config import Settings, settings

    # The shipped default and what this environment runs (.env may override it).
    default_cap = Settings.model_fields["MAX_UPLOAD_SIZE_MB"].default
    assert default_cap <= MEASURED_AT_UPLOAD_CAP_MB
    assert settings.MAX_UPLOAD_SIZE_MB <= MEASURED_AT_UPLOAD_CAP_MB
    spec = service("celery-worker-extract")
    concurrency = int(default(re.search(r"--concurrency (\S+)", spec["command"])[1]))
    worst = IDLE_EXTRACT_WORKER_MIB + concurrency * (
        PARSE_CHILD_PEAK_MIB + POOL_CHILD_JOB_MIB
    )
    assert megabytes(default(spec["mem_limit"])) >= worst
