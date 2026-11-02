"""What docker-compose.yml promises about the runtime (pipeline step 8, pivot P6)."""

import re

from src.core.config import PipelineSettings, Settings
from src.workers.__main__ import PHASES
from tests.support.compose import (
    compose,
    default,
    grace_period_s,
    megabytes,
    resolved,
    service,
)

WORKER_COMMAND = re.compile(r"python -m src\.workers (\w+)")


def _workers() -> dict[str, str]:
    """Phase → the compose service that runs its worker."""
    found = {}
    for name, spec in compose()["services"].items():
        if match := WORKER_COMMAND.fullmatch(str(spec.get("command", "")).strip()):
            found[match[1]] = name
    return found


def _worker_services() -> list[str]:
    """Every phase's service; never an empty list a loop would pass on."""
    workers = _workers()
    assert set(workers) == set(PHASES), workers
    return list(workers.values())


def test_redis_is_a_cache_and_persists_nothing() -> None:
    """The work owed lives in Postgres, not here (ADR-027). The image's defaults
    would snapshot to an anonymous volume."""
    spec = service("redis")
    assert '--save ""' in spec["command"]
    assert "--appendonly no" in spec["command"]
    assert "volumes" not in spec


def test_every_service_restarts_after_a_crash() -> None:
    """Without a policy a crashed or OOM-killed container stays down until someone
    starts it (S8-8). `unless-stopped`: a `docker compose stop` still stops it."""
    for name, spec in compose()["services"].items():
        assert spec.get("restart") == "unless-stopped", name


def test_each_phase_runs_as_its_own_service() -> None:
    """Bulkheads: a stuck or memory-hungry extraction can't starve embedding."""
    workers = _workers()
    assert set(workers) == set(PHASES)
    assert len(set(workers.values())) == len(workers)


def test_the_shutdown_grace_ends_inside_the_stop_grace_period() -> None:
    """On SIGTERM running attempts get PIPELINE_SHUTDOWN_GRACE, then the worker
    cancels them and releases their rows; that must happen before compose's SIGKILL.
    The margin covers the cancel and the release writes."""
    from src.core.config import settings

    margin_s = 5
    # The shipped default and what this environment runs: both workers read it
    # from .env (extract by name, embed through env_file).
    default_grace = Settings.model_fields["PIPELINE_SHUTDOWN_GRACE"].default
    for name in _worker_services():
        for grace in (default_grace, settings.PIPELINE_SHUTDOWN_GRACE):
            assert grace + margin_s <= grace_period_s(name), name


def test_the_extract_worker_gets_no_provider_key() -> None:
    """It parses untrusted files and needs no provider (S8-19). An allow-list: no
    env_file, so a key added to .env later doesn't reach it either."""
    spec = service(_workers()["extract"])
    assert "env_file" not in spec
    names = {str(entry).split("=")[0] for entry in spec.get("environment", [])}
    keys = {
        name
        for name, field in Settings.model_fields.items()
        if name.endswith("_API_KEY") and field.annotation is str
    }
    assert keys and not names & keys


def test_the_extract_worker_gets_what_it_cannot_start_without() -> None:
    """Its allow-list must still carry the database URL (the default is localhost)
    and SECRET_KEY (Settings rejects the dev default)."""
    names = {
        str(entry).split("=")[0]
        for entry in service(_workers()["extract"]).get("environment", [])
    }
    assert {"DATABASE_URL", "SECRET_KEY"} <= names


def test_the_extract_worker_gets_every_pipeline_setting_it_reads() -> None:
    """A name missing from the allow-list keeps its default in that worker alone,
    silently: .env and the shell can't reach it."""
    names = {
        str(entry).split("=")[0]
        for entry in service(_workers()["extract"]).get("environment", [])
    }
    reads = {name for name in PipelineSettings.model_fields if "EMBED" not in name}
    assert reads and reads <= names, reads - names


def test_the_extract_worker_reads_the_uploads_the_api_writes() -> None:
    """Rows hold the file's path under UPLOAD_DIR; without the same volume every
    extraction fails to read it, and ends internal_error after its retries."""

    def uploads(name: str) -> set[str]:
        mounts = service(name).get("volumes", [])
        return {m for m in mounts if m.endswith(":/data/uploads")}

    assert uploads("app")
    assert uploads(_workers()["extract"]) == uploads("app")


def test_the_workers_need_no_redis() -> None:
    """Redis holds no work: a worker starts and runs with the cache down."""
    for name in _worker_services():
        assert "redis" not in service(name).get("depends_on", {}), name


# Measured in the built image at the 25 MB upload cap. The idle extract worker: one
# process since P6d (RSS 183 MiB; `docker stats` showed 158), down from 332 MiB for
# Celery's main process + 2 pool children. Unchanged since step 8c-0: the largest
# parse child seen (a 70 MB decoded stream, stopped by pypdf's limit; the same
# `python -I pdf_text.py` child) and what the worker holds beyond its base during
# one extraction (the bytes, the text, its encoded copy).
IDLE_EXTRACT_WORKER_MIB = 183
PARSE_CHILD_PEAK_MIB = 455
EXTRACTION_JOB_MIB = 88
MEASURED_AT_UPLOAD_CAP_MB = 25


def test_the_extract_worker_cannot_swap() -> None:
    """With mem_limit alone Docker allows as much again in swap: a hungry parse would
    swap into its timeout instead of being killed (S8-10)."""
    spec = service(_workers()["extract"])
    assert "mem_limit" in spec
    assert default(spec.get("memswap_limit")) == default(spec["mem_limit"])


def test_the_extract_memory_cap_covers_the_measured_worst_case() -> None:
    """The cap must not kill what we measured: every slot running the worst parse
    seen. Raising the upload cap or the concurrency means measuring again."""
    from src.core.config import settings

    # The shipped default and what this environment runs (.env may override it).
    default_cap = Settings.model_fields["MAX_UPLOAD_SIZE_MB"].default
    assert default_cap <= MEASURED_AT_UPLOAD_CAP_MB
    assert settings.MAX_UPLOAD_SIZE_MB <= MEASURED_AT_UPLOAD_CAP_MB
    spec = service(_workers()["extract"])
    # Shipped defaults, and what this environment runs: the concurrency from
    # Settings, the cap as compose interpolates it.
    default_slots = Settings.model_fields["PIPELINE_EXTRACT_CONCURRENCY"].default
    for slots, cap in (
        (default_slots, default(spec["mem_limit"])),
        (settings.PIPELINE_EXTRACT_CONCURRENCY, resolved(spec["mem_limit"])),
    ):
        worst = IDLE_EXTRACT_WORKER_MIB + slots * (
            PARSE_CHILD_PEAK_MIB + EXTRACTION_JOB_MIB
        )
        assert megabytes(cap) >= worst, (slots, cap)
