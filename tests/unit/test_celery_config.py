"""Verify that the Celery app is configured with the correct reliability settings."""


def test_task_acks_late() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.task_acks_late is True


def test_task_reject_on_worker_lost() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.task_reject_on_worker_lost is True


def test_worker_prefetch_multiplier() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_task_always_eager_is_false() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.task_always_eager is False


def test_worker_max_tasks_per_child() -> None:
    from src.core.config import settings
    from src.workers.celery_app import celery_app

    expected = settings.CELERY_WORKER_MAX_TASKS_PER_CHILD
    assert celery_app.conf.worker_max_tasks_per_child == expected


def test_task_routes() -> None:
    from src.workers.celery_app import QUEUE_EMBED, QUEUE_EXTRACT, celery_app

    routes = celery_app.conf.task_routes
    assert routes["src.workers.tasks.extract_text.extract_text"] == {
        "queue": QUEUE_EXTRACT
    }
    assert routes["src.workers.tasks.embed_chunks.embed_chunks"] == {
        "queue": QUEUE_EMBED
    }


def test_extract_task_time_limits() -> None:
    from src.core.config import settings
    from src.workers.tasks.extract_text import extract_text

    assert extract_text.soft_time_limit == settings.CELERY_EXTRACT_SOFT_TIME_LIMIT
    assert extract_text.time_limit == settings.CELERY_EXTRACT_TIME_LIMIT


def test_embed_task_time_limits() -> None:
    from src.core.config import settings
    from src.workers.tasks.embed_chunks import embed_chunks

    assert embed_chunks.soft_time_limit == settings.CELERY_EMBED_SOFT_TIME_LIMIT
    assert embed_chunks.time_limit == settings.CELERY_EMBED_TIME_LIMIT


def test_json_serializer_only() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.task_serializer == "json"
    assert celery_app.conf.result_serializer == "json"
    assert celery_app.conf.accept_content == ["json"]


def test_task_ignore_result() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.task_ignore_result is True


def test_no_result_backend() -> None:
    """No caller reads a task's result (task_ignore_result), so nothing is stored."""
    from src.workers.celery_app import celery_app

    assert celery_app.conf.result_backend is None


def test_broker_connection_retry_on_startup() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.broker_connection_retry_on_startup is True


def test_worker_send_task_events() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.worker_send_task_events is True


def test_pipeline_tasks_retry_through_celerys_autoretry() -> None:
    # Outside-world phases are retried with capped exponential backoff and full
    # jitter; a PermanentFailure (a pure phase failed) is not retried at all.
    from src.workers.pipeline import PermanentFailure
    from src.workers.tasks.embed_chunks import embed_chunks
    from src.workers.tasks.extract_text import extract_text

    for task, backoff in ((extract_text, 30), (embed_chunks, 60)):
        assert task.autoretry_for == (Exception,)
        assert task.dont_autoretry_for == (PermanentFailure,)
        assert task.throws == (PermanentFailure,)
        assert (task.max_retries, task.retry_backoff, task.retry_jitter) == (
            3,
            backoff,
            True,
        )


def test_a_permanent_failure_keeps_its_reason_through_celerys_pickling() -> None:
    # Celery hands on_failure the exception after a pickle round trip (S6-16); one
    # that can't be rebuilt arrives wrapped, and its reason is lost.
    from celery.utils.serialization import get_pickleable_exception

    from src.domain.documents import FailureReason
    from src.workers.pipeline import PermanentFailure

    received = get_pickleable_exception(PermanentFailure(FailureReason.NO_TEXT))

    assert isinstance(received, PermanentFailure)
    assert received.reason is FailureReason.NO_TEXT


def test_a_killed_parse_keeps_its_reason_through_celerys_pickling() -> None:
    from celery.utils.serialization import get_pickleable_exception

    from src.domain.documents import FailureReason
    from src.workers.pipeline import ParseKilled, failure_reason

    received = get_pickleable_exception(ParseKilled("killed by SIGKILL"))

    assert isinstance(received, ParseKilled)
    assert failure_reason(received) is FailureReason.EXTRACTION_TOO_LARGE


def test_worker_soft_shutdown_timeout() -> None:
    from src.core.config import settings
    from src.workers.celery_app import celery_app

    expected = settings.CELERY_WORKER_SOFT_SHUTDOWN_TIMEOUT
    assert celery_app.conf.worker_soft_shutdown_timeout == expected
