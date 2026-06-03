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


def test_broker_connection_retry_on_startup() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.broker_connection_retry_on_startup is True


def test_worker_send_task_events() -> None:
    from src.workers.celery_app import celery_app

    assert celery_app.conf.worker_send_task_events is True
