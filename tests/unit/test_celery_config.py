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
