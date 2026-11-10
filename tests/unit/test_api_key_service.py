"""Unit tests for ApiKeyService with Protocol stubs."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog.testing
from fakeredis.aioredis import FakeRedis
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import OperationalError

from src.core.cache import ResponseCache
from src.core.exceptions import ConflictError, InvalidTokenError, NotFoundError
from src.repositories.unit_of_work import NestedUnitOfWorkError
from src.services.api_key import ApiKeyService
from tests.support.unit_of_work import FakeUnitOfWork


def _service(repo: MagicMock, cache: ResponseCache | None = None) -> ApiKeyService:
    return ApiKeyService(
        FakeUnitOfWork(api_keys=repo), cache or ResponseCache(FakeRedis())
    )


def _db_error(message: str) -> OperationalError:
    return OperationalError("UPDATE", {}, Exception(message))


def _make_api_key(
    key_id: uuid.UUID | None = None,
    user_id: uuid.UUID | None = None,
    key_hash: str = "abc",
    prefix: str = "abcdefgh",
    name: str = "test",
    is_active: bool = True,
    user: Any = None,
) -> MagicMock:
    key = MagicMock()
    key.id = key_id or uuid.uuid4()
    key.user_id = user_id or uuid.uuid4()
    key.key_hash = key_hash
    key.prefix = prefix
    key.name = name
    key.is_active = is_active
    key.user = user or MagicMock()
    return key


def _make_repo(
    *,
    active_count: int = 0,
    existing_key: MagicMock | None = None,
    user_keys: list[MagicMock] | None = None,
) -> MagicMock:
    repo = MagicMock()
    repo.count_active_for_user = AsyncMock(return_value=active_count)
    repo.create = AsyncMock(side_effect=lambda **kw: _make_api_key(**kw))
    repo.get_by_hash = AsyncMock(return_value=existing_key)
    repo.list_for_user = AsyncMock(return_value=user_keys or [])
    repo.deactivate = AsyncMock(return_value=None)
    repo.touch_last_used = AsyncMock(return_value=None)
    return repo


class TestCreateKey:
    @pytest.mark.asyncio
    async def test_stores_hash_not_raw_key(self) -> None:
        created: list[MagicMock] = []

        async def _create(**kw: Any) -> MagicMock:
            key = _make_api_key(**kw)
            created.append(key)
            return key

        repo = _make_repo(active_count=0)
        repo.create = AsyncMock(side_effect=_create)
        service = _service(repo)

        api_key, raw_key = await service.create(user_id=uuid.uuid4(), name="my key")

        assert len(created) == 1
        expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        assert created[0].key_hash == expected_hash
        assert raw_key not in (created[0].prefix, created[0].name)

    @pytest.mark.asyncio
    async def test_prefix_is_first_8_chars_of_raw_key(self) -> None:
        repo = _make_repo(active_count=0)

        async def _create(**kw: Any) -> MagicMock:
            return _make_api_key(**kw)

        repo.create = AsyncMock(side_effect=_create)
        service = _service(repo)
        _, raw_key = await service.create(user_id=uuid.uuid4(), name="k")
        call_kwargs = repo.create.call_args.kwargs
        assert call_kwargs["prefix"] == raw_key[:8]

    @pytest.mark.asyncio
    async def test_returns_raw_key_only_once(self) -> None:
        repo = _make_repo(active_count=0)
        service = _service(repo)
        _api_key, raw_key = await service.create(user_id=uuid.uuid4(), name="k")
        assert len(raw_key) > 8

    @pytest.mark.asyncio
    async def test_sixth_key_raises_conflict(self) -> None:
        repo = _make_repo(active_count=5)
        service = _service(repo)
        with pytest.raises(ConflictError, match="Maximum"):
            await service.create(user_id=uuid.uuid4(), name="overflow")


class TestAuthenticate:
    @pytest.mark.asyncio
    async def test_correct_key_returns_user(self) -> None:
        raw_key = "supersecretkey1234"
        expected_user = MagicMock()
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        mock_key = _make_api_key(key_hash=key_hash, is_active=True, user=expected_user)
        repo = _make_repo(existing_key=mock_key)
        service = _service(repo)

        user = await service.authenticate(raw_key)

        assert user is expected_user
        repo.get_by_hash.assert_awaited_once_with(key_hash)
        repo.touch_last_used.assert_awaited_once_with(mock_key.id)

    @pytest.mark.asyncio
    async def test_wrong_key_raises(self) -> None:
        repo = _make_repo(existing_key=None)
        service = _service(repo)
        with pytest.raises(InvalidTokenError):
            await service.authenticate("wrongkey")

    @pytest.mark.asyncio
    async def test_deactivated_key_raises(self) -> None:
        raw_key = "validbutdeactivated"
        mock_key = _make_api_key(is_active=False)
        repo = _make_repo(existing_key=mock_key)
        service = _service(repo)
        with pytest.raises(InvalidTokenError, match="revoked"):
            await service.authenticate(raw_key)

    @pytest.mark.asyncio
    async def test_inactive_user_raises(self) -> None:
        raw_key = "keyforinactiveuser"
        inactive_user = MagicMock()
        inactive_user.is_active = False
        mock_key = _make_api_key(is_active=True, user=inactive_user)
        repo = _make_repo(existing_key=mock_key)
        service = _service(repo)
        with pytest.raises(InvalidTokenError, match="revoked"):
            await service.authenticate(raw_key)
        repo.touch_last_used.assert_not_awaited()


class TestRecordUse:
    @pytest.mark.asyncio
    async def test_bookkeeping_failure_does_not_fail_auth(self) -> None:
        raw_key = "keywhoseusagewritefails"
        expected_user = MagicMock()
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        mock_key = _make_api_key(key_hash=key_hash, user=expected_user)
        repo = _make_repo(existing_key=mock_key)
        repo.touch_last_used = AsyncMock(side_effect=_db_error("DB hiccup"))
        service = _service(repo)

        with structlog.testing.capture_logs() as logs:
            assert await service.authenticate(raw_key) is expected_user

        # Swallowed, but not silently: the cause must reach the log.
        [entry] = [e for e in logs if e["event"] == "failed to record api key use"]
        assert "DB hiccup" in entry["error"]

    @pytest.mark.asyncio
    async def test_a_refused_connection_does_not_fail_auth(self) -> None:
        # asyncpg's refused connection reaches us as an OSError, not wrapped.
        key = _make_api_key(key_hash=hashlib.sha256(b"k").hexdigest())
        repo = _make_repo(existing_key=key)
        repo.touch_last_used = AsyncMock(side_effect=ConnectionRefusedError())

        assert await _service(repo).authenticate("k") is key.user

    @pytest.mark.asyncio
    async def test_a_nested_unit_of_work_is_a_bug_not_a_failed_write(self) -> None:
        # Sanity check SC3: the catch is for the database's errors only, so the
        # nesting guard can't be logged away as bookkeeping.
        key = _make_api_key(key_hash=hashlib.sha256(b"k").hexdigest())
        repo = _make_repo(existing_key=key)
        repo.touch_last_used = AsyncMock(side_effect=NestedUnitOfWorkError())

        with pytest.raises(NestedUnitOfWorkError):
            await _service(repo).authenticate("k")

    @pytest.mark.asyncio
    async def test_the_write_is_committed_in_its_own_unit_of_work(self) -> None:
        key = _make_api_key(key_hash=hashlib.sha256(b"k").hexdigest())
        uow = FakeUnitOfWork(api_keys=_make_repo(existing_key=key))

        await ApiKeyService(uow, ResponseCache(FakeRedis())).authenticate("k")

        assert (uow.opened, uow.commits) == (2, 1)  # the read, then the write


class TestRecordUseThrottle:
    """ADR-019: at most one ``last_used_at`` write per key per window."""

    @staticmethod
    def _key_repo() -> tuple[MagicMock, dict[str, MagicMock]]:
        keys: dict[str, MagicMock] = {}
        repo = _make_repo()

        async def _get_by_hash(key_hash: str) -> MagicMock:
            return keys.setdefault(key_hash, _make_api_key(key_hash=key_hash))

        repo.get_by_hash = AsyncMock(side_effect=_get_by_hash)
        return repo, keys

    @pytest.mark.asyncio
    async def test_writes_once_per_window_per_key(self) -> None:
        repo, keys = self._key_repo()
        service = _service(repo)

        await service.authenticate("key-a")
        await service.authenticate("key-a")
        await service.authenticate("key-b")  # another key has its own window

        written = [c.args[0] for c in repo.touch_last_used.await_args_list]
        assert sorted(map(str, written)) == sorted(str(k.id) for k in keys.values())

    @pytest.mark.asyncio
    async def test_a_failed_write_is_not_retried_within_the_window(self) -> None:
        # Deliberate (ADR-019): during a DB outage each key costs one failed write
        # per window, not one per request.
        repo, _ = self._key_repo()
        repo.touch_last_used = AsyncMock(side_effect=[_db_error("DB down"), None])
        service = _service(repo)

        await service.authenticate("key-a")
        await service.authenticate("key-a")

        assert repo.touch_last_used.await_count == 1

    @pytest.mark.asyncio
    async def test_fails_closed_without_redis(self) -> None:
        # No claim, no write: a Redis outage must not turn into a write per request.
        redis = MagicMock()
        redis.set = AsyncMock(side_effect=RedisConnectionError("Redis down"))
        repo, keys = self._key_repo()
        service = _service(repo, ResponseCache(redis))

        user = await service.authenticate("key-a")

        assert user is keys[hashlib.sha256(b"key-a").hexdigest()].user
        repo.touch_last_used.assert_not_awaited()


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_owner_can_deactivate(self) -> None:
        user_id = uuid.uuid4()
        key_id = uuid.uuid4()
        mock_key = _make_api_key(key_id=key_id, user_id=user_id, is_active=True)
        repo = _make_repo(user_keys=[mock_key])
        service = _service(repo)

        await service.deactivate(key_id=key_id, requesting_user_id=user_id)

        repo.deactivate.assert_awaited_once_with(key_id, user_id)

    @pytest.mark.asyncio
    async def test_unknown_key_raises_not_found(self) -> None:
        repo = _make_repo(user_keys=[])
        service = _service(repo)
        with pytest.raises(NotFoundError):
            await service.deactivate(
                key_id=uuid.uuid4(), requesting_user_id=uuid.uuid4()
            )
