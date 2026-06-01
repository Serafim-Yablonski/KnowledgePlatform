"""Unit tests for ApiKeyService with Protocol stubs."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from src.services.api_key import ApiKeyService


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
    repo.invalidate_all_for_user = AsyncMock(return_value=None)
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
        service = ApiKeyService(repo)

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
        service = ApiKeyService(repo)
        _, raw_key = await service.create(user_id=uuid.uuid4(), name="k")
        call_kwargs = repo.create.call_args.kwargs
        assert call_kwargs["prefix"] == raw_key[:8]

    @pytest.mark.asyncio
    async def test_returns_raw_key_only_once(self) -> None:
        repo = _make_repo(active_count=0)
        service = ApiKeyService(repo)
        _api_key, raw_key = await service.create(user_id=uuid.uuid4(), name="k")
        assert len(raw_key) > 8

    @pytest.mark.asyncio
    async def test_sixth_key_raises_conflict(self) -> None:
        repo = _make_repo(active_count=5)
        service = ApiKeyService(repo)
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
        service = ApiKeyService(repo)

        user = await service.authenticate(raw_key)

        assert user is expected_user
        repo.get_by_hash.assert_awaited_once_with(key_hash)

    @pytest.mark.asyncio
    async def test_wrong_key_raises(self) -> None:
        repo = _make_repo(existing_key=None)
        service = ApiKeyService(repo)
        with pytest.raises(ForbiddenError):
            await service.authenticate("wrongkey")

    @pytest.mark.asyncio
    async def test_deactivated_key_raises(self) -> None:
        raw_key = "validbutdeactivated"
        mock_key = _make_api_key(is_active=False)
        repo = _make_repo(existing_key=mock_key)
        service = ApiKeyService(repo)
        with pytest.raises(ForbiddenError, match="revoked"):
            await service.authenticate(raw_key)

    @pytest.mark.asyncio
    async def test_inactive_user_raises(self) -> None:
        raw_key = "keyforinactiveuser"
        inactive_user = MagicMock()
        inactive_user.is_active = False
        mock_key = _make_api_key(is_active=True, user=inactive_user)
        repo = _make_repo(existing_key=mock_key)
        service = ApiKeyService(repo)
        with pytest.raises(ForbiddenError, match="revoked"):
            await service.authenticate(raw_key)


class TestInvalidateUserKeys:
    @pytest.mark.asyncio
    async def test_delegates_to_repo(self) -> None:
        user_id = uuid.uuid4()
        repo = _make_repo()
        service = ApiKeyService(repo)

        await service.invalidate_user_keys(user_id)

        repo.invalidate_all_for_user.assert_awaited_once_with(user_id)


class TestDeactivate:
    @pytest.mark.asyncio
    async def test_owner_can_deactivate(self) -> None:
        user_id = uuid.uuid4()
        key_id = uuid.uuid4()
        mock_key = _make_api_key(key_id=key_id, user_id=user_id, is_active=True)
        repo = _make_repo(user_keys=[mock_key])
        service = ApiKeyService(repo)

        await service.deactivate(key_id=key_id, requesting_user_id=user_id)

        repo.deactivate.assert_awaited_once_with(key_id, user_id)

    @pytest.mark.asyncio
    async def test_unknown_key_raises_not_found(self) -> None:
        repo = _make_repo(user_keys=[])
        service = ApiKeyService(repo)
        with pytest.raises(NotFoundError):
            await service.deactivate(
                key_id=uuid.uuid4(), requesting_user_id=uuid.uuid4()
            )
