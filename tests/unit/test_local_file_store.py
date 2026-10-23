"""LocalFileStore: durable, never overwrites, and a failed save leaves nothing."""

import errno
import io
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import structlog

from src.domain.documents import ContentType
from src.repositories.file_store import LocalFileStore


class _FailsOnSecondRead(io.BytesIO):
    def __init__(self, content: bytes) -> None:
        super().__init__(content)
        self._reads = 0

    def read(self, size: int | None = -1) -> bytes:
        self._reads += 1
        if self._reads == 2:
            raise OSError(errno.EIO, "I/O error")
        return super().read(size)


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


async def test_save_stores_under_workspace_and_document_ids(tmp_path: Path) -> None:
    ws, doc = uuid.uuid4(), uuid.uuid4()

    key = await LocalFileStore(tmp_path).save(
        ws, doc, ContentType.MARKDOWN, io.BytesIO(b"# hi")
    )

    assert Path(key) == tmp_path / str(ws) / f"{doc}.md"
    assert Path(key).read_bytes() == b"# hi"


async def test_save_is_durable_when_it_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []
    real_fsync = os.fsync

    def recording_fsync(fd: int) -> None:
        synced.append(os.fstat(fd).st_ino)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)

    key = await LocalFileStore(tmp_path).save(
        uuid.uuid4(), uuid.uuid4(), ContentType.PLAINTEXT, io.BytesIO(b"x")
    )

    # The file's data, its name in the workspace dir, and the workspace dir's own name.
    path = Path(key)
    assert {p.stat().st_ino for p in (path, path.parent, path.parent.parent)} <= set(
        synced
    )


async def test_read_error_mid_copy_leaves_no_file(tmp_path: Path) -> None:
    stream = _FailsOnSecondRead(b"a" * 100_000)

    with pytest.raises(OSError):
        await LocalFileStore(tmp_path).save(
            uuid.uuid4(), uuid.uuid4(), ContentType.PLAINTEXT, stream
        )

    assert _files(tmp_path) == []


async def test_fsync_failure_leaves_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def disk_full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", disk_full)

    with pytest.raises(OSError):
        await LocalFileStore(tmp_path).save(
            uuid.uuid4(), uuid.uuid4(), ContentType.PLAINTEXT, io.BytesIO(b"x")
        )

    assert _files(tmp_path) == []


async def test_save_never_overwrites(tmp_path: Path) -> None:
    ws, doc = uuid.uuid4(), uuid.uuid4()
    target = tmp_path / str(ws) / f"{doc}.txt"
    target.parent.mkdir()
    target.write_bytes(b"someone else's document")

    with pytest.raises(FileExistsError):
        await LocalFileStore(tmp_path).save(
            ws, doc, ContentType.PLAINTEXT, io.BytesIO(b"new")
        )

    assert target.read_bytes() == b"someone else's document"


async def test_remove_tolerates_a_missing_file(tmp_path: Path) -> None:
    await LocalFileStore(tmp_path).remove(str(tmp_path / "gone.txt"))


async def _store_two_layouts(root: Path, ws: uuid.UUID) -> list[Path]:
    new = Path(
        await LocalFileStore(root).save(
            ws, uuid.uuid4(), ContentType.PDF, io.BytesIO(b"%PDF")
        )
    )
    # Before step 2 uploads were stored as <ws>/<doc_id>/<client name>.
    old = root / str(ws) / str(uuid.uuid4()) / "report.pdf"
    old.parent.mkdir()
    old.write_bytes(b"%PDF")
    return [new, old]


async def test_remove_workspace_removes_both_layouts_and_nothing_else(
    tmp_path: Path,
) -> None:
    ws, other = uuid.uuid4(), uuid.uuid4()
    await _store_two_layouts(tmp_path, ws)
    kept = await _store_two_layouts(tmp_path, other)

    await LocalFileStore(tmp_path).remove_workspace(ws)

    assert not (tmp_path / str(ws)).exists()
    assert sorted(_files(tmp_path)) == sorted(kept)


async def test_remove_workspace_without_uploads_is_fine(tmp_path: Path) -> None:
    await LocalFileStore(tmp_path).remove_workspace(uuid.uuid4())


async def test_remove_workspace_continues_past_a_failure_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = uuid.uuid4()
    new, old = await _store_two_layouts(tmp_path, ws)
    real_unlink = os.unlink

    def stuck(path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == new.name:
            raise PermissionError(errno.EACCES, "Permission denied")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", stuck)

    with structlog.testing.capture_logs() as logs:
        await LocalFileStore(tmp_path).remove_workspace(ws)

    assert new.exists()
    assert not old.exists()
    # The stuck file is reported; so is its directory, which then isn't empty.
    failures = [e for e in logs if e["event"] == "failed to remove workspace file"]
    assert {
        (e["workspace_id"], e["errno"])
        for e in failures
        if e["path"].endswith(new.name)
    } == {(str(ws), errno.EACCES)}
