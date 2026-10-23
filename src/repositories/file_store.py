"""Uploaded files on the local filesystem: <root>/<workspace_id>/<document_id><ext>.

Paths are built from IDs only, so a client's filename never reaches the disk layout.
Blocking I/O runs in a worker thread.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

import anyio
import structlog

from src.domain.documents import ContentType

logger = structlog.get_logger(__name__)

_CHUNK = 65536

_EXTENSIONS: dict[ContentType, str] = {
    ContentType.PDF: ".pdf",
    ContentType.MARKDOWN: ".md",
    ContentType.PLAINTEXT: ".txt",
}


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(src: BinaryIO, dest_path: Path) -> None:
    """Write durably: data and directory entries are on disk when this returns.

    A file this call created is removed again if anything fails.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # "x": an existing file is never overwritten — it would be another document's.
    # Raised here, FileExistsError leaves that file alone: it isn't ours.
    dest = dest_path.open("xb")
    try:
        with dest:
            while chunk := src.read(_CHUNK):
                dest.write(chunk)
            dest.flush()
            os.fsync(dest.fileno())
        # The file's name, and the workspace directory's own name when it is new.
        _fsync_dir(dest_path.parent)
        _fsync_dir(dest_path.parent.parent)
    except BaseException:
        dest_path.unlink(missing_ok=True)
        raise


class LocalFileStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def save(
        self,
        workspace_id: uuid.UUID,
        document_id: uuid.UUID,
        content_type: ContentType,
        stream: BinaryIO,
    ) -> str:
        # The key is the absolute path for now: stored rows and the worker use it.
        path = (
            self._root / str(workspace_id) / f"{document_id}{_EXTENSIONS[content_type]}"
        )
        await anyio.to_thread.run_sync(_write_file, stream, path)
        return str(path)

    async def remove(self, key: str) -> None:
        await anyio.to_thread.run_sync(lambda: Path(key).unlink(missing_ok=True))

    async def remove_workspace(self, workspace_id: uuid.UUID) -> None:
        directory = self._root / str(workspace_id)

        def log_failure(_func: object, path: object, exc: BaseException) -> None:
            # Keep going: remove all that can be removed; leftovers are only garbage.
            logger.warning(
                "failed to remove workspace file",
                workspace_id=str(workspace_id),
                path=str(path),
                errno=getattr(exc, "errno", None),
            )

        def remove_tree() -> None:
            if directory.exists():
                shutil.rmtree(directory, onexc=log_failure)

        await anyio.to_thread.run_sync(remove_tree)
