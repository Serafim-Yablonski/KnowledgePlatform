"""MCP resources — read-only workspace views. No side effects."""

from __future__ import annotations

import uuid
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.core.database import get_session
from src.core.exceptions import InputValidationError
from src.mcp_server.auth import get_mcp_user


def _make_workspace_service(session: Any) -> Any:
    from src.repositories.user import SQLAlchemyUserRepository
    from src.repositories.workspace import SQLAlchemyWorkspaceRepository
    from src.services.workspace import WorkspaceService

    return WorkspaceService(
        workspace_repo=SQLAlchemyWorkspaceRepository(session),
        user_repo=SQLAlchemyUserRepository(session),
    )


def _make_document_service(session: Any) -> Any:
    from src.repositories.document import SQLAlchemyDocumentRepository
    from src.services.document import DocumentService

    return DocumentService(repo=SQLAlchemyDocumentRepository(session), session=session)


def _parse_workspace_id(workspace_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(workspace_id)
    except ValueError as exc:
        raise InputValidationError(f"Invalid workspace_id: {workspace_id!r}") from exc


async def list_workspace_documents(workspace_id: str) -> list[dict[str, Any]]:
    user = get_mcp_user()
    ws_uuid = _parse_workspace_id(workspace_id)
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        await workspace_svc.get_user_role(user, ws_uuid)
        doc_svc = _make_document_service(session)
        docs = await doc_svc._list_by_workspace_id(ws_uuid, limit=100)
    return [
        {
            "id": str(d.id),
            "title": d.title,
            "content_type": d.content_type,
            "status": d.status,
            "version": d.version,
            "file_size_bytes": d.file_size_bytes,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


async def get_workspace_stats(workspace_id: str) -> dict[str, Any]:
    user = get_mcp_user()
    ws_uuid = _parse_workspace_id(workspace_id)
    async with get_session() as session:
        workspace_svc = _make_workspace_service(session)
        await workspace_svc.get_user_role(user, ws_uuid)
        doc_svc = _make_document_service(session)
        stats = await doc_svc._get_workspace_stats(ws_uuid)
    return {
        "document_count": stats.document_count,
        "chunk_count": stats.chunk_count,
        "total_tokens_indexed": stats.total_tokens_indexed,
        "last_document_updated_at": (
            stats.last_document_updated_at.isoformat()
            if stats.last_document_updated_at
            else None
        ),
    }


def register_resources(mcp: FastMCP) -> None:
    mcp.resource(
        "workspace://documents/{workspace_id}",
        description=(
            "List of documents in the workspace (up to 100). Read-only. "
            "Includes id, title, content_type, status, and created_at."
        ),
    )(list_workspace_documents)

    mcp.resource(
        "workspace://stats/{workspace_id}",
        description=(
            "Aggregate statistics for the workspace knowledge base: "
            "document count, chunk count, total tokens indexed, and freshness."
        ),
    )(get_workspace_stats)
