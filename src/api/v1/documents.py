import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request, UploadFile, status

from src.core.dependencies import (
    get_current_user,
    get_current_workspace,
    get_document_service,
)
from src.domain.documents import DocumentStatus
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.document import DocumentResponse, DocumentUpdate, PaginatedResponse
from src.services.document import DocumentService

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/documents",
    tags=["documents"],
)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    title: Annotated[str, Form(min_length=1, max_length=255)],
    file: UploadFile,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    return await service.create(
        actor, workspace, request.state.workspace_role, title, file
    )


@router.get("", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    cursor: str | None = Query(default=None),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    doc_status: DocumentStatus | None = Query(default=None, alias="status"),
    service: DocumentService = Depends(get_document_service),
) -> PaginatedResponse[DocumentResponse]:
    return await service.list(actor, workspace, cursor, limit, doc_status)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    return await service.get(actor, workspace, document_id)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    data: DocumentUpdate,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    return await service.update(
        actor, workspace, request.state.workspace_role, document_id, data
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> None:
    await service.delete(actor, workspace, request.state.workspace_role, document_id)
