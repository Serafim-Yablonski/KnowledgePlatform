import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request, UploadFile, status

from src.core.config import get_settings
from src.core.dependencies import (
    get_current_user,
    get_current_workspace,
    get_document_service,
)
from src.core.rate_limit import workspace_rate_limit
from src.domain.documents import (
    DEFAULT_PAGE_SIZE,
    DocumentPageSize,
    DocumentStatus,
    DocumentUpdateInput,
    DocumentUpload,
)
from src.models.user import User
from src.models.workspace import Workspace
from src.schemas.document import DocumentResponse, DocumentUpdate, PaginatedResponse
from src.services.document import DocumentService

_cfg = get_settings()
_upload_rate_limit = workspace_rate_limit(
    "document_upload",
    _cfg.RATE_LIMIT_DOCUMENT_UPLOAD_REQUESTS,
    _cfg.RATE_LIMIT_DOCUMENT_UPLOAD_WINDOW,
)

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/documents",
    tags=["documents"],
)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(_upload_rate_limit)],
)
async def upload_document(
    file: UploadFile,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
    # Optional: defaults to the file's name without its extension. The service
    # checks the cleaned title again (NFC can change the length).
    title: Annotated[str | None, Form(max_length=255)] = None,
) -> DocumentResponse:
    upload = DocumentUpload(
        filename=file.filename or "",
        content_type=file.content_type or "",
        declared_size=file.size,
        stream=file.file,
    )
    doc = await service.create(
        actor, workspace.id, request.state.workspace_role, title, upload
    )
    return DocumentResponse.model_validate(doc)


@router.get("", response_model=PaginatedResponse[DocumentResponse])
async def list_documents(
    workspace: Workspace = Depends(get_current_workspace),
    cursor: str | None = Query(default=None),
    limit: Annotated[DocumentPageSize, Query()] = DEFAULT_PAGE_SIZE,
    doc_status: DocumentStatus | None = Query(default=None, alias="status"),
    service: DocumentService = Depends(get_document_service),
) -> PaginatedResponse[DocumentResponse]:
    page = await service.list(workspace.id, cursor, limit, doc_status)
    return PaginatedResponse(
        items=[DocumentResponse.model_validate(d) for d in page.items],
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    workspace: Workspace = Depends(get_current_workspace),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    doc = await service.get(workspace.id, document_id)
    return DocumentResponse.model_validate(doc)


@router.patch("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: uuid.UUID,
    data: DocumentUpdate,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    domain_data = DocumentUpdateInput(title=data.title)
    doc = await service.update(
        actor, workspace.id, request.state.workspace_role, document_id, domain_data
    )
    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    workspace: Workspace = Depends(get_current_workspace),
    actor: User = Depends(get_current_user),
    service: DocumentService = Depends(get_document_service),
) -> None:
    await service.delete(actor, workspace.id, request.state.workspace_role, document_id)
