from __future__ import annotations

from fastapi import APIRouter, Depends

from src.core.dependencies import get_current_workspace, get_search_service
from src.core.rate_limit import rate_limit
from src.models.workspace import Workspace
from src.schemas.search import SearchRequest, SearchResponse, SearchResultItem
from src.services.search import SearchService

router = APIRouter(
    prefix="/api/v1/workspaces/{workspace_id}/search",
    tags=["search"],
)

_search_rate_limit = rate_limit("search", 20, 60)


@router.post(
    "",
    response_model=SearchResponse,
    dependencies=[Depends(_search_rate_limit)],
)
async def search_documents(
    body: SearchRequest,
    workspace: Workspace = Depends(get_current_workspace),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    domain = await service.search(
        workspace_id=workspace.id,
        query=body.query,
        top_k=body.top_k,
        min_score=body.min_score,
    )
    return SearchResponse(
        results=[
            SearchResultItem(
                chunk_text=r.chunk_text,
                document_id=r.document_id,
                document_title=r.document_title,
                score=r.score,
                chunk_metadata=r.chunk_metadata,
            )
            for r in domain.results
        ],
        query=domain.query,
        total_results=domain.total_results,
    )
