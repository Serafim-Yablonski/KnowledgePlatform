from fastapi import APIRouter

from src.api.v1.ai import router as ai_router
from src.api.v1.auth import router as auth_router
from src.api.v1.documents import router as document_router
from src.api.v1.research import router as research_router
from src.api.v1.search import router as search_router
from src.api.v1.workspaces import router as workspace_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(workspace_router)
router.include_router(document_router)
router.include_router(search_router)
router.include_router(ai_router)
router.include_router(research_router)
