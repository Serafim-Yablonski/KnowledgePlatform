import uuid

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.documents import ContentType, Cursor, DocumentStatus
from src.models.document import Document
from src.schemas.document import DocumentUpdate


class SQLAlchemyDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        workspace_id: uuid.UUID,
        title: str,
        content_type: ContentType,
        file_path: str,
        file_size_bytes: int,
        uploaded_by: uuid.UUID,
    ) -> Document:
        doc = Document(
            workspace_id=workspace_id,
            title=title,
            content_type=content_type,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            uploaded_by=uploaded_by,
            status=DocumentStatus.PENDING,
            version=1,
        )
        self._session.add(doc)
        # Flush to get the server-generated uuidv7 ID without committing.
        # The caller (DocumentService.create) writes the file and commits once
        # after updating file_path, keeping the whole operation in one transaction.
        await self._session.flush()
        await self._session.refresh(doc)
        return doc

    async def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        result = await self._session.scalars(
            sa.select(Document).where(Document.id == document_id)
        )
        return result.first()

    async def update(self, document: Document, data: DocumentUpdate) -> Document:
        if data.title is not None:
            document.title = data.title
        document.version += 1
        await self._session.commit()
        await self._session.refresh(document)
        return document

    async def delete(self, document: Document) -> None:
        await self._session.delete(document)
        await self._session.commit()

    async def list_by_workspace(
        self,
        workspace_id: uuid.UUID,
        *,
        limit: int = 20,
        cursor: Cursor | None = None,
        status: DocumentStatus | None = None,
    ) -> tuple[list[Document], Cursor | None]:
        query = sa.select(Document).where(Document.workspace_id == workspace_id)
        if cursor is not None:
            # Uses composite index on (workspace_id, created_at, id).
            query = query.where(
                sa.or_(
                    Document.created_at < cursor.created_at,
                    sa.and_(
                        Document.created_at == cursor.created_at,
                        Document.id < cursor.id,
                    ),
                )
            )
        if status is not None:
            query = query.where(Document.status == status)
        query = query.order_by(Document.created_at.desc(), Document.id.desc()).limit(
            limit + 1
        )
        result = await self._session.scalars(query)
        rows = list(result.all())
        next_cursor: Cursor | None
        if len(rows) > limit:
            rows.pop()
            last = rows[-1]
            next_cursor = Cursor(created_at=last.created_at, id=last.id)
        else:
            next_cursor = None
        return rows, next_cursor
