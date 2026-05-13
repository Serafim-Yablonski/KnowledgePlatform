import uuid
from datetime import datetime

from sqlalchemy import func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk}>"


class TimestampMixin:
    """Provides created_at / updated_at for models that manage their own PK."""

    created_at: Mapped[datetime] = mapped_column(server_default=text("now()"))
    updated_at: Mapped[datetime] = mapped_column(
        server_default=text("now()"), onupdate=func.now()
    )


class HasIDMixin(TimestampMixin):
    """Surrogate UUID v7 primary key + timestamps — the standard model base."""

    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=text("uuidv7()")
    )
