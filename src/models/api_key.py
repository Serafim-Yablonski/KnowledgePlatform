from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, HasIDMixin

if TYPE_CHECKING:
    from src.models.user import User


class ApiKey(HasIDMixin, Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        sa.Index("ix_api_keys_key_hash", "key_hash", unique=True),
        sa.Index("ix_api_keys_user_id", "user_id"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    key_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False, unique=True)
    prefix: Mapped[str] = mapped_column(sa.String(8), nullable=False)
    name: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    last_used_at: Mapped[datetime | None] = mapped_column(sa.DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true")
    )

    user: Mapped[User] = relationship("User", lazy="raise")
