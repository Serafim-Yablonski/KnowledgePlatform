import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        # Partial unique index: only active users occupy an email slot.
        # Deactivated accounts free the email for re-registration.
        sa.Index(
            "ix_users_email_active",
            "email",
            unique=True,
            postgresql_where=sa.text("is_active = true"),
        ),
        # Non-unique index for lookup performance (get_by_email queries all users).
        sa.Index("ix_users_email", "email"),
    )

    email: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    hashed_password: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("true")
    )
    display_name: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
