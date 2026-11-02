from __future__ import annotations

import uuid

import sqlalchemy as sa

from src.domain.documents import Transition
from src.models.document import Document


def guarded_update(document_id: uuid.UUID, transition: Transition) -> sa.Update:
    """An UPDATE that applies ``transition`` only while the document is in one of its
    from-statuses; callers add ``.values()`` and ``.returning()``.

    Check and write are one step: an UPDATE that waits on a concurrent writer's row
    lock re-evaluates its WHERE against the committed row (READ COMMITTED). No row
    returned: another run moved the document first, or it is gone.
    """
    return (
        sa.update(Document)
        .where(
            Document.id == document_id,
            Document.status.in_(transition.allowed_from),
        )
        .values(status=transition.to)
    )
