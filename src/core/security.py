import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from src.core.config import settings
from src.core.exceptions import ForbiddenError
from src.schemas.auth import TokenPayload


async def hash_password(plain: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()
    )


async def verify_password(plain: str, hashed: str) -> bool:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, lambda: bool(bcrypt.checkpw(plain.encode(), hashed.encode()))
        )
    except ValueError:
        return False


def create_access_token(
    user_id: uuid.UUID, expires_delta: timedelta | None = None
) -> str:
    now = datetime.now(UTC)
    delta = expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + delta).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: uuid.UUID) -> str:
    now = datetime.now(UTC)
    delta = timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int((now + delta).timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload:
    try:
        raw = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return TokenPayload.model_validate(raw)
    except jwt.PyJWTError as exc:
        raise ForbiddenError("Invalid or expired token") from exc
