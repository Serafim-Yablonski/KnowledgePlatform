from typing import Any


class AppError(Exception):
    status_code: int = 500
    detail: str = "Internal server error"
    errors: dict[str, Any] | None = None

    def __init__(
        self,
        detail: str | None = None,
        *,
        errors: dict[str, Any] | None = None,
    ) -> None:
        self.detail = detail if detail is not None else self.__class__.detail
        self.errors = errors
        super().__init__(self.detail)


class UnauthorizedError(AppError):
    status_code = 401
    detail = "Unauthorized"


class ForbiddenError(AppError):
    status_code = 403
    detail = "Forbidden"


class NotFoundError(AppError):
    status_code = 404
    detail = "Not found"


class ConflictError(AppError):
    status_code = 409
    detail = "Conflict"


class InputValidationError(AppError):
    """Domain validation error — distinct from pydantic.ValidationError."""

    status_code = 422
    detail = "Validation error"


class RateLimitError(AppError):
    status_code = 429
    detail = "Rate limit exceeded"

    def __init__(
        self,
        detail: str | None = None,
        *,
        retry_after: int = 60,
        errors: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail, errors=errors)
        self.retry_after = retry_after
