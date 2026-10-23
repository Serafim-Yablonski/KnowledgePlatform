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
    """No usable credentials: the client should (re-)authenticate."""

    status_code = 401
    detail = "Unauthorized"
    # RFC 6750 §3 error code; None when the request carried no credentials.
    bearer_error: str | None = None

    @property
    def www_authenticate(self) -> str:
        """The ``WWW-Authenticate`` challenge every 401 must carry."""
        challenge = 'Bearer realm="knowledge-platform"'
        if self.bearer_error is None:
            return challenge
        return f'{challenge}, error="{self.bearer_error}"'


class InvalidTokenError(UnauthorizedError):
    """A token or API key was sent and is bad: malformed, expired, revoked, wrong
    type, or its user is gone or inactive. Not for "authenticated but not allowed"
    — that is ForbiddenError."""

    detail = "Invalid token"
    bearer_error = "invalid_token"


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
        limit: int = 0,
        errors: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail, errors=errors)
        self.retry_after = retry_after
        self.limit = limit


class ServiceUnavailableError(AppError):
    status_code = 503
    detail = "Service temporarily unavailable"
