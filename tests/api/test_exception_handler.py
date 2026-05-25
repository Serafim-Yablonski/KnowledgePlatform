"""Unit tests for the AppError exception handler and unhandled exception handler."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.core.exceptions import (
    AppError,
    ConflictError,
    ForbiddenError,
    InputValidationError,
    NotFoundError,
    RateLimitError,
    UnauthorizedError,
)
from src.main import _app_error_handler, _unhandled_error_handler


def _client_raising(exc: Exception) -> TestClient:
    application = FastAPI()
    application.add_exception_handler(AppError, _app_error_handler)
    application.add_exception_handler(Exception, _unhandled_error_handler)

    @application.get("/_test/raise")
    async def _raise() -> None:
        raise exc

    return TestClient(application, raise_server_exceptions=False)


def test_not_found_error() -> None:
    response = _client_raising(NotFoundError("thing not found")).get("/_test/raise")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == 404
    assert body["error"]["message"] == "thing not found"
    assert body["error"]["errors"] is None


def test_unauthorized_error() -> None:
    response = _client_raising(UnauthorizedError("token expired")).get("/_test/raise")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401


def test_forbidden_error() -> None:
    response = _client_raising(ForbiddenError("access denied")).get("/_test/raise")
    assert response.status_code == 403
    assert response.json()["error"]["code"] == 403


def test_conflict_error() -> None:
    response = _client_raising(ConflictError("already exists")).get("/_test/raise")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == 409


def test_input_validation_error_with_errors_dict() -> None:
    exc = InputValidationError("invalid input", errors={"field": "required"})
    response = _client_raising(exc).get("/_test/raise")
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["errors"] == {"field": "required"}
    assert body["error"]["message"] == "invalid input"


def test_input_validation_error_default_message() -> None:
    response = _client_raising(InputValidationError()).get("/_test/raise")
    assert response.status_code == 422
    assert response.json()["error"]["message"] == "Validation error"


def test_rate_limit_error() -> None:
    exc = RateLimitError("slow down", retry_after=30)
    response = _client_raising(exc).get("/_test/raise")
    assert response.status_code == 429
    assert response.json()["error"]["code"] == 429
    assert response.headers["retry-after"] == "30"


def test_app_error_base_preserves_status_code() -> None:
    exc = AppError("custom error")
    exc.status_code = 418
    response = _client_raising(exc).get("/_test/raise")
    assert response.status_code == 418
    assert response.json()["error"]["code"] == 418


def test_unhandled_exception_returns_500() -> None:
    response = _client_raising(RuntimeError("boom")).get("/_test/raise")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == 500
    assert "boom" not in body["error"]["message"]
    assert body["error"]["message"] == "Internal server error"


def test_error_response_structure() -> None:
    """All error responses share the same envelope shape."""
    response = _client_raising(NotFoundError("x")).get("/_test/raise")
    body = response.json()
    assert set(body.keys()) == {"error"}
    assert set(body["error"].keys()) == {"code", "message", "errors"}
