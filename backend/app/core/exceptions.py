"""
core/exceptions.py — Custom exception hierarchy.

All application exceptions inherit from RCABaseException.
The exception handler in main.py catches these and formats them into
consistent JSON error responses. This means route handlers never write
JSONResponse directly — they raise exceptions and the handler formats them.

Usage in route handlers:
    raise NotFoundError(f"Paper {paper_id} not found")
    raise FileTooLargeError(size_bytes=file.size, max_bytes=50*1024*1024)

Usage in tests:
    with pytest.raises(NotFoundError) as exc_info:
        await get_paper("nonexistent-id")
    assert exc_info.value.status_code == 404
"""

from typing import Any


class RCABaseException(Exception):
    """
    Root exception for all application errors.
    Never raise this directly — raise a specific subclass.
    """

    status_code: int = 500
    detail: str = "An unexpected error occurred"

    def __init__(self, detail: str | None = None, **context: Any):
        self.detail = detail or self.__class__.detail
        self.context = context  # extra fields for structured logging
        super().__init__(self.detail)


# ── 4xx Client errors ─────────────────────────────────────────────────────────


class NotFoundError(RCABaseException):
    """Resource does not exist."""

    status_code = 404
    detail = "Resource not found"


class ValidationError(RCABaseException):
    """Request data failed business-logic validation."""

    status_code = 422
    detail = "Validation failed"


class InvalidFileTypeError(RCABaseException):
    """Uploaded file is not an accepted type."""

    status_code = 415
    detail = "Unsupported file type"


class FileTooLargeError(RCABaseException):
    """Uploaded file exceeds the configured size limit."""

    status_code = 413
    detail = "File too large"

    def __init__(self, size_bytes: int, max_bytes: int):
        size_mb = round(size_bytes / (1024 * 1024), 2)
        max_mb = round(max_bytes / (1024 * 1024), 2)
        super().__init__(
            f"File is {size_mb}MB. Maximum allowed size is {max_mb}MB.",
            size_bytes=size_bytes,
            max_bytes=max_bytes,
        )


class DuplicatePaperError(RCABaseException):
    """The same PDF (by content hash) has already been uploaded."""

    status_code = 409
    detail = "Paper already exists"


class RateLimitError(RCABaseException):
    """Too many requests from this client."""

    status_code = 429
    detail = "Rate limit exceeded"


# ── 5xx Server errors ─────────────────────────────────────────────────────────


class StorageError(RCABaseException):
    """File system or object storage operation failed."""

    status_code = 500
    detail = "File storage operation failed"


class ServiceUnavailableError(RCABaseException):
    """Downstream service (Qdrant, Redis, Postgres) is unreachable."""

    status_code = 503
    detail = "Downstream service unavailable"


class ProcessingError(RCABaseException):
    """PDF parsing or ML processing pipeline failed."""

    status_code = 500
    detail = "Paper processing failed"
