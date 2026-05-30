"""
models/paper.py — Pydantic models for paper domain objects.

Three layers of models, each with a distinct job:

  PaperBase          — shared fields used by multiple schemas
  PaperUploadResponse — what the API returns after a successful upload
  PaperRecord        — the full internal representation (includes file path, etc.)
  PaperStatus        — enum for the paper's processing lifecycle

Rule: models here define shape and validation only.
      No database calls, no file I/O, no business logic.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

# ── Enums ─────────────────────────────────────────────────────────────────────


class PaperStatus(str, Enum):
    """
    Lifecycle of a paper through the system.

      UPLOADED  → file saved to disk, not yet processed
      PARSING   → PDF text extraction in progress
      CHUNKING  → text split into claim/evidence/abstract chunks
      INDEXING  → chunks being embedded and stored in Qdrant
      READY     → fully indexed, available for retrieval
      FAILED    → processing failed at some stage (see error_message)
    """

    UPLOADED = "uploaded"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# ── Base ──────────────────────────────────────────────────────────────────────


class PaperBase(BaseModel):
    """Fields shared across paper schemas."""

    title: str | None = Field(
        default=None,
        description="Extracted from PDF metadata or first heading. "
        "None until parsing completes.",
        max_length=512,
    )
    authors: list[str] = Field(
        default_factory=list,
        description="Author names extracted from PDF metadata.",
    )
    year: int | None = Field(
        default=None,
        description="Publication year extracted from PDF metadata.",
        ge=1900,
        le=2100,
    )
    arxiv_id: str | None = Field(
        default=None,
        description="arXiv identifier if detected (e.g. '2305.14314').",
        pattern=r"^\d{4}\.\d{4,5}(v\d+)?$",
    )


# ── API response models ───────────────────────────────────────────────────────


class PaperUploadResponse(PaperBase):
    """
    Returned immediately after a successful upload.
    Processing happens asynchronously — status starts as UPLOADED.
    Poll GET /papers/{paper_id} to track progress.
    """

    paper_id: str = Field(
        description="UUID identifying this paper. Use for all subsequent requests."
    )
    filename: str = Field(description="Original filename as uploaded by the user.")
    file_size_bytes: int = Field(
        description="File size in bytes.",
        ge=0,
    )
    status: PaperStatus = Field(
        default=PaperStatus.UPLOADED,
        description="Processing status. Starts as 'uploaded'.",
    )
    uploaded_at: datetime = Field(description="UTC timestamp of successful upload.")
    message: str = Field(description="Human-readable status message for the UI.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "paper_id": "a3f2c1d0-8b4e-4f7a-9c2d-1e5f6a7b8c9d",
                "filename": "attention_is_all_you_need.pdf",
                "file_size_bytes": 2_154_321,
                "status": "uploaded",
                "title": None,
                "authors": [],
                "year": None,
                "arxiv_id": None,
                "uploaded_at": "2025-03-15T10:30:00Z",
                "message": "Paper uploaded successfully. Processing will begin shortly.",
            }
        }
    }


class PaperRecord(PaperBase):
    """
    Full internal representation of a paper.
    Returned by GET /papers/{paper_id} — includes processing details.
    """

    paper_id: str
    filename: str
    file_size_bytes: int
    file_path: str = Field(description="Absolute path to the stored PDF on disk.")
    status: PaperStatus
    uploaded_at: datetime
    processed_at: datetime | None = Field(
        default=None,
        description="UTC timestamp when processing completed (READY or FAILED).",
    )
    section_count: int | None = Field(
        default=None,
        description="Number of sections created during processing.",
        ge=0,
    )
    error_message: str | None = Field(
        default=None,
        description="Error detail if status is FAILED.",
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Normalize to forward slashes for cross-platform consistency."""
        return Path(v).as_posix()

    @model_validator(mode="after")
    def check_failed_has_error(self) -> "PaperRecord":
        """FAILED papers must have an error_message for debuggability."""
        if self.status == PaperStatus.FAILED and not self.error_message:
            raise ValueError("FAILED papers must have an error_message")
        return self


class PaperListResponse(BaseModel):
    """Paginated list of papers — returned by GET /papers."""

    papers: list[PaperRecord]
    total: int = Field(description="Total number of papers (for pagination).")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
