"""
api/v1/papers.py — PDF upload and paper management endpoints.

Endpoints implemented here:
  POST /api/v1/papers/upload  — Upload a PDF, validate it, save to disk
  GET  /api/v1/papers/{id}    — Get status/metadata for a specific paper
  GET  /api/v1/papers         — List all uploaded papers

Design decisions:
  - Validation happens in two stages:
      1. Fast checks (extension, content-type header) before reading the file
      2. Deep checks (magic bytes, actual file size) after reading
    This avoids streaming a 200MB file just to reject it on extension.

  - Files are saved with a UUID name, not the original filename.
    This prevents path traversal attacks and name collisions.

  - The original filename is preserved in the response for the UI.

  - Parsing is currently synchronous for Phase-1 development.
    Chunking, embedding, and indexing will move to async workers later.
    This endpoint only saves the file and returns immediately.
    A Celery task (future) picks it up asynchronously.

  - In-memory paper store is used now. Replace with PostgreSQL
    repository (db/repositories/paper_repo.py) in the next step.
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import aiofiles
from fastapi import APIRouter, File, Query, UploadFile, status
from app.services.pdf_parser import parse_pdf

from app.api.dependencies import SettingsDep, UploadDirDep
from app.core.exceptions import (
    DuplicatePaperError,
    FileTooLargeError,
    InvalidFileTypeError,
    NotFoundError,
    StorageError,
)
from app.core.logging import get_logger
from app.models.paper import (
    PaperListResponse,
    PaperRecord,
    PaperStatus,
    PaperUploadResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["papers"])

# ── In-memory store (replace with PostgreSQL repo in next step) ───────────────
# Keyed by paper_id. Thread-safe for single-process dev.
# Do NOT use this in production — it resets on every restart.
_paper_store: dict[str, PaperRecord] = {}

# Content-hash → paper_id map for duplicate detection
_hash_index: dict[str, str] = {}


# ── Validation helpers ────────────────────────────────────────────────────────

_ALLOWED_EXTENSIONS = {".pdf"}
_PDF_MAGIC_BYTES = b"%PDF"  # every valid PDF starts with these 4 bytes
_MAX_FILENAME_LENGTH = 255


def _validate_extension(filename: str) -> None:
    """
    First-pass check before reading the file.
    Rejects obviously wrong file types immediately.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(
            f"File '{filename}' has extension '{suffix}'. "
            f"Only PDF files are accepted."
        )


def _validate_filename(filename: str) -> None:
    """Reject filenames that could cause issues on disk."""
    if len(filename) > _MAX_FILENAME_LENGTH:
        raise InvalidFileTypeError(
            f"Filename exceeds {_MAX_FILENAME_LENGTH} characters."
        )
    # Reject null bytes and path separators — defense against path traversal
    forbidden = {"\x00", "/", "\\", "..", ":"}
    for char in forbidden:
        if char in filename and char not in (".", "/"):
            raise InvalidFileTypeError(
                f"Filename contains forbidden character: {repr(char)}"
            )


def _validate_content_type(content_type: str | None) -> None:
    """
    Check the Content-Type header the client declared.
    This is NOT a security check — headers are client-controlled.
    It catches honest mistakes (e.g. uploading a .txt file).
    The real security check is magic byte validation below.
    """
    allowed_content_types = {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",  # some browsers send this for PDFs
    }
    if content_type and content_type.split(";")[0].strip() not in allowed_content_types:
        raise InvalidFileTypeError(
            f"Content-Type '{content_type}' is not accepted for PDF uploads."
        )


def _validate_magic_bytes(content: bytes, filename: str) -> None:
    """
    Read the first 4 bytes of the file content to confirm it's a real PDF.
    This is the actual security check — it cannot be spoofed by renaming a file.
    A .pdf file that doesn't start with %PDF is corrupt or malicious.
    """
    if not content[:4] == _PDF_MAGIC_BYTES:
        raise InvalidFileTypeError(
            f"'{filename}' is not a valid PDF file. "
            "The file header does not match the PDF specification."
        )


def _compute_sha256(content: bytes) -> str:
    """Return hex SHA-256 of file content for duplicate detection."""
    return hashlib.sha256(content).hexdigest()


def _check_duplicate(content_hash: str) -> str | None:
    """Return existing paper_id if this content hash was already uploaded."""
    return _hash_index.get(content_hash)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=PaperUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a research paper PDF",
    description="""
Upload a PDF research paper for processing.

**Validation performed:**
- File extension must be `.pdf`
- File magic bytes must match the PDF specification (`%PDF`)
- File size must not exceed the configured limit (default 50MB)
- Duplicate PDFs (by content hash) are rejected with 409

**After upload:**
The file is saved to disk immediately. Processing (text extraction,
chunking, vector indexing) is queued asynchronously.
Poll `GET /papers/{paper_id}` to track processing status.
""",
    responses={
        201: {"description": "Paper uploaded successfully"},
        409: {"description": "Duplicate paper — already uploaded"},
        413: {"description": "File exceeds size limit"},
        415: {"description": "Not a valid PDF file"},
        500: {"description": "Disk write failed"},
    },
)
async def upload_paper(
    settings: SettingsDep,
    upload_dir: UploadDirDep,
    file: UploadFile = File(
        ...,
        description="PDF file to upload. Must be a valid PDF under 50MB.",
    ),
) -> PaperUploadResponse:
    """
    Upload a PDF research paper.

    Processing pipeline after this endpoint returns:
      1. This endpoint: validate → save to disk → return paper_id
      2. Celery task (future): parse → chunk → embed → index into Qdrant
    """
    original_filename = file.filename or "unknown.pdf"

    logger.info(
        "paper.upload_started",
        filename=original_filename,
        content_type=file.content_type,
    )

    # ── Stage 1: Fast validation (before reading file bytes) ─────────────────
    _validate_filename(original_filename)
    _validate_extension(original_filename)
    _validate_content_type(file.content_type)

    # ── Stage 2: Read file content ────────────────────────────────────────────
    # Read the entire file into memory for hashing and magic byte check.
    # 50MB ceiling means this is safe. For larger limits, use streaming.
    try:
        content: bytes = await file.read()
    except Exception as exc:
        logger.exception("paper.read_failed", filename=original_filename)
        raise StorageError(f"Failed to read uploaded file: {exc}") from exc

    file_size = len(content)

    logger.info(
        "paper.file_read",
        filename=original_filename,
        size_bytes=file_size,
    )

    # ── Stage 3: Deep validation (requires actual file bytes) ─────────────────
    if file_size > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            size_bytes=file_size,
            max_bytes=settings.max_upload_size_bytes,
        )

    if file_size == 0:
        raise InvalidFileTypeError("Uploaded file is empty.")

    _validate_magic_bytes(content, original_filename)

    # ── Stage 4: Duplicate detection ──────────────────────────────────────────
    content_hash = _compute_sha256(content)
    existing_id = _check_duplicate(content_hash)

    if existing_id:
        logger.info(
            "paper.duplicate_rejected",
            filename=original_filename,
            existing_paper_id=existing_id,
            content_hash=content_hash[:16],  # log prefix only
        )
        raise DuplicatePaperError(
            f"This PDF has already been uploaded (paper_id: {existing_id}). "
            "Use the existing paper_id for analysis."
        )

    # ── Stage 5: Persist to disk ───────────────────────────────────────────────
    # Use UUID as filename — never trust user-provided filenames for storage.
    paper_id = str(uuid.uuid4())
    safe_name = f"{paper_id}.pdf"
    file_path = upload_dir / safe_name

    try:
        async with aiofiles.open(file_path, "wb") as out:
            await out.write(content)
    except OSError as exc:
        logger.exception(
            "paper.disk_write_failed",
            paper_id=paper_id,
            path=str(file_path),
        )
        raise StorageError(f"Failed to save file to disk: {exc}") from exc

    # ── Stage 6: Register in store ─────────────────────────────────────────────

    now = datetime.now(timezone.utc)

    record = PaperRecord(
        paper_id=paper_id,
        filename=original_filename,
        file_size_bytes=file_size,
        file_path=str(file_path),
        status=PaperStatus.UPLOADED,
        uploaded_at=now,
    )

    _paper_store[paper_id] = record
    _hash_index[content_hash] = paper_id

    # ── Stage 7: Parse PDF immediately (temporary synchronous pipeline) ──

    try:
        logger.info(
            "paper.parsing_started",
            paper_id=paper_id,
        )

        # TODO:
        # Move parsing/chunking/indexing to Celery background workers
        # in Phase-3 to avoid blocking request latency.

        # TODO (Phase-3):
        # Move PDF parsing to a background worker to avoid
        # blocking upload requests for large documents.

        record.status = PaperStatus.PARSING

        parsed_paper = await parse_pdf(
            file_path=file_path,
            paper_id=paper_id,
        )

        # Store extracted metadata
        record.title = parsed_paper.metadata.title
        record.authors = parsed_paper.metadata.authors
        record.year = parsed_paper.metadata.year
        record.arxiv_id = parsed_paper.metadata.arxiv_id

        # Update status
        record.status = PaperStatus.READY
        record.processed_at = datetime.now(timezone.utc)
        record.section_count = len(parsed_paper.sections)

        logger.info(
            "paper.parsing_completed",
            paper_id=paper_id,
            sections=len(parsed_paper.sections),
        )

    except Exception as exc:
        logger.exception(
            "paper.parsing_failed",
            paper_id=paper_id,
            error=str(exc),
        )

        record.status = PaperStatus.FAILED
        record.error_message = str(exc)

    logger.info(
        "paper.upload_complete",
        paper_id=paper_id,
        filename=original_filename,
        size_bytes=file_size,
        size_mb=round(file_size / (1024 * 1024), 2),
        path=str(file_path),
    )

    return PaperUploadResponse(
        paper_id=paper_id,
        filename=original_filename,
        file_size_bytes=file_size,
        title=record.title,
        authors=record.authors,
        year=record.year,
        arxiv_id=record.arxiv_id,
        status=record.status,
        uploaded_at=now,
        message=(
            "Paper uploaded successfully."
            if record.status != PaperStatus.READY
            else "Paper uploaded and parsed successfully."
        ),
    )


@router.get(
    "/{paper_id}",
    response_model=PaperRecord,
    summary="Get paper status and metadata",
    description="Returns the full record for a paper including processing status.",
    responses={
        404: {"description": "Paper not found"},
    },
)
async def get_paper(paper_id: str) -> PaperRecord:
    """
    Retrieve a paper by its ID.

    The `status` field reflects where the paper is in the pipeline:
      - `uploaded`  — saved to disk, awaiting processing
      - `parsing`   — PDF text extraction in progress
      - `chunking`  — splitting into claim/evidence chunks
      - `indexing`  — embedding and writing to Qdrant
      - `ready`     — fully indexed, available for analysis
      - `failed`    — processing failed (see `error_message`)
    """
    record = _paper_store.get(paper_id)
    if not record:
        raise NotFoundError(f"Paper '{paper_id}' not found.")

    logger.info("paper.get", paper_id=paper_id, status=record.status)
    return record


@router.get(
    "",
    response_model=PaperListResponse,
    summary="List all uploaded papers",
    description="Returns a paginated list of all papers, newest first.",
)
async def list_papers(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Results per page")] = 20,
    status: Annotated[PaperStatus | None, Query(description="Filter by status")] = None,
) -> PaperListResponse:
    """List papers with optional status filter and pagination."""
    all_papers = list(_paper_store.values())

    # Sort newest first
    all_papers.sort(key=lambda p: p.uploaded_at, reverse=True)

    # Optional status filter
    if status is not None:
        all_papers = [p for p in all_papers if p.status == status]

    total = len(all_papers)

    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    page_items = all_papers[start:end]

    logger.info(
        "paper.list",
        total=total,
        page=page,
        page_size=page_size,
        status_filter=status,
    )

    return PaperListResponse(
        papers=page_items,
        total=total,
        page=page,
        page_size=page_size,
    )
