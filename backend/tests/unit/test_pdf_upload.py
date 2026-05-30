"""
tests/unit/test_pdf_parser.py — Unit tests for the PDF upload endpoint.

These tests run without uvicorn, without real files on disk, and without
any external services. FastAPI's TestClient handles HTTP simulation.

Run with:
    pytest tests/unit/test_pdf_parser.py -v
    pytest tests/unit/test_pdf_parser.py -v --tb=short   # shorter tracebacks
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    TestClient wrapping the full FastAPI app.
    Uses a temporary upload directory so tests don't write to the real uploads/.
    The tmp_path_factory is module-scoped — one temp dir for all tests in this file.
    """
    tmp_dir = tmp_path_factory.mktemp("uploads")

    # Override the upload directory dependency to use the temp path
    from app.api.dependencies import get_upload_dir

    app.dependency_overrides[get_upload_dir] = lambda: tmp_dir

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

    # Cleanup overrides after tests complete
    app.dependency_overrides.clear()


def _make_pdf(content: bytes = b"%PDF-1.4 fake content") -> tuple[str, bytes, str]:
    """Helper: returns (filename, content, content_type) for a valid-looking PDF."""
    return ("test_paper.pdf", content, "application/pdf")


# ── Happy path ────────────────────────────────────────────────────────────────


class TestUploadSuccess:
    def test_returns_201(self, client):
        name, data, ct = _make_pdf()
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        assert response.status_code == 201

    def test_response_has_paper_id(self, client):
        name, data, ct = _make_pdf(b"%PDF-1.4 unique content abc")
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        body = response.json()
        assert "paper_id" in body
        assert len(body["paper_id"]) == 36  # UUID4 format

    def test_response_has_expected_fields(self, client):
        name, data, ct = _make_pdf(b"%PDF-1.4 field check content")
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        body = response.json()
        required_fields = {
            "paper_id",
            "filename",
            "file_size_bytes",
            "status",
            "uploaded_at",
            "message",
        }
        assert required_fields.issubset(body.keys())

    def test_status_is_uploaded(self, client):
        name, data, ct = _make_pdf(b"%PDF-1.4 status check content xyz")
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        assert response.json()["status"] in (
            "ready",
            "failed",
        )

    def test_filename_preserved_in_response(self, client):
        name, data, ct = _make_pdf(b"%PDF-1.4 filename test content 999")
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        assert response.json()["filename"] == name

    def test_file_size_correct(self, client):
        payload = b"%PDF-1.4 size test content"
        name, data, ct = _make_pdf(payload)
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        assert response.json()["file_size_bytes"] == len(payload)

    def test_paper_retrievable_after_upload(self, client):
        name, data, ct = _make_pdf(b"%PDF-1.4 retrieve test content abc123")
        upload_resp = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        paper_id = upload_resp.json()["paper_id"]
        get_resp = client.get(f"/api/v1/papers/{paper_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["paper_id"] == paper_id

    def test_file_saved_to_upload_dir(self, client, tmp_path_factory):
        """Verify the actual file lands on disk with a .pdf extension."""
        name, data, ct = _make_pdf(b"%PDF-1.4 disk write test content")
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        paper_id = response.json()["paper_id"]
        get_resp = client.get(f"/api/v1/papers/{paper_id}")
        file_path = Path(get_resp.json()["file_path"])
        assert file_path.exists()
        assert file_path.suffix == ".pdf"


# ── Duplicate detection ───────────────────────────────────────────────────────


class TestDuplicateDetection:
    def test_duplicate_returns_409(self, client):
        payload = b"%PDF-1.4 duplicate detection test unique payload zzz"
        name, data, ct = _make_pdf(payload)

        # First upload succeeds
        first = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(data), ct)},
        )
        assert first.status_code == 201

        # Second upload of same content rejected
        second = client.post(
            "/api/v1/papers/upload",
            files={"file": (name, io.BytesIO(payload), ct)},
        )
        assert second.status_code == 409

    def test_duplicate_error_contains_existing_id(self, client):
        payload = b"%PDF-1.4 duplicate with id check payload qqq"
        name, data, ct = _make_pdf(payload)

        first = client.post(
            "/api/v1/papers/upload", files={"file": (name, io.BytesIO(payload), ct)}
        )
        second = client.post(
            "/api/v1/papers/upload", files={"file": (name, io.BytesIO(payload), ct)}
        )

        assert first.json()["paper_id"] in second.json()["detail"]


# ── Validation errors ─────────────────────────────────────────────────────────


class TestValidation:
    def test_wrong_extension_returns_415(self, client):
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": ("paper.txt", io.BytesIO(b"%PDF fake"), "text/plain")},
        )
        assert response.status_code == 415

    def test_empty_file_returns_415(self, client):
        response = client.post(
            "/api/v1/papers/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
        )
        assert response.status_code == 415

    def test_fake_pdf_wrong_magic_bytes_returns_415(self, client):
        """A file named .pdf but starting with wrong bytes is rejected."""
        response = client.post(
            "/api/v1/papers/upload",
            files={
                "file": (
                    "fake.pdf",
                    io.BytesIO(b"PK\x03\x04 this is a zip"),
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 415

    def test_valid_magic_bytes_accepted(self, client):
        """Confirm %PDF prefix is what makes a file valid."""
        response = client.post(
            "/api/v1/papers/upload",
            files={
                "file": (
                    "real.pdf",
                    io.BytesIO(b"%PDF-1.7 content here xyz"),
                    "application/pdf",
                )
            },
        )
        assert response.status_code == 201


# ── Paper retrieval ───────────────────────────────────────────────────────────


class TestPaperRetrieval:
    def test_get_nonexistent_paper_returns_404(self, client):
        response = client.get("/api/v1/papers/nonexistent-id-12345")
        assert response.status_code == 404

    def test_list_papers_returns_200(self, client):
        response = client.get("/api/v1/papers")
        assert response.status_code == 200

    def test_list_papers_has_pagination_fields(self, client):
        response = client.get("/api/v1/papers")
        body = response.json()
        assert "papers" in body
        assert "total" in body
        assert "page" in body
        assert "page_size" in body

    def test_list_papers_pagination(self, client):
        response = client.get("/api/v1/papers?page=1&page_size=2")
        body = response.json()
        assert len(body["papers"]) <= 2
        assert body["page_size"] == 2
