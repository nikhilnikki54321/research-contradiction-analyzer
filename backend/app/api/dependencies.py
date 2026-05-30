"""
api/dependencies.py — FastAPI dependency injection container.

All reusable Depends() objects live here. Route handlers import from
this file — never instantiate services or clients directly inside routes.

Why dependency injection?
  - Routes stay thin and readable
  - Services are swapped in tests by overriding the dependency
  - Connections are created once (on startup) and reused across requests
  - FastAPI handles cleanup via generator dependencies (yield pattern)

Usage in route handlers:
    @router.post("/papers")
    async def upload(settings: SettingsDep, upload_dir: UploadDirDep):
        ...
"""

from pathlib import Path
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Settings dependency ───────────────────────────────────────────────────────


def get_app_settings() -> Settings:
    """
    Provides the cached Settings singleton.
    In tests, override with:
        app.dependency_overrides[get_app_settings] = lambda: test_settings
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


# ── Upload directory dependency ───────────────────────────────────────────────


def get_upload_dir(settings: SettingsDep) -> Path:
    """
    Returns the upload directory as a Path, creating it if needed.
    Route handlers receive a ready-to-use Path object — no mkdir calls inside.

    The directory is resolved relative to the working directory
    (where uvicorn is launched from, i.e. backend/).
    """
    upload_path = Path(settings.upload_dir).resolve()

    if not upload_path.exists():
        upload_path.mkdir(parents=True, exist_ok=True)
        logger.info("uploads.dir_created", path=str(upload_path))

    return upload_path


UploadDirDep = Annotated[Path, Depends(get_upload_dir)]
