"""
main.py — FastAPI application factory.

This file does three things only:
  1. Creates the FastAPI app with metadata
  2. Registers middleware (CORS, request ID, timing)
  3. Mounts routers

Business logic lives in api/, agent/, services/. Never here.

Entry point:
  uvicorn app.main:app --reload            # development
  uvicorn app.main:app --workers 4         # production
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.config import settings
from app.core.exceptions import RCABaseException
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Runs once at startup (before yield) and once at shutdown (after yield).

    Startup order matters:
      1. Logging first — every subsequent step can then log properly
      2. Validate config — fail fast before connecting to anything
      3. Connect to external services
      4. Warm up ML models (future)

    Shutdown order is the reverse: release resources cleanly.
    """
    # ── Startup ──────────────────────────────────────────────────────────────
    configure_logging(
        log_level=settings.log_level,
        as_json=settings.is_production,    # JSON in prod, pretty in dev
    )

    logger.info(
        "app.starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        debug=settings.debug,
    )

    # Validate configuration early — better to crash here than mid-request
    _validate_startup_config()

    logger.info("app.started", port=settings.app_port)

    yield   # ← application is running and serving requests

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("app.shutting_down")
    # Future: await db_pool.close(), await redis.aclose(), etc.
    logger.info("app.stopped")


def _validate_startup_config() -> None:
    """Raise immediately if required config is missing or invalid."""
    if settings.is_production and settings.debug:
        raise RuntimeError("DEBUG must be False in production")

    if settings.is_production and not settings.openai_api_key:
        if settings.llm_provider == "openai":
            raise RuntimeError("OPENAI_API_KEY is required in production")

    logger.info(
        "config.validated",
        llm_provider=settings.llm_provider,
        qdrant_host=settings.qdrant_host,
        redis_url=settings.redis_url.split("@")[-1],    # strip credentials
    )


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Factory function — returns a fully configured FastAPI instance.

    Using a factory (rather than a module-level app) means:
    - Tests can create fresh app instances with different settings
    - Easier to configure middleware conditionally per environment
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Agentic AI system that retrieves academic papers, extracts "
            "empirical claims, detects contradictions between papers, and "
            "generates evidence-grounded explanations of disagreements."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    _register_middleware(app)
    _register_routers(app)
    _register_exception_handlers(app)

    return app


# ── Middleware ────────────────────────────────────────────────────────────────

def _register_middleware(app: FastAPI) -> None:
    """
    Middleware executes in reverse registration order.
    Registered last → runs first on incoming requests.

    Order here (outermost to innermost):
      CORS → Request ID → Timing → route handler
    """

    # CORS — must be outermost so preflight OPTIONS requests are handled
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )

    # Request timing — measures total wall-clock time per request
    @app.middleware("http")
    async def timing_middleware(request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(duration_ms)

        logger.info(
            "http.request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        return response

    # Request ID — every request gets a unique ID for log correlation
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ── Routers ───────────────────────────────────────────────────────────────────

def _register_routers(app: FastAPI) -> None:
    """Mount all API routers under their versioned prefix."""
    app.include_router(v1_router, prefix="/api/v1")


# ── Exception handlers ────────────────────────────────────────────────────────

def _register_exception_handlers(app: FastAPI) -> None:
    """
    Convert exceptions into consistent JSON error responses.
    Clients always get: {"error": "...", "detail": "...", "request_id": "..."}
    """

    @app.exception_handler(RCABaseException)
    async def rca_exception_handler(
        request: Request, exc: RCABaseException
    ) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.warning(
            "app.handled_exception",
            exception_type=type(exc).__name__,
            detail=exc.detail,
            status_code=exc.status_code,
            request_id=request_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "detail": exc.detail,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Catch-all for unexpected exceptions.
        Logs the full traceback but returns a generic message to the client
        — never expose internal details in production.
        """
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.exception(
            "app.unhandled_exception",
            exception_type=type(exc).__name__,
            request_id=request_id,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": (
                    str(exc) if settings.is_development
                    else "An unexpected error occurred"
                ),
                "request_id": request_id,
            },
        )


# ── Module-level app instance ─────────────────────────────────────────────────

# uvicorn app.main:app  →  this is the object uvicorn imports
app: FastAPI = create_app()
