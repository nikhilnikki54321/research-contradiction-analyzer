"""
api/v1/health.py — Health and readiness endpoints.

Two distinct endpoints — this distinction matters in production:

  GET /api/v1/health  →  liveness probe
      "Is the process alive?" — Kubernetes restarts the pod if this fails.
      Must NEVER depend on external services. Only checks the process itself.

  GET /api/v1/ready   →  readiness probe
      "Can the process serve traffic?" — Load balancer removes the pod if this
      fails. Checks every external dependency the app needs to function.

Both return the same HealthResponse shape so callers handle them identically.
"""

import time
from app.schemas.health import (
    DependencyCheck,
    HealthResponse,
    ServiceStatus,
)

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])

# Module-level startup timestamp — never changes after import
_START_TIME: float = time.monotonic()


# ── Dependency checkers ───────────────────────────────────────────────────────

async def _check_qdrant() -> DependencyCheck:
    """Ping Qdrant and measure round-trip latency."""
    start = time.monotonic()
    try:
        from qdrant_client import AsyncQdrantClient  # type: ignore
        client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            timeout=5,
        )
        await client.get_collections()
        await client.close()
        return DependencyCheck(
            name="qdrant",
            status=ServiceStatus.OK,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.warning("health.qdrant_down", error=str(exc))
        return DependencyCheck(
            name="qdrant",
            status=ServiceStatus.DOWN,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            detail=str(exc),
        )


async def _check_redis() -> DependencyCheck:
    """Ping Redis and measure round-trip latency."""
    start = time.monotonic()
    try:
        import redis.asyncio as aioredis  # type: ignore
        client = aioredis.from_url(settings.redis_url, socket_timeout=5)
        await client.ping()
        await client.aclose()
        return DependencyCheck(
            name="redis",
            status=ServiceStatus.OK,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.warning("health.redis_down", error=str(exc))
        return DependencyCheck(
            name="redis",
            status=ServiceStatus.DOWN,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            detail=str(exc),
        )


async def _check_postgres() -> DependencyCheck:
    """Run a trivial query against Postgres and measure latency."""
    start = time.monotonic()
    try:
        import asyncpg  # type: ignore
        conn = await asyncpg.connect(
            settings.database_url.replace("+asyncpg", ""),
            timeout=5,
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return DependencyCheck(
            name="postgres",
            status=ServiceStatus.OK,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
        )
    except Exception as exc:
        logger.warning("health.postgres_down", error=str(exc))
        return DependencyCheck(
            name="postgres",
            status=ServiceStatus.DOWN,
            latency_ms=round((time.monotonic() - start) * 1000, 2),
            detail=str(exc),
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 if the process is alive. "
        "Does NOT check external dependencies. "
        "Use this for Kubernetes liveness probes."
    ),
)
async def health() -> HealthResponse:
    """
    Liveness check — always fast, never fails due to external services.
    If this endpoint is unreachable, the process itself has crashed.
    """
    return HealthResponse(
        status=ServiceStatus.OK,
        version=settings.app_version,
        environment=settings.app_env,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    description=(
        "Returns 200 only when all dependencies are reachable. "
        "Returns 503 if any critical dependency is down. "
        "Use this for Kubernetes readiness probes and load balancer checks."
    ),
    responses={
        503: {
            "description": "One or more dependencies are unavailable",
            "model": HealthResponse,
        }
    },
)
async def ready() -> HealthResponse:
    """
    Readiness check — verifies every external dependency in parallel.
    Returns 503 if any dependency is DOWN so load balancers stop routing here.
    DEGRADED dependencies log a warning but still return 200.
    """
    import asyncio

    checks: list[DependencyCheck] = await asyncio.gather(
        _check_redis(),
        _check_qdrant(),
        _check_postgres(),
    )

    # Determine overall status
    statuses = {c.status for c in checks}
    if ServiceStatus.DOWN in statuses:
        overall = ServiceStatus.DOWN
    elif ServiceStatus.DEGRADED in statuses:
        overall = ServiceStatus.DEGRADED
    else:
        overall = ServiceStatus.OK

    response = HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.app_env,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
        dependencies=checks,
    )

    logger.info(
        "health.ready_check",
        overall_status=overall,
        dependency_statuses={c.name: c.status for c in checks},
    )

    if overall == ServiceStatus.DOWN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response
