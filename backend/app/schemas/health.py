from enum import Enum

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class DependencyCheck(BaseModel):
    name: str
    status: ServiceStatus
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str = Field(description="Application version from config")
    environment: str = Field(description="deployment environment")
    uptime_seconds: float = Field(description="seconds since process started")
    dependencies: list[DependencyCheck] = Field(
        default_factory=list,
        description="Populated only on /ready — empty on /health"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "uptime_seconds": 42.3,
                "dependencies": []
            }
        }
    }