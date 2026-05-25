"""
api/v1/router.py — Root router for API version 1.

Every sub-router is mounted here with its prefix and tags.
main.py imports only this file — never individual sub-routers directly.

To add a new feature (e.g. /users):
  1. Create app/api/v1/users.py with a router
  2. Import and include it here — one line
  3. main.py needs zero changes
"""
"""
To add a new feature:
  1. Create a router file
  2. Import it here
  3. Include it below
"""

from fastapi import APIRouter

from app.api.v1 import health
from app.api.v1.upload import router as upload_router

# All routes here are automatically prefixed with /api/v1 in main.py
v1_router = APIRouter()

# Health endpoints
v1_router.include_router(health.router)

# Upload endpoints
v1_router.include_router(upload_router)

# Future routers are added here as the project grows:
# from app.api.v1 import papers, analysis, stream
# v1_router.include_router(papers.router, prefix="/papers", tags=["papers"])
# v1_router.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
# v1_router.include_router(stream.router, prefix="/stream", tags=["stream"])
