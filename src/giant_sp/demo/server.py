"""FastAPI app serving the demo site and the data dashboard's API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from giant_sp.demo.dashboard import Dashboard

logger = logging.getLogger(__name__)

# The static site lives at the repo root, next to `src/`.
DEFAULT_SITE_DIR = Path(__file__).resolve().parents[3] / "demo"

# The dashboard page (`<site>/dashboard/index.html`) fetches `./api/...`
# relative to itself, so its API lives under the same prefix.
API_PREFIX = "/dashboard/api"


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that makes browsers revalidate (cheap ETag 304s) on reload."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        """Add ``Cache-Control: no-cache`` to every static response."""
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def _api_router(dashboard: Dashboard) -> APIRouter:
    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/repo")
    def repo() -> dict[str, Any]:
        source = dashboard.source
        snapshot = dashboard.snapshot()
        return {
            "repo": source.repo,
            "branch": source.branch,
            "root_path": source.root_path,
            "snapshot": {
                "id": snapshot.id,
                "written_at": snapshot.written_at,
                "message": snapshot.message,
            },
        }

    @router.get("/catalog")
    def catalog() -> dict[str, Any]:
        return dashboard.catalog()

    @router.get("/series")
    def series(experiment: str, variable: str) -> dict[str, Any]:
        try:
            return dashboard.series(experiment, variable)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @router.get("/map")
    def map_png(experiment: str, variable: str, date: str) -> Response:
        try:
            png = dashboard.map_png(experiment, variable, date)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return Response(content=png, media_type="image/png")

    return router


def create_app(
    dashboard: Dashboard, site_dir: Path = DEFAULT_SITE_DIR, warm_up: bool = True
) -> FastAPI:
    """Build the demo app.

    Parameters
    ----------
    dashboard : Dashboard
        Backs the ``/dashboard/api`` endpoints.
    site_dir : Path
        Directory of static pages served at ``/``.
    warm_up : bool
        Whether to build the dashboard catalog on startup, so the first page
        load doesn't pay for discovering it.

    Returns
    -------
    FastAPI
        The app.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if warm_up:
            try:
                dashboard.catalog()
                logger.info("dashboard catalog ready")
            except Exception as e:  # best effort; requests retry on their own
                logger.warning("dashboard warm-up failed (continuing): %s", e)
        yield

    app = FastAPI(title="GIANT sensor placement demo", lifespan=lifespan)
    app.include_router(_api_router(dashboard))
    app.mount("/", _NoCacheStaticFiles(directory=site_dir, html=True), name="site")
    return app
