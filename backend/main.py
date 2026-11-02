"""EcoVision AI FastAPI application."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from backend.api.routes import (
    feedback,
    health,
    history,
    model_info,
    predict,
    statistics,
)
from backend.core.config import APP_VERSION, Settings
from backend.core.constants import API_PREFIX
from backend.core.errors import AppError, register_exception_handlers
from backend.core.logging import configure_logging, get_logger
from backend.database.database import Database
from backend.services.advice_service import AdviceService
from backend.services.artifact_service import fetch_model_artifacts
from backend.services.model_service import ModelService

ADVICE_PATH = Path(__file__).resolve().parent / "data" / "recycling_guide.json"
logger = get_logger("ecovision.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    database = Database(settings.db_path)
    database.migrate()
    app.state.database = database

    app.state.advice_service = AdviceService(ADVICE_PATH)

    await run_in_threadpool(fetch_model_artifacts, settings)

    model_service = ModelService(settings)
    await run_in_threadpool(model_service.load)
    app.state.model_service = model_service

    app.state.inference_semaphore = asyncio.Semaphore(
        max(1, settings.max_concurrent_inference)
    )
    app.state.started_at = time.monotonic()
    logger.info(
        "%s %s ready (model=%s, frontend=%s)",
        settings.app_name,
        APP_VERSION,
        model_service.loaded.stage if model_service.available else "unavailable",
        settings.serve_frontend,
    )
    try:
        yield
    finally:
        logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Waste image classification with Grad-CAM explanations, recycling "
            "guidance, feedback, and prediction history."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
        redoc_url=None,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        elapsed_ms = (time.perf_counter() - started) * 1000
        if request.url.path.startswith(API_PREFIX):
            logger.info(
                "%s %s -> %s (%.1f ms) [request_id=%s]",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
                request_id,
            )
        return response

    @app.middleware("http")
    async def asset_cache_headers(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if request.method == "GET" and (
            path == "/" or path.endswith((".html", ".js", ".css"))
        ):
            response.headers["Cache-Control"] = "no-cache"
        return response

    register_exception_handlers(app)

    for module in (health, predict, feedback, history, statistics, model_info):
        app.include_router(module.router, prefix=API_PREFIX)

    @app.api_route(
        f"{API_PREFIX}/{{path:path}}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    async def api_not_found(path: str) -> JSONResponse:
        raise AppError(404, "NOT_FOUND", "That API endpoint does not exist.")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/media",
        StaticFiles(directory=settings.upload_dir, check_dir=False),
        name="media",
    )

    if settings.serve_frontend:
        frontend_dir = settings.frontend_dir
        if (frontend_dir / "index.html").exists():
            app.mount(
                "/",
                StaticFiles(directory=frontend_dir, html=True),
                name="frontend",
            )
        else:
            logger.warning(
                "Frontend serving is enabled but %s/index.html is missing.",
                frontend_dir,
            )

    return app


app = create_app()
