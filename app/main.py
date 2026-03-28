"""
FastAPI application entry point
Registers all routes, configures CORS, exception handling, and startup events.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import health as health_router
from app.api import train as train_router
from app.api import voices as voices_router
from app.config import settings
from app.utils.logger import setup_logging

# ── Configure logging (after all imports, before FastAPI initialization) ──────
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle management.
    startup: ensure directories exist, attempt to preload the model
    shutdown: clean up resources (no special cleanup needed currently)
    """
    # ── Startup event ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Voice Cloning Service is starting...")
    logger.info("=" * 60)

    # Record startup time (for uptime calculation in health check)
    health_router.set_start_time(time.time())

    # Ensure storage directories exist
    settings.ensure_directories()
    logger.info(f"Storage directories ready: {Path(settings.storage_dir).absolute()}")

    # Set HuggingFace mirror environment variable
    import os
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    logger.info(f"HuggingFace mirror endpoint: {settings.hf_endpoint}")

    # Check whether the model has been downloaded
    model_path = Path(settings.model_dir) / "Fun-CosyVoice3-0.5B-2512"
    if model_path.exists():
        logger.info(f"Pretrained model detected: {model_path}")
        # Attempt to preload the TTS engine (non-blocking; service starts regardless)
        try:
            from app.core.tts_engine import TTSEngine
            engine = TTSEngine.get_instance(
                model_dir=settings.model_dir,
                storage_dir=settings.storage_dir,
            )
            loaded = engine.load_model()
            if loaded:
                logger.info("TTS model preloaded successfully")
            else:
                logger.warning("TTS model preloading failed (service will still start, but synthesis is unavailable)")
        except Exception as e:
            logger.warning(f"TTS model preloading exception: {e}")
    else:
        logger.warning(
            f"Pretrained model directory not found: {model_path}. "
            "Please run python setup/download_models.py to download the model."
        )

    logger.info(f"Service started | Address: http://{settings.host}:{settings.port}")
    logger.info(f"  - Web UI:       http://{settings.host}:{settings.port}/")
    logger.info(f"  - API docs:     http://{settings.host}:{settings.port}/docs")
    logger.info(f"  - Health check: http://{settings.host}:{settings.port}/api/health")

    yield  # Everything after this point is the shutdown event

    # ── Shutdown event ────────────────────────────────────────────────────────
    logger.info("Service is shutting down...")


# ── Create FastAPI application ────────────────────────────────────────────────
app = FastAPI(
    title="Voice Cloning Service",
    description=(
        "A voice cloning training service based on CosyVoice3.\n\n"
        "Upload 3~10 reference audio clips to extract speaker voice features "
        "and package them into a .voicepack file."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS configuration (allow all origins during development) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception handler ──────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler: catches all unhandled exceptions and returns
    a friendly error response.

    Args:
        request: HTTP request object
        exc: Caught exception

    Returns:
        JSONResponse: Uniformly formatted error response
    """
    logger.error(f"Unhandled exception | {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )

# ── Register API routes ───────────────────────────────────────────────────────
app.include_router(
    health_router.router,
    prefix="/api",
    tags=["Health Check"],
)
app.include_router(
    train_router.router,
    prefix="/api/train",
    tags=["Training Management"],
)
app.include_router(
    voices_router.router,
    prefix="/api/voices",
    tags=["Voice Management"],
)

# ── Static files (Web management UI) ─────────────────────────────────────────
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Return the Web management UI home page"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse(
            content={"message": "Voice cloning service is running", "docs": "/docs"},
        )
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "message": "Voice cloning service is running",
                "docs": "/docs",
                "health": "/api/health",
            }
        )
