"""
FastAPI application entry point
Registers all routes, configures CORS, exception handling, and startup events.
GPT-SoVITS based voice cloning service.
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

# -- Configure logging ---------------------------------------------------------
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI application lifecycle management.
    startup: ensure directories exist, check GPT-SoVITS availability
    shutdown: clean up resources
    """
    # -- Startup ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("  Voice Cloning Service (GPT-SoVITS) is starting...")
    logger.info("=" * 60)

    # Record startup time
    health_router.set_start_time(time.time())

    # Ensure storage directories exist
    settings.ensure_directories()
    logger.info(f"Storage directories ready: {Path(settings.storage_dir).absolute()}")

    # Set HuggingFace mirror
    import os
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    logger.info(f"HuggingFace endpoint: {settings.hf_endpoint}")

    # Check GPT-SoVITS availability
    gptsovits_path = Path(settings.gptsovits_dir)
    if gptsovits_path.exists():
        logger.info(f"GPT-SoVITS directory found: {gptsovits_path.absolute()}")
    else:
        logger.warning(
            f"GPT-SoVITS directory not found: {gptsovits_path.absolute()}\n"
            "Training and synthesis will be unavailable until GPT-SoVITS is cloned.\n"
            "Run: bash setup/clone_gptsovits.sh"
        )

    # Check pretrained models
    pretrained_path = Path(settings.pretrained_gptsovits_dir)
    required_pretrained = [
        pretrained_path / "pretrained_s1.ckpt",
        pretrained_path / "pretrained_s2G.pth",
        pretrained_path / "pretrained_s2D.pth",
        pretrained_path / "chinese-hubert-base" / "config.json",
        pretrained_path / "chinese-roberta-wwm-ext-large" / "config.json",
    ]
    missing_pretrained = [str(path) for path in required_pretrained if not path.exists()]
    if not missing_pretrained:
        logger.info(f"GPT-SoVITS pretrained models verified: {pretrained_path.absolute()}")
    else:
        logger.warning(
            "GPT-SoVITS pretrained models are incomplete:\n"
            + "\n".join(missing_pretrained)
            + "\nRun: python setup/download_models.py"
        )

    # Log available voices
    try:
        from app.core.voice_manager import VoiceManager
        manager = VoiceManager(storage_dir=settings.storage_dir)
        voice_count = len(manager.list_all())
        logger.info(f"Available trained voices: {voice_count}")
    except Exception:
        pass

    logger.info(f"Service started | http://{settings.host}:{settings.port}")
    logger.info(f"  Web UI:       http://{settings.host}:{settings.port}/")
    logger.info(f"  API docs:     http://{settings.host}:{settings.port}/docs")
    logger.info(f"  Health check: http://{settings.host}:{settings.port}/api/health")

    yield  # Shutdown follows

    # -- Shutdown --------------------------------------------------------------
    logger.info("Service is shutting down...")


# -- Create FastAPI application ------------------------------------------------
app = FastAPI(
    title="Voice Cloning Service",
    description=(
        "A voice cloning service based on GPT-SoVITS real fine-tuning.\n\n"
        "Upload 1-3 minutes of reference audio to fine-tune a personal voice model. "
        "Synthesize speech with the trained model via the /api/voices/{voice_id}/test endpoint."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -- CORS configuration --------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Global exception handler --------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch all unhandled exceptions and return a friendly error response."""
    logger.error(f"Unhandled exception | {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# -- Register API routes -------------------------------------------------------
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

# -- Static files (Web management UI) ------------------------------------------
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """Return the Web management UI home page."""
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
                "message": "Voice cloning service (GPT-SoVITS) is running",
                "docs": "/docs",
                "health": "/api/health",
            }
        )
