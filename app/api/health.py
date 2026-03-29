"""
Health check API router
Provides runtime information including service status, GPU info,
GPT-SoVITS availability, and trained voice count.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter

from app.models.schemas import GPUInfo, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Service start time (set in main.py)
_service_start_time: float = time.time()


def set_start_time(t: float) -> None:
    """Set the service start time (called by main.py in the startup event)."""
    global _service_start_time
    _service_start_time = t


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description=(
        "Returns service status, GPU info, GPT-SoVITS availability, "
        "and the number of available trained voices."
    ),
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse: Service status, GPU info, voice count, uptime, etc.
    """
    from app.config import settings

    # -- GPU information -------------------------------------------------------
    gpu_info: Optional[GPUInfo] = None
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_info = GPUInfo(
                name=props.name,
                compute_capability=f"{props.major}.{props.minor}",
                total_memory_gb=round(props.total_memory / (1024 ** 3), 2),
                cuda_available=True,
            )
        else:
            gpu_info = GPUInfo(
                name="No GPU",
                compute_capability="N/A",
                total_memory_gb=0.0,
                cuda_available=False,
            )
    except Exception as e:
        logger.warning(f"Failed to retrieve GPU info: {e}")

    # -- GPT-SoVITS readiness -------------------------------------------------
    from pathlib import Path
    gptsovits_ready = Path(settings.gptsovits_dir).exists()

    # -- Number of available voices -------------------------------------------
    try:
        from app.core.voice_manager import VoiceManager
        manager = VoiceManager(storage_dir=settings.storage_dir)
        voice_count = len(manager.list_all())
    except Exception:
        voice_count = 0

    # -- Uptime ---------------------------------------------------------------
    uptime = time.time() - _service_start_time

    # -- Service status determination -----------------------------------------
    if gptsovits_ready and (gpu_info is None or gpu_info.cuda_available):
        svc_status = "ok"
    elif not gptsovits_ready:
        svc_status = "degraded"  # GPT-SoVITS not cloned; training/synthesis unavailable
    else:
        svc_status = "ok"

    return HealthResponse(
        status=svc_status,
        version="2.0.0",
        gptsovits_ready=gptsovits_ready,
        gpu_info=gpu_info,
        voice_count=voice_count,
        uptime_seconds=round(uptime, 2),
    )
