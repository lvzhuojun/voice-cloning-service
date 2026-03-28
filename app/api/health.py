"""
健康检查 API 路由
提供服务状态、GPU 信息、模型加载情况等运行时信息。
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter

from app.models.schemas import GPUInfo, HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# 服务启动时间（在 main.py 中设置）
_service_start_time: float = time.time()


def set_start_time(t: float) -> None:
    """设置服务启动时间（由 main.py 在 startup 事件中调用）"""
    global _service_start_time
    _service_start_time = t


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="服务健康检查",
    description="返回服务状态、GPU 信息、已加载模型和已有音色数量",
)
async def health_check() -> HealthResponse:
    """
    健康检查端点。

    Returns:
        HealthResponse: 包含服务状态、GPU 信息、音色数量等
    """
    from app.config import settings
    from app.core.tts_engine import TTSEngine
    from app.core.voicepack_manager import VoicePackManager

    # ── GPU 信息 ─────────────────────────────────────────────────────────────
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
                name="无 GPU",
                compute_capability="N/A",
                total_memory_gb=0.0,
                cuda_available=False,
            )
    except Exception as e:
        logger.warning(f"获取 GPU 信息失败: {e}")

    # ── 模型加载状态 ──────────────────────────────────────────────────────────
    engine = TTSEngine.get_instance(
        model_dir=settings.model_dir,
        storage_dir=settings.storage_dir,
    )
    model_loaded = engine.is_model_loaded()

    # ── 已有音色数量 ──────────────────────────────────────────────────────────
    try:
        manager = VoicePackManager(storage_dir=settings.storage_dir)
        voice_count = len(manager.list_all())
    except Exception:
        voice_count = 0

    # ── 运行时长 ──────────────────────────────────────────────────────────────
    uptime = time.time() - _service_start_time

    # ── 服务状态判断 ──────────────────────────────────────────────────────────
    if model_loaded and (gpu_info is None or gpu_info.cuda_available):
        status = "ok"
    elif not model_loaded:
        status = "degraded"  # 模型未加载，功能受限
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        version="1.0.0",
        model_loaded=model_loaded,
        gpu_info=gpu_info,
        voice_count=voice_count,
        uptime_seconds=round(uptime, 2),
    )
