"""
FastAPI 应用入口
注册所有路由、配置 CORS、异常处理和启动事件。
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

# ── 配置日志（在所有 import 之后，FastAPI 初始化之前）────────────────────────
setup_logging(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理。
    startup: 确保目录存在、尝试预加载模型
    shutdown: 清理资源（当前无特殊清理需求）
    """
    # ── 启动事件 ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  语音克隆服务 正在启动...")
    logger.info("=" * 60)

    # 记录启动时间（供健康检查计算 uptime）
    health_router.set_start_time(time.time())

    # 确保存储目录存在
    settings.ensure_directories()
    logger.info(f"存储目录已就绪: {Path(settings.storage_dir).absolute()}")

    # 设置 HuggingFace 镜像源环境变量
    import os
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint
    logger.info(f"HuggingFace 镜像源: {settings.hf_endpoint}")

    # 检查模型是否已下载
    model_path = Path(settings.model_dir) / "Fun-CosyVoice3-0.5B-2512"
    if model_path.exists():
        logger.info(f"检测到预训练模型: {model_path}")
        # 尝试预加载 TTS 引擎（后台异步，不阻塞服务启动）
        try:
            from app.core.tts_engine import TTSEngine
            engine = TTSEngine.get_instance(
                model_dir=settings.model_dir,
                storage_dir=settings.storage_dir,
            )
            loaded = engine.load_model()
            if loaded:
                logger.info("TTS 模型预加载成功")
            else:
                logger.warning("TTS 模型预加载失败（服务仍可启动，但合成功能不可用）")
        except Exception as e:
            logger.warning(f"TTS 模型预加载异常: {e}")
    else:
        logger.warning(
            f"未找到预训练模型目录: {model_path}。"
            "请运行 python setup/download_models.py 下载模型。"
        )

    logger.info(f"服务已启动 | 地址: http://{settings.host}:{settings.port}")
    logger.info(f"  - Web 界面:    http://{settings.host}:{settings.port}/")
    logger.info(f"  - API 文档:    http://{settings.host}:{settings.port}/docs")
    logger.info(f"  - 健康检查:    http://{settings.host}:{settings.port}/api/health")

    yield  # 此处之后为 shutdown 事件

    # ── 关闭事件 ─────────────────────────────────────────────────────────────
    logger.info("服务正在关闭...")


# ── 创建 FastAPI 应用 ────────────────────────────────────────────────────────
app = FastAPI(
    title="语音音色克隆服务",
    description=(
        "基于 CosyVoice3 的语音音色克隆训练服务。\n\n"
        "通过上传 3~10 段参考音频，提取说话人音色特征并打包为 .voicepack 文件。"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS 配置（开发阶段允许所有来源）────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 全局异常处理 ─────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    全局异常处理：捕获所有未处理的异常，返回友好的错误响应。

    Args:
        request: HTTP 请求对象
        exc: 捕获的异常

    Returns:
        JSONResponse: 统一格式的错误响应
    """
    logger.error(f"未处理的异常 | {request.method} {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc),
            "path": str(request.url),
        },
    )

# ── 注册 API 路由 ────────────────────────────────────────────────────────────
app.include_router(
    health_router.router,
    prefix="/api",
    tags=["健康检查"],
)
app.include_router(
    train_router.router,
    prefix="/api/train",
    tags=["训练管理"],
)
app.include_router(
    voices_router.router,
    prefix="/api/voices",
    tags=["音色管理"],
)

# ── 静态文件（Web 管理界面）──────────────────────────────────────────────────
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        """返回 Web 管理界面主页"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return JSONResponse(
            content={"message": "语音克隆服务运行中", "docs": "/docs"},
        )
else:
    @app.get("/", include_in_schema=False)
    async def root():
        return JSONResponse(
            content={
                "message": "语音克隆服务运行中",
                "docs": "/docs",
                "health": "/api/health",
            }
        )
