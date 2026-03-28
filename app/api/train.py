"""
训练相关 API 路由
提供从上传文件或本地文件夹创建音色克隆训练任务的接口。
训练任务异步执行，通过 task_id 查询进度。
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.models.schemas import (
    Language,
    TaskStatus,
    TaskStatusEnum,
    TrainFromFolderRequest,
    TrainResponse,
)
from app.utils.file_utils import (
    cleanup_dir,
    create_temp_dir,
    get_audio_files_in_dir,
    is_allowed_audio_file,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ── 内存中的任务状态存储 ───────────────────────────────────────────────────────
# 生产环境应替换为 Redis 或数据库，此处使用字典简化实现
_task_store: Dict[str, TaskStatus] = {}


def _update_task(
    task_id: str,
    status: TaskStatusEnum,
    progress: int,
    message: str,
    voice_id: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """
    更新任务状态（内部辅助函数）。

    Args:
        task_id: 任务 ID
        status: 新状态
        progress: 进度（0~100）
        message: 阶段描述
        voice_id: 完成后的音色 ID（可选）
        error: 错误信息（可选）
    """
    if task_id in _task_store:
        task = _task_store[task_id]
        task.status = status
        task.progress = progress
        task.message = message
        task.voice_id = voice_id
        task.error = error
        task.updated_at = datetime.now(timezone.utc)


def _run_training_pipeline(
    task_id: str,
    audio_files: List[str],
    voice_name: str,
    language: str,
    temp_dir: Optional[str] = None,
) -> None:
    """
    训练流水线主函数（在后台任务中执行）。

    流程：
    1. 批量预处理音频（格式转换 + 质量检测 + 降噪）
    2. 提取并融合 speaker embedding
    3. 选取最优参考音频
    4. 打包为 .voicepack 文件

    Args:
        task_id: 任务 ID
        audio_files: 音频文件路径列表
        voice_name: 音色名称
        language: 语言代码
        temp_dir: 临时目录路径（处理完成后会被清理）
    """
    try:
        from app.core.audio_processor import AudioProcessor
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.voicepack_manager import VoicePackManager

        # ── 阶段1：音频预处理 ────────────────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 10, "音频预处理中...")
        logger.info(f"[{task_id}] 开始音频预处理，文件数: {len(audio_files)}")

        processor = AudioProcessor(
            min_duration=settings.min_audio_duration_seconds,
            max_duration=settings.max_audio_duration_seconds,
        )

        processed_audio, batch_report = processor.process_batch(audio_files)

        if batch_report.success_count == 0:
            raise RuntimeError(
                f"所有音频处理失败: {list(batch_report.errors.values())[:3]}"
            )

        logger.info(
            f"[{task_id}] 预处理完成 | "
            f"成功: {batch_report.success_count}/{batch_report.total_files}"
        )

        # ── 阶段2：提取 Speaker Embedding ────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 40, "提取说话人特征...")
        logger.info(f"[{task_id}] 开始提取 speaker embedding")

        extractor = EmbeddingExtractor(
            model_dir=settings.model_dir,
        )

        # 构建 (audio, sr, quality_report) 三元组列表
        audio_quality_pairs = [
            (audio, sr, batch_report.file_reports[fname])
            for fname, (audio, sr) in processed_audio.items()
            if fname in batch_report.file_reports
        ]

        fused_embedding, weight_records = extractor.extract_and_fuse(audio_quality_pairs)
        logger.info(f"[{task_id}] Embedding 融合完成 | 维度: {fused_embedding.shape[0]}")

        # ── 阶段3：选取最优参考音频 ───────────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 70, "选取最优参考音频...")

        best_audio, best_sr, best_report = extractor.select_reference_audio(audio_quality_pairs)
        logger.info(
            f"[{task_id}] 参考音频选取完成 | "
            f"时长: {best_report.duration_seconds:.1f}s | 质量: {best_report.quality_score:.3f}"
        )

        # ── 阶段4：打包 .voicepack ────────────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 85, "打包音色包...")

        manager = VoicePackManager(storage_dir=settings.storage_dir)
        voice_id = manager.pack(
            embedding=fused_embedding,
            reference_audio=best_audio,
            reference_sample_rate=best_sr,
            voice_name=voice_name,
            language=language,
            sample_count=batch_report.success_count,
            total_duration_seconds=batch_report.total_duration_seconds,
            quality_score=batch_report.average_quality_score,
        )

        logger.info(f"[{task_id}] 音色包打包完成 | voice_id: {voice_id}")

        # ── 完成 ─────────────────────────────────────────────────────────────
        _update_task(
            task_id,
            TaskStatusEnum.DONE,
            100,
            f"训练完成！音色 ID: {voice_id}",
            voice_id=voice_id,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{task_id}] 训练失败: {error_msg}")
        _update_task(
            task_id,
            TaskStatusEnum.FAILED,
            0,
            "训练失败",
            error=error_msg,
        )
    finally:
        # 清理临时文件
        if temp_dir:
            cleanup_dir(temp_dir)


# ══════════════════════════════════════════════════════════════════════════════
# API 路由
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/from-upload",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传音频文件创建训练任务",
    description="上传 1~10 个音频文件，异步创建音色训练任务。返回 task_id 用于查询进度。",
)
async def train_from_upload(
    background_tasks: BackgroundTasks,
    voice_name: str = Form(..., description="音色名称（1~64字符）", min_length=1, max_length=64),
    language: Language = Form(default=Language.ZH, description="主要语言"),
    files: List[UploadFile] = File(..., description="参考音频文件（支持 WAV/MP3/M4A/FLAC/OGG）"),
) -> TrainResponse:
    """
    从上传的音频文件创建训练任务。

    Args:
        background_tasks: FastAPI 后台任务管理器
        voice_name: 用户自定义音色名称
        language: 主要语言
        files: 上传的音频文件列表（1~10个）

    Returns:
        TrainResponse: 包含 task_id 的响应

    Raises:
        HTTPException 400: 文件格式不支持 / 文件数量不合法
        HTTPException 413: 文件过大
    """
    # 校验文件数量
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="至少需要上传 1 个音频文件",
        )
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"最多支持 10 个音频文件，当前上传 {len(files)} 个",
        )

    # 校验文件格式
    for f in files:
        if not is_allowed_audio_file(f.filename or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式: {f.filename}。支持格式: WAV/MP3/M4A/FLAC/OGG",
            )

    # 创建临时目录保存上传文件
    temp_dir = create_temp_dir(prefix=f"upload_{uuid.uuid4().hex[:8]}_")
    saved_paths: List[str] = []

    try:
        for upload_file in files:
            file_content = await upload_file.read()

            # 检查文件大小
            if len(file_content) > settings.max_upload_size_bytes:
                cleanup_dir(temp_dir)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"文件 {upload_file.filename} 超过大小限制 "
                        f"({settings.max_upload_size_mb} MB)"
                    ),
                )

            # 保存到临时目录
            safe_name = Path(upload_file.filename or "audio.wav").name
            save_path = os.path.join(temp_dir, safe_name)
            with open(save_path, "wb") as f:
                f.write(file_content)
            saved_paths.append(save_path)

    except HTTPException:
        raise
    except Exception as e:
        cleanup_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件保存失败: {e}",
        )

    # 创建任务记录
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _task_store[task_id] = TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        progress=0,
        message=f"任务已创建，等待处理 ({len(saved_paths)} 个文件)",
        created_at=now,
        updated_at=now,
    )

    # 添加后台任务
    background_tasks.add_task(
        _run_training_pipeline,
        task_id=task_id,
        audio_files=saved_paths,
        voice_name=voice_name,
        language=language.value,
        temp_dir=temp_dir,
    )

    logger.info(
        f"训练任务已创建 | task_id: {task_id} | "
        f"文件数: {len(saved_paths)} | 音色名: {voice_name}"
    )

    return TrainResponse(
        task_id=task_id,
        message=f"训练任务已提交，共 {len(saved_paths)} 个音频文件。使用 task_id 查询进度。",
        status=TaskStatusEnum.PENDING,
    )


@router.post(
    "/from-folder",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="从本地文件夹创建训练任务",
    description="指定 data/samples/ 下的子文件夹名，使用其中的音频文件创建训练任务。",
)
async def train_from_folder(
    background_tasks: BackgroundTasks,
    request: TrainFromFolderRequest,
) -> TrainResponse:
    """
    从本地文件夹创建训练任务。

    Args:
        background_tasks: FastAPI 后台任务管理器
        request: 包含 folder_name、voice_name、language 的请求体

    Returns:
        TrainResponse: 包含 task_id 的响应

    Raises:
        HTTPException 404: 文件夹不存在
        HTTPException 400: 文件夹为空或无有效音频
    """
    folder_path = os.path.join(settings.samples_dir, request.folder_name)

    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"文件夹不存在: data/samples/{request.folder_name}。"
                "请先在该目录下创建子文件夹并放入音频文件。"
            ),
        )

    try:
        audio_files = get_audio_files_in_dir(folder_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"扫描文件夹失败: {e}",
        )

    if not audio_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"文件夹 data/samples/{request.folder_name} 中没有找到有效的音频文件。"
                "支持格式: WAV/MP3/M4A/FLAC/OGG"
            ),
        )

    if len(audio_files) > 10:
        audio_files = audio_files[:10]
        logger.info(f"文件夹中文件超过10个，仅使用前10个")

    # 创建任务记录
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _task_store[task_id] = TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        progress=0,
        message=f"任务已创建，使用文件夹: {request.folder_name} ({len(audio_files)} 个文件)",
        created_at=now,
        updated_at=now,
    )

    # 添加后台任务
    background_tasks.add_task(
        _run_training_pipeline,
        task_id=task_id,
        audio_files=audio_files,
        voice_name=request.voice_name,
        language=request.language.value,
        temp_dir=None,  # 不需要清理（使用原始文件路径）
    )

    logger.info(
        f"训练任务已创建（文件夹模式）| task_id: {task_id} | "
        f"文件夹: {request.folder_name} | 文件数: {len(audio_files)}"
    )

    return TrainResponse(
        task_id=task_id,
        message=(
            f"训练任务已提交，使用文件夹 data/samples/{request.folder_name} 中的 "
            f"{len(audio_files)} 个音频文件。"
        ),
        status=TaskStatusEnum.PENDING,
    )


@router.get(
    "/{task_id}/status",
    response_model=TaskStatus,
    summary="查询训练任务状态",
    description="通过 task_id 查询训练任务的当前状态和进度。",
)
async def get_task_status(task_id: str) -> TaskStatus:
    """
    查询训练任务状态。

    Args:
        task_id: 训练任务的唯一 ID

    Returns:
        TaskStatus: 任务详细状态

    Raises:
        HTTPException 404: task_id 不存在
    """
    if task_id not in _task_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务不存在: {task_id}",
        )
    return _task_store[task_id]
