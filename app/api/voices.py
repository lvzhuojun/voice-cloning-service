"""
音色管理 API 路由
提供音色包的列表查询、详情获取、下载、删除和测试合成接口。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import settings
from app.models.schemas import TestVoiceRequest, VoiceInfo, VoiceListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_manager():
    """获取 VoicePackManager 实例（懒加载）"""
    from app.core.voicepack_manager import VoicePackManager
    return VoicePackManager(storage_dir=settings.storage_dir)


def _get_engine():
    """获取 TTSEngine 单例（懒加载）"""
    from app.core.tts_engine import TTSEngine
    return TTSEngine.get_instance(
        model_dir=settings.model_dir,
        storage_dir=settings.storage_dir,
    )


@router.get(
    "/",
    response_model=VoiceListResponse,
    summary="列出所有音色",
    description="返回所有已训练音色的元数据列表（不含 embedding 向量），按创建时间降序排列。",
)
async def list_voices() -> VoiceListResponse:
    """
    获取所有音色包列表。

    Returns:
        VoiceListResponse: 包含音色总数和列表的响应
    """
    try:
        manager = _get_manager()
        voices = manager.list_all()
        return VoiceListResponse(total=len(voices), voices=voices)
    except Exception as e:
        logger.error(f"获取音色列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取音色列表失败: {e}",
        )


@router.get(
    "/{voice_id}",
    response_model=VoiceInfo,
    summary="获取音色详情",
    description="返回指定音色的完整元数据。",
)
async def get_voice(voice_id: str) -> VoiceInfo:
    """
    获取指定音色的详细信息。

    Args:
        voice_id: 音色的 UUID v4 字符串

    Returns:
        VoiceInfo: 音色详细信息

    Raises:
        HTTPException 404: 音色不存在
    """
    try:
        manager = _get_manager()
        return manager.get_info(voice_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"音色不存在: {voice_id}",
        )
    except Exception as e:
        logger.error(f"获取音色详情失败 ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取音色详情失败: {e}",
        )


@router.get(
    "/{voice_id}/download",
    summary="下载音色包",
    description="下载指定音色的 .voicepack 文件。",
    response_class=FileResponse,
)
async def download_voice(voice_id: str):
    """
    下载 .voicepack 文件。

    Args:
        voice_id: 音色的 UUID v4 字符串

    Returns:
        FileResponse: .voicepack 文件下载响应

    Raises:
        HTTPException 404: 音色包文件不存在
    """
    manager = _get_manager()
    voicepack_path = manager.get_voicepack_path(voice_id)

    import os
    if not os.path.exists(voicepack_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"音色包文件不存在: {voice_id}",
        )

    return FileResponse(
        path=voicepack_path,
        media_type="application/zip",
        filename=f"{voice_id}.voicepack",
        headers={"Content-Disposition": f'attachment; filename="{voice_id}.voicepack"'},
    )


@router.delete(
    "/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除音色",
    description="永久删除指定音色包文件。此操作不可撤销。",
)
async def delete_voice(voice_id: str) -> None:
    """
    删除指定音色包。

    Args:
        voice_id: 音色的 UUID v4 字符串

    Raises:
        HTTPException 404: 音色不存在
        HTTPException 500: 删除失败
    """
    try:
        manager = _get_manager()
        deleted = manager.delete(voice_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"音色不存在: {voice_id}",
            )
        logger.info(f"音色已删除: {voice_id}")
    except HTTPException:
        raise
    except OSError as e:
        logger.error(f"删除音色失败 ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除音色失败: {e}",
        )


@router.post(
    "/{voice_id}/test",
    summary="测试音色合成",
    description="使用指定音色合成一段测试语音，返回 WAV 音频文件。",
)
async def test_voice(voice_id: str, request: TestVoiceRequest) -> Response:
    """
    使用指定音色合成测试语音。

    Args:
        voice_id: 音色的 UUID v4 字符串
        request: 合成参数（文字、语速、语言）

    Returns:
        Response: WAV 格式音频数据（Content-Type: audio/wav）

    Raises:
        HTTPException 404: 音色不存在
        HTTPException 503: 模型未加载
        HTTPException 500: 合成失败
    """
    # 检查音色是否存在
    try:
        manager = _get_manager()
        manager.get_info(voice_id)  # 若不存在会抛出 FileNotFoundError
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"音色不存在: {voice_id}",
        )

    # 执行合成
    try:
        engine = _get_engine()
        wav_bytes = engine.synthesize(
            text=request.text,
            voice_id=voice_id,
            speed=request.speed,
            language=request.language.value,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="test_{voice_id}.wav"',
            },
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        error_msg = str(e)
        if "模型未加载" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS 模型未加载，请先运行 setup/download_models.py 下载模型并重启服务",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"语音合成失败: {error_msg}",
        )
    except Exception as e:
        logger.error(f"测试合成失败 ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"语音合成失败: {e}",
        )
