"""
Voice management API router
Provides endpoints for listing, retrieving, downloading, deleting, and test-synthesizing voice packs.
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
    """Get a VoicePackManager instance (lazy loading)"""
    from app.core.voicepack_manager import VoicePackManager
    return VoicePackManager(storage_dir=settings.storage_dir)


def _get_engine():
    """Get the TTSEngine singleton (lazy loading)"""
    from app.core.tts_engine import TTSEngine
    return TTSEngine.get_instance(
        model_dir=settings.model_dir,
        storage_dir=settings.storage_dir,
    )


@router.get(
    "/",
    response_model=VoiceListResponse,
    summary="List all voices",
    description="Returns a metadata list of all trained voices (without embedding vectors), sorted by creation time (newest first).",
)
async def list_voices() -> VoiceListResponse:
    """
    Get a list of all voice packs.

    Returns:
        VoiceListResponse: Response containing total voice count and list
    """
    try:
        manager = _get_manager()
        voices = manager.list_all()
        return VoiceListResponse(total=len(voices), voices=voices)
    except Exception as e:
        logger.error(f"Failed to retrieve voice list: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve voice list: {e}",
        )


@router.get(
    "/{voice_id}",
    response_model=VoiceInfo,
    summary="Get voice details",
    description="Returns the complete metadata for the specified voice.",
)
async def get_voice(voice_id: str) -> VoiceInfo:
    """
    Get detailed information for the specified voice.

    Args:
        voice_id: Voice UUID v4 string

    Returns:
        VoiceInfo: Detailed voice information

    Raises:
        HTTPException 404: Voice does not exist
    """
    try:
        manager = _get_manager()
        return manager.get_info(voice_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice does not exist: {voice_id}",
        )
    except Exception as e:
        logger.error(f"Failed to retrieve voice details ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve voice details: {e}",
        )


@router.get(
    "/{voice_id}/download",
    summary="Download voice pack",
    description="Download the .voicepack file for the specified voice.",
    response_class=FileResponse,
)
async def download_voice(voice_id: str):
    """
    Download a .voicepack file.

    Args:
        voice_id: Voice UUID v4 string

    Returns:
        FileResponse: .voicepack file download response

    Raises:
        HTTPException 404: Voice pack file does not exist
    """
    manager = _get_manager()
    voicepack_path = manager.get_voicepack_path(voice_id)

    import os
    if not os.path.exists(voicepack_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice pack file does not exist: {voice_id}",
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
    summary="Delete voice",
    description="Permanently delete the specified voice pack file. This action cannot be undone.",
)
async def delete_voice(voice_id: str) -> None:
    """
    Delete the specified voice pack.

    Args:
        voice_id: Voice UUID v4 string

    Raises:
        HTTPException 404: Voice does not exist
        HTTPException 500: Deletion failed
    """
    try:
        manager = _get_manager()
        deleted = manager.delete(voice_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Voice does not exist: {voice_id}",
            )
        logger.info(f"Voice deleted: {voice_id}")
    except HTTPException:
        raise
    except OSError as e:
        logger.error(f"Failed to delete voice ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {e}",
        )


@router.post(
    "/{voice_id}/test",
    summary="Test voice synthesis",
    description="Synthesize a test speech using the specified voice and return a WAV audio file.",
)
async def test_voice(voice_id: str, request: TestVoiceRequest) -> Response:
    """
    Synthesize test speech using the specified voice.

    Args:
        voice_id: Voice UUID v4 string
        request: Synthesis parameters (text, speed, language)

    Returns:
        Response: WAV audio data (Content-Type: audio/wav)

    Raises:
        HTTPException 404: Voice does not exist
        HTTPException 503: Model not loaded
        HTTPException 500: Synthesis failed
    """
    # Check whether the voice exists
    try:
        manager = _get_manager()
        manager.get_info(voice_id)  # Raises FileNotFoundError if not found
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice does not exist: {voice_id}",
        )

    # Execute synthesis
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
        if "model is not loaded" in error_msg.lower() or "TTS model" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="TTS model is not loaded. Please run setup/download_models.py to download the model and restart the service.",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {error_msg}",
        )
    except Exception as e:
        logger.error(f"Test synthesis failed ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis failed: {e}",
        )
