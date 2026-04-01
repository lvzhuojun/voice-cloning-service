"""
Voice management API router
Provides endpoints for listing, retrieving, downloading, deleting, and
test-synthesizing GPT-SoVITS voice models.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from app.config import settings
from app.models.schemas import TestVoiceRequest, VoiceInfo, VoiceListResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_manager():
    """Get a VoiceManager instance (lazy loading)."""
    from app.core.voice_manager import VoiceManager
    return VoiceManager(storage_dir=settings.storage_dir)


def _get_engine():
    """Get the TTSEngine singleton (lazy loading)."""
    from app.core.tts_engine import TTSEngine
    return TTSEngine.get_instance(
        gptsovits_dir=settings.gptsovits_dir,
        pretrained_dir=settings.pretrained_gptsovits_dir,
        models_dir=settings.models_dir,
    )


# ================================================================================
# List / Get
# ================================================================================

@router.get(
    "/",
    response_model=VoiceListResponse,
    summary="List all voices",
    description="Returns a list of all trained GPT-SoVITS voices, sorted by creation time (newest first).",
)
async def list_voices() -> VoiceListResponse:
    """
    Get a list of all voice models.

    Returns:
        VoiceListResponse: Total count and list of VoiceInfo objects
    """
    try:
        manager = _get_manager()
        raw_list = manager.list_all()
        voices = []
        for meta in raw_list:
            try:
                voices.append(VoiceInfo(**meta))
            except Exception as e:
                logger.warning(f"Skipping malformed voice metadata: {e}")
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
    description="Returns metadata for the specified voice model.",
)
async def get_voice(voice_id: str) -> VoiceInfo:
    """
    Get detailed information for a specific voice.

    Args:
        voice_id: Voice UUID string

    Returns:
        VoiceInfo: Voice metadata

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


# ================================================================================
# Delete
# ================================================================================

@router.delete(
    "/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete voice",
    description="Permanently delete the specified voice model and all its files. This cannot be undone.",
)
async def delete_voice(voice_id: str) -> None:
    """
    Delete the specified voice model.

    Args:
        voice_id: Voice UUID string

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
        # Evict from TTS engine cache if loaded
        try:
            engine = _get_engine()
            engine.evict_cache(voice_id)
        except Exception:
            pass
        logger.info(f"Voice deleted: {voice_id}")
    except HTTPException:
        raise
    except OSError as e:
        logger.error(f"Failed to delete voice ({voice_id}): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice: {e}",
        )


# ================================================================================
# Test synthesis
# ================================================================================

@router.post(
    "/{voice_id}/test",
    summary="Test voice synthesis",
    description="Synthesize a test speech using the specified GPT-SoVITS voice and return a WAV file.",
)
async def test_voice(voice_id: str, request: TestVoiceRequest) -> Response:
    """
    Synthesize test speech using the specified voice.

    Args:
        voice_id: Voice UUID string
        request: Synthesis parameters (text, speed, language)

    Returns:
        Response: WAV audio data (Content-Type: audio/wav)

    Raises:
        HTTPException 404: Voice does not exist
        HTTPException 503: GPT-SoVITS not ready
        HTTPException 500: Synthesis failed
    """
    # Check that voice exists
    try:
        manager = _get_manager()
        manager.get_metadata(voice_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice does not exist: {voice_id}",
        )

    # Run synthesis
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
        if "not cloned" in error_msg.lower() or "GPT-SoVITS directory" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "GPT-SoVITS is not set up. "
                    "Please run: bash setup/clone_gptsovits.sh && python setup/download_models.py"
                ),
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


# ================================================================================
# Downloads
# ================================================================================

@router.get(
    "/{voice_id}/download/gpt",
    summary="Download GPT model",
    description="Download the GPT (s1) checkpoint file for the specified voice.",
)
async def download_gpt_model(voice_id: str):
    """
    Download the GPT checkpoint (.ckpt) for a voice.

    Args:
        voice_id: Voice UUID string

    Returns:
        FileResponse: .ckpt file download

    Raises:
        HTTPException 404: Voice or GPT model file not found
    """
    manager = _get_manager()
    paths = manager.get_model_paths(voice_id)
    gpt_path = paths["gpt"]

    if not gpt_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GPT model file not found for voice: {voice_id}",
        )

    filename = f"{voice_id}_gpt.ckpt"
    return FileResponse(
        path=str(gpt_path),
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{voice_id}/download/sovits",
    summary="Download SoVITS model",
    description="Download the SoVITS (s2G) generator weights for the specified voice.",
)
async def download_sovits_model(voice_id: str):
    """
    Download the SoVITS generator weights (.pth) for a voice.

    Args:
        voice_id: Voice UUID string

    Returns:
        FileResponse: .pth file download

    Raises:
        HTTPException 404: Voice or SoVITS model file not found
    """
    manager = _get_manager()
    paths = manager.get_model_paths(voice_id)
    sovits_path = paths["sovits"]

    if not sovits_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SoVITS model file not found for voice: {voice_id}",
        )

    filename = f"{voice_id}_sovits.pth"
    return FileResponse(
        path=str(sovits_path),
        media_type="application/octet-stream",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{voice_id}/download/all",
    summary="Download all voice files as ZIP",
    description="Download a ZIP archive containing GPT model, SoVITS model, and metadata.json.",
)
async def download_all(voice_id: str) -> FileResponse:
    """
    Download all voice model files as a ZIP archive.

    Contents:
        - {voice_id}_gpt.ckpt
        - {voice_id}_sovits.pth
        - metadata.json
        - reference.wav (if present)

    Args:
        voice_id: Voice UUID string

    Returns:
        FileResponse: ZIP file streamed from a temporary file

    Raises:
        HTTPException 404: Voice does not exist
        HTTPException 500: ZIP creation failed
    """
    manager = _get_manager()

    if not manager.voice_exists(voice_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice does not exist: {voice_id}",
        )

    paths = manager.get_model_paths(voice_id)
    zip_filename = f"{voice_id}_voice_model.zip"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp_path = tmp.name
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as zf:
            for file_path in [paths["gpt"], paths["sovits"], paths["metadata"], paths["reference"]]:
                if file_path.exists():
                    zf.write(str(file_path), arcname=file_path.name)
        logger.info(f"ZIP created for {voice_id}: {os.path.getsize(tmp_path):,} bytes")
    except Exception as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Failed to create ZIP for voice {voice_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create ZIP archive: {exc}",
        )

    return FileResponse(
        path=tmp_path,
        media_type="application/zip",
        filename=zip_filename,
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
        background=BackgroundTask(os.unlink, tmp_path),
    )
