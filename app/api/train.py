"""
Training API router
Provides endpoints for creating voice cloning training tasks from uploaded files
or local folders. Training tasks execute asynchronously; query progress via task_id.
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

# ── In-memory task status store ───────────────────────────────────────────────
# In production, replace with Redis or a database; a dict is used here for simplicity
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
    Update task status (internal helper function).

    Args:
        task_id: Task ID
        status: New status
        progress: Progress (0~100)
        message: Stage description
        voice_id: Voice ID after completion (optional)
        error: Error message (optional)
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
    Main training pipeline function (executed as a background task).

    Workflow:
    1. Batch preprocess audio (format conversion + quality detection + denoising)
    2. Extract and fuse speaker embeddings
    3. Select the optimal reference audio
    4. Pack as a .voicepack file

    Args:
        task_id: Task ID
        audio_files: List of audio file paths
        voice_name: Voice name
        language: Language code
        temp_dir: Temporary directory path (cleaned up after processing)
    """
    try:
        from app.core.audio_processor import AudioProcessor
        from app.core.embedding_extractor import EmbeddingExtractor
        from app.core.voicepack_manager import VoicePackManager

        # ── Stage 1: Audio preprocessing ─────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 10, "Preprocessing audio...")
        logger.info(f"[{task_id}] Starting audio preprocessing, file count: {len(audio_files)}")

        processor = AudioProcessor(
            min_duration=settings.min_audio_duration_seconds,
            max_duration=settings.max_audio_duration_seconds,
        )

        processed_audio, batch_report = processor.process_batch(audio_files)

        if batch_report.success_count == 0:
            raise RuntimeError(
                f"All audio files failed processing: {list(batch_report.errors.values())[:3]}"
            )

        logger.info(
            f"[{task_id}] Preprocessing complete | "
            f"Succeeded: {batch_report.success_count}/{batch_report.total_files}"
        )

        # ── Stage 2: Extract speaker embedding ───────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 40, "Extracting speaker features...")
        logger.info(f"[{task_id}] Starting speaker embedding extraction")

        extractor = EmbeddingExtractor(
            model_dir=settings.model_dir,
        )

        # Build (audio, sr, quality_report) triple list
        audio_quality_pairs = [
            (audio, sr, batch_report.file_reports[fname])
            for fname, (audio, sr) in processed_audio.items()
            if fname in batch_report.file_reports
        ]

        fused_embedding, weight_records = extractor.extract_and_fuse(audio_quality_pairs)
        logger.info(f"[{task_id}] Embedding fusion complete | Dimension: {fused_embedding.shape[0]}")

        # ── Stage 3: Select optimal reference audio ───────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 70, "Selecting optimal reference audio...")

        best_audio, best_sr, best_report = extractor.select_reference_audio(audio_quality_pairs)
        logger.info(
            f"[{task_id}] Reference audio selected | "
            f"Duration: {best_report.duration_seconds:.1f}s | Quality: {best_report.quality_score:.3f}"
        )

        # ── Stage 4: Pack .voicepack ──────────────────────────────────────────
        _update_task(task_id, TaskStatusEnum.PROCESSING, 85, "Packing voice pack...")

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

        logger.info(f"[{task_id}] Voice pack packed successfully | voice_id: {voice_id}")

        # ── Complete ──────────────────────────────────────────────────────────
        _update_task(
            task_id,
            TaskStatusEnum.DONE,
            100,
            f"Training complete! Voice ID: {voice_id}",
            voice_id=voice_id,
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{task_id}] Training failed: {error_msg}")
        _update_task(
            task_id,
            TaskStatusEnum.FAILED,
            0,
            "Training failed",
            error=error_msg,
        )
    finally:
        # Clean up temporary files
        if temp_dir:
            cleanup_dir(temp_dir)


# ══════════════════════════════════════════════════════════════════════════════
# API routes
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/from-upload",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create training task from uploaded audio files",
    description="Upload 1~10 audio files to asynchronously create a voice training task. Returns a task_id for querying progress.",
)
async def train_from_upload(
    background_tasks: BackgroundTasks,
    voice_name: str = Form(..., description="Voice name (1~64 characters)", min_length=1, max_length=64),
    language: Language = Form(default=Language.ZH, description="Primary language"),
    files: List[UploadFile] = File(..., description="Reference audio files (supports WAV/MP3/M4A/FLAC/OGG)"),
) -> TrainResponse:
    """
    Create a training task from uploaded audio files.

    Args:
        background_tasks: FastAPI background task manager
        voice_name: User-defined voice name
        language: Primary language
        files: List of uploaded audio files (1~10)

    Returns:
        TrainResponse: Response containing task_id

    Raises:
        HTTPException 400: Unsupported file format or invalid file count
        HTTPException 413: File too large
    """
    # Validate file count
    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 audio file must be uploaded",
        )
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum 10 audio files are supported; {len(files)} were uploaded",
        )

    # Validate file formats
    for f in files:
        if not is_allowed_audio_file(f.filename or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {f.filename}. Supported formats: WAV/MP3/M4A/FLAC/OGG",
            )

    # Create a temporary directory to save uploaded files
    temp_dir = create_temp_dir(prefix=f"upload_{uuid.uuid4().hex[:8]}_")
    saved_paths: List[str] = []

    try:
        for upload_file in files:
            file_content = await upload_file.read()

            # Check file size
            if len(file_content) > settings.max_upload_size_bytes:
                cleanup_dir(temp_dir)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File {upload_file.filename} exceeds the size limit "
                        f"({settings.max_upload_size_mb} MB)"
                    ),
                )

            # Save to temporary directory
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
            detail=f"File save failed: {e}",
        )

    # Create task record
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _task_store[task_id] = TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        progress=0,
        message=f"Task created, waiting to process ({len(saved_paths)} file(s))",
        created_at=now,
        updated_at=now,
    )

    # Add background task
    background_tasks.add_task(
        _run_training_pipeline,
        task_id=task_id,
        audio_files=saved_paths,
        voice_name=voice_name,
        language=language.value,
        temp_dir=temp_dir,
    )

    logger.info(
        f"Training task created | task_id: {task_id} | "
        f"File count: {len(saved_paths)} | Voice name: {voice_name}"
    )

    return TrainResponse(
        task_id=task_id,
        message=f"Training task submitted with {len(saved_paths)} audio file(s). Use task_id to query progress.",
        status=TaskStatusEnum.PENDING,
    )


@router.post(
    "/from-folder",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create training task from a local folder",
    description="Specify a subfolder under data/samples/ to create a training task using its audio files.",
)
async def train_from_folder(
    background_tasks: BackgroundTasks,
    request: TrainFromFolderRequest,
) -> TrainResponse:
    """
    Create a training task from a local folder.

    Args:
        background_tasks: FastAPI background task manager
        request: Request body containing folder_name, voice_name, and language

    Returns:
        TrainResponse: Response containing task_id

    Raises:
        HTTPException 404: Folder does not exist
        HTTPException 400: Folder is empty or contains no valid audio files
    """
    folder_path = os.path.join(settings.samples_dir, request.folder_name)

    if not os.path.exists(folder_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Folder does not exist: data/samples/{request.folder_name}. "
                "Please create the subfolder and add audio files first."
            ),
        )

    try:
        audio_files = get_audio_files_in_dir(folder_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan folder: {e}",
        )

    if not audio_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No valid audio files found in folder data/samples/{request.folder_name}. "
                "Supported formats: WAV/MP3/M4A/FLAC/OGG"
            ),
        )

    if len(audio_files) > 10:
        audio_files = audio_files[:10]
        logger.info(f"Folder contains more than 10 files; using only the first 10")

    # Create task record
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _task_store[task_id] = TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        progress=0,
        message=f"Task created, using folder: {request.folder_name} ({len(audio_files)} file(s))",
        created_at=now,
        updated_at=now,
    )

    # Add background task
    background_tasks.add_task(
        _run_training_pipeline,
        task_id=task_id,
        audio_files=audio_files,
        voice_name=request.voice_name,
        language=request.language.value,
        temp_dir=None,  # No cleanup needed (using original file paths)
    )

    logger.info(
        f"Training task created (folder mode) | task_id: {task_id} | "
        f"Folder: {request.folder_name} | File count: {len(audio_files)}"
    )

    return TrainResponse(
        task_id=task_id,
        message=(
            f"Training task submitted using {len(audio_files)} audio file(s) "
            f"from folder data/samples/{request.folder_name}."
        ),
        status=TaskStatusEnum.PENDING,
    )


@router.get(
    "/{task_id}/status",
    response_model=TaskStatus,
    summary="Query training task status",
    description="Query the current status and progress of a training task by task_id.",
)
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Query training task status.

    Args:
        task_id: Unique ID of the training task

    Returns:
        TaskStatus: Detailed task status

    Raises:
        HTTPException 404: task_id does not exist
    """
    if task_id not in _task_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task does not exist: {task_id}",
        )
    return _task_store[task_id]
