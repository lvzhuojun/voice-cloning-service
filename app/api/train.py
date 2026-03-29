"""
Training API router
Provides endpoints for creating GPT-SoVITS voice fine-tuning tasks from uploaded files
or local folders. Training tasks execute asynchronously; query progress via task_id.
"""

from __future__ import annotations

import logging
import os
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

# -- In-memory task status store -----------------------------------------------
# In production, replace with Redis or a database
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
    Update task status (internal helper).

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
    GPT-SoVITS fine-tuning pipeline (executed as a background task).

    Workflow:
    1. Slice audio into 3-10s segments
    2. Transcribe with Whisper
    3. Extract Hubert, semantic, and BERT features
    4. Fine-tune GPT (s1) model
    5. Fine-tune SoVITS (s2) model
    6. Save metadata

    Args:
        task_id: Task ID
        audio_files: List of audio file paths
        voice_name: Voice display name
        language: Language code
        temp_dir: Temporary directory (cleaned up after processing)
    """
    from app.core.trainer import GPTSoVITSTrainer
    from app.core.voice_manager import VoiceManager

    voice_id = str(uuid.uuid4())

    def progress_callback(stage: str, pct: int, msg: str) -> None:
        _update_task(task_id, TaskStatusEnum.PROCESSING, pct, msg)

    try:
        _update_task(task_id, TaskStatusEnum.PROCESSING, 5, "Starting training pipeline...")
        logger.info(f"[{task_id}] Starting GPT-SoVITS training | voice_id: {voice_id}")

        trainer = GPTSoVITSTrainer(
            gptsovits_dir=settings.gptsovits_dir,
            pretrained_dir=settings.pretrained_gptsovits_dir,
            output_dir=settings.models_dir,
        )

        metadata = trainer.train(
            voice_id=voice_id,
            voice_name=voice_name,
            audio_files=audio_files,
            language=language,
            epochs_gpt=settings.training_epochs_gpt,
            epochs_sovits=settings.training_epochs_sovits,
            progress_callback=progress_callback,
        )

        # Save metadata to voice directory
        manager = VoiceManager(storage_dir=settings.storage_dir)
        manager.save_metadata(voice_id, metadata)

        logger.info(f"[{task_id}] Training complete | voice_id: {voice_id}")
        _update_task(
            task_id,
            TaskStatusEnum.DONE,
            100,
            f"Training complete! Voice ID: {voice_id}",
            voice_id=voice_id,
        )

    except RuntimeError as e:
        error_msg = str(e)
        logger.error(f"[{task_id}] Training failed (RuntimeError): {error_msg}")
        _update_task(
            task_id,
            TaskStatusEnum.FAILED,
            0,
            "Training failed",
            error=error_msg,
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{task_id}] Training failed: {error_msg}", exc_info=True)
        _update_task(
            task_id,
            TaskStatusEnum.FAILED,
            0,
            "Training failed",
            error=error_msg,
        )
    finally:
        if temp_dir:
            cleanup_dir(temp_dir)


# ================================================================================
# API routes
# ================================================================================

@router.post(
    "/from-upload",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create training task from uploaded audio files",
    description=(
        "Upload 1~10 audio files to asynchronously start a GPT-SoVITS fine-tuning task. "
        "Returns a task_id for querying progress. "
        "Recommended: 1-3 minutes of clean speech, split across multiple files."
    ),
)
async def train_from_upload(
    background_tasks: BackgroundTasks,
    voice_name: str = Form(..., description="Voice name (1~64 characters)", min_length=1, max_length=64),
    language: Language = Form(default=Language.ZH, description="Primary language"),
    files: List[UploadFile] = File(..., description="Reference audio files (WAV/MP3/M4A/FLAC/OGG)"),
) -> TrainResponse:
    """
    Create a GPT-SoVITS training task from uploaded audio files.

    Args:
        background_tasks: FastAPI background task manager
        voice_name: User-defined voice name
        language: Primary language
        files: Uploaded audio files (1~10)

    Returns:
        TrainResponse: Response containing task_id

    Raises:
        HTTPException 400: Unsupported file format or invalid file count
        HTTPException 413: File too large
    """
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

    for f in files:
        if not is_allowed_audio_file(f.filename or ""):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format: {f.filename}. Supported: WAV/MP3/M4A/FLAC/OGG",
            )

    # Save to temp dir
    temp_dir = create_temp_dir(prefix=f"upload_{uuid.uuid4().hex[:8]}_")
    saved_paths: List[str] = []

    try:
        for upload_file in files:
            file_content = await upload_file.read()

            if len(file_content) > settings.max_upload_size_bytes:
                cleanup_dir(temp_dir)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File {upload_file.filename} exceeds the size limit "
                        f"({settings.max_upload_size_mb} MB)"
                    ),
                )

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

    # Create task
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
        f"Files: {len(saved_paths)} | Voice: {voice_name}"
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
    description="Specify a subfolder under data/samples/ to start a GPT-SoVITS training task.",
)
async def train_from_folder(
    background_tasks: BackgroundTasks,
    request: TrainFromFolderRequest,
) -> TrainResponse:
    """
    Create a training task from a local audio folder.

    Args:
        background_tasks: FastAPI background task manager
        request: Request body (folder_name, voice_name, language)

    Returns:
        TrainResponse: Response containing task_id

    Raises:
        HTTPException 404: Folder does not exist
        HTTPException 400: No valid audio files in folder
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
                f"No valid audio files found in data/samples/{request.folder_name}. "
                "Supported formats: WAV/MP3/M4A/FLAC/OGG"
            ),
        )

    if len(audio_files) > 10:
        audio_files = audio_files[:10]
        logger.info("Folder has more than 10 files; using first 10")

    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    _task_store[task_id] = TaskStatus(
        task_id=task_id,
        status=TaskStatusEnum.PENDING,
        progress=0,
        message=f"Task created for folder: {request.folder_name} ({len(audio_files)} file(s))",
        created_at=now,
        updated_at=now,
    )

    background_tasks.add_task(
        _run_training_pipeline,
        task_id=task_id,
        audio_files=audio_files,
        voice_name=request.voice_name,
        language=request.language.value,
        temp_dir=None,
    )

    logger.info(
        f"Training task created (folder) | task_id: {task_id} | "
        f"Folder: {request.folder_name} | Files: {len(audio_files)}"
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
