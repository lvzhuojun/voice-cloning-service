"""
Pydantic data model definitions
Defines data structures for all API requests/responses and internal data objects.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    """Supported language enumeration"""
    ZH = "zh"   # Chinese
    EN = "en"   # English


class TaskStatusEnum(str, Enum):
    """Training task status enumeration"""
    PENDING = "pending"       # Waiting to execute
    PROCESSING = "processing" # In progress
    DONE = "done"             # Completed
    FAILED = "failed"         # Failed


# ══════════════════════════════════════════════════════════════════════════════
# Audio quality report
# ══════════════════════════════════════════════════════════════════════════════

class QualityReport(BaseModel):
    """
    Quality inspection report for a single audio segment.

    Attributes:
        duration_seconds: Audio duration in seconds
        duration_ok: Whether the duration is within the allowed range (3~60 s)
        snr_db: Estimated signal-to-noise ratio (dB); higher is better
        quality_score: Overall quality score (0.0~1.0)
        has_long_silence: Whether there is a continuous silence segment longer than 2 s
        sample_rate: Original sample rate (Hz)
        channels: Number of channels
        warnings: List of quality warnings (human-readable)
    """
    duration_seconds: float = Field(description="Audio duration in seconds")
    duration_ok: bool = Field(description="Whether the duration is within the allowed range")
    snr_db: float = Field(description="Estimated signal-to-noise ratio (dB)")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score (0.0~1.0)")
    has_long_silence: bool = Field(description="Whether there is a continuous silence segment longer than 2 s")
    sample_rate: int = Field(description="Original sample rate (Hz)")
    channels: int = Field(description="Number of channels (1=mono, 2=stereo)")
    warnings: List[str] = Field(default_factory=list, description="List of quality warnings")


class BatchProcessReport(BaseModel):
    """
    Summary report for batch audio processing.

    Attributes:
        total_files: Total number of audio files processed
        success_count: Number of successfully processed files
        failed_count: Number of files that failed processing
        total_duration_seconds: Total duration of successfully processed files
        average_quality_score: Average quality score
        file_reports: Detailed quality report per file (keyed by filename)
        errors: Files that failed processing along with their error messages
    """
    total_files: int = Field(description="Total number of files processed")
    success_count: int = Field(description="Number of successfully processed files")
    failed_count: int = Field(description="Number of failed files")
    total_duration_seconds: float = Field(description="Total duration of successful files (seconds)")
    average_quality_score: float = Field(ge=0.0, le=1.0, description="Average quality score")
    file_reports: Dict[str, QualityReport] = Field(default_factory=dict, description="Quality report per file")
    errors: Dict[str, str] = Field(default_factory=dict, description="Failed files and their error messages")


# ══════════════════════════════════════════════════════════════════════════════
# Training-related
# ══════════════════════════════════════════════════════════════════════════════

class TrainFromFolderRequest(BaseModel):
    """
    Request body for starting a training job from a local folder.

    Attributes:
        folder_name: Subfolder name under data/samples/
        voice_name: User-defined voice name
        language: Primary language (zh/en)
    """
    folder_name: str = Field(description="Subfolder name under data/samples/", min_length=1, max_length=128)
    voice_name: str = Field(description="User-defined voice name", min_length=1, max_length=64)
    language: Language = Field(default=Language.ZH, description="Primary language")

    @field_validator("folder_name")
    @classmethod
    def validate_folder_name(cls, v: str) -> str:
        """Disallow path traversal attacks"""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("folder_name must not contain path separators or '..'")
        return v.strip()


class TrainResponse(BaseModel):
    """
    Response body for creating a training task.

    Attributes:
        task_id: Unique task ID for querying progress
        message: Task creation description
        status: Initial status (usually pending)
    """
    task_id: str = Field(description="Unique task ID")
    message: str = Field(description="Task creation description")
    status: TaskStatusEnum = Field(default=TaskStatusEnum.PENDING)


class TaskStatus(BaseModel):
    """
    Status information for a training task.

    Attributes:
        task_id: Task ID
        status: Current status
        progress: Progress percentage (0~100)
        message: Current stage description
        voice_id: Voice ID generated upon completion (only set when status is done)
        error: Error message (only set when status is failed)
        created_at: Task creation time
        updated_at: Last update time
    """
    task_id: str = Field(description="Task ID")
    status: TaskStatusEnum = Field(description="Current status")
    progress: int = Field(ge=0, le=100, description="Progress percentage (0~100)")
    message: str = Field(description="Current stage description")
    voice_id: Optional[str] = Field(default=None, description="Voice ID after training completes")
    error: Optional[str] = Field(default=None, description="Error message (when failed)")
    created_at: datetime = Field(description="Task creation time")
    updated_at: datetime = Field(description="Last update time")


# ══════════════════════════════════════════════════════════════════════════════
# Voice information
# ══════════════════════════════════════════════════════════════════════════════

class VoiceInfo(BaseModel):
    """
    Basic information about a voice pack (without embedding data; used for list display).

    Attributes:
        voice_id: UUID v4 unique identifier
        voice_name: User-defined name
        created_at: Creation time (ISO 8601)
        model_type: Model type (cosyvoice3)
        model_version: Model version
        language: Primary language
        sample_count: Number of reference audio segments used for training
        total_duration_seconds: Total duration of reference audio
        embedding_dim: Embedding vector dimension
        quality_score: Overall audio quality score (0.0~1.0)
        voicepack_size_bytes: .voicepack file size in bytes
    """
    voice_id: str = Field(description="UUID v4 voice unique identifier")
    voice_name: str = Field(description="User-defined name")
    created_at: str = Field(description="Creation time (ISO 8601)")
    model_type: str = Field(description="Model type")
    model_version: str = Field(description="Model version")
    language: str = Field(description="Primary language")
    sample_count: int = Field(description="Number of reference audio segments")
    total_duration_seconds: float = Field(description="Total duration of reference audio (seconds)")
    embedding_dim: int = Field(description="Embedding vector dimension")
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall audio quality score")
    voicepack_size_bytes: Optional[int] = Field(default=None, description=".voicepack file size in bytes")


class VoiceListResponse(BaseModel):
    """
    Voice list response.

    Attributes:
        total: Total number of voices
        voices: List of voice information
    """
    total: int = Field(description="Total number of voices")
    voices: List[VoiceInfo] = Field(description="List of voices")


# ══════════════════════════════════════════════════════════════════════════════
# TTS synthesis
# ══════════════════════════════════════════════════════════════════════════════

class TestVoiceRequest(BaseModel):
    """
    Request body for testing voice synthesis.

    Attributes:
        text: Text to synthesize (1~200 characters)
        speed: Speech speed (0.5=slow, 1.0=normal, 2.0=fast)
        language: Synthesis language
    """
    text: str = Field(
        description="Text to synthesize",
        min_length=1,
        max_length=200,
        default="Hello, this is a voice cloning test."
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed (0.5~2.0)")
    language: Language = Field(default=Language.ZH, description="Synthesis language")


# ══════════════════════════════════════════════════════════════════════════════
# Health check
# ══════════════════════════════════════════════════════════════════════════════

class GPUInfo(BaseModel):
    """
    GPU information.

    Attributes:
        name: GPU model name
        compute_capability: CUDA compute capability (e.g. "12.0")
        total_memory_gb: Total VRAM (GB)
        cuda_available: Whether CUDA is available
    """
    name: str = Field(description="GPU model name")
    compute_capability: str = Field(description="CUDA compute capability")
    total_memory_gb: float = Field(description="Total VRAM (GB)")
    cuda_available: bool = Field(description="Whether CUDA is available")


class HealthResponse(BaseModel):
    """
    Health check response.

    Attributes:
        status: Service status (ok/degraded/error)
        version: Service version
        model_loaded: Whether the model is loaded
        gpu_info: GPU information
        voice_count: Number of trained voices
        uptime_seconds: Service uptime in seconds
    """
    status: str = Field(description="Service status (ok/degraded/error)")
    version: str = Field(default="1.0.0", description="Service version")
    model_loaded: bool = Field(description="Whether the model is loaded")
    gpu_info: Optional[GPUInfo] = Field(default=None, description="GPU information")
    voice_count: int = Field(description="Number of trained voices")
    uptime_seconds: float = Field(description="Service uptime in seconds")
