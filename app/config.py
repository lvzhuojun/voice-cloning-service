"""
Configuration management module
Reads all configuration items from .env files and environment variables,
and provides a global settings singleton.
Uses pydantic-settings for automatic type conversion and validation.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global configuration class; fields are automatically read from .env / environment variables.

    Priority: environment variables > .env file > field defaults
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Service configuration --------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Service listen address")
    port: int = Field(default=8000, description="Service listen port")

    # -- Storage directories ---------------------------------------------------
    model_dir: str = Field(
        default="storage/pretrained_models",
        description="Pretrained model storage directory",
    )
    storage_dir: str = Field(
        default="storage",
        description="Storage root directory (parent of uploads / models)",
    )

    # -- GPT-SoVITS configuration ----------------------------------------------
    gptsovits_dir: str = Field(
        default="GPT-SoVITS",
        description="GPT-SoVITS source directory (cloned separately)",
    )
    models_dir: str = Field(
        default="storage/models",
        description="Directory for trained voice model files",
    )
    whisper_model: str = Field(
        default="medium",
        description="Whisper model size for transcription (tiny/base/small/medium/large)",
    )
    training_epochs_gpt: int = Field(
        default=15,
        description="Default number of GPT training epochs",
    )
    training_epochs_sovits: int = Field(
        default=8,
        description="Default number of SoVITS training epochs",
    )

    # -- HuggingFace configuration ---------------------------------------------
    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace mirror URL",
    )

    # -- Logging configuration -------------------------------------------------
    log_level: str = Field(default="INFO", description="Log level")

    # -- Upload limits ---------------------------------------------------------
    max_upload_size_mb: int = Field(
        default=200,
        description="Single-file upload size limit (MB)",
    )
    max_audio_duration_seconds: float = Field(
        default=300.0,
        description="Maximum allowed audio duration (seconds)",
    )
    min_audio_duration_seconds: float = Field(
        default=3.0,
        description="Minimum allowed audio duration (seconds)",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that the log level is a legal value"""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got: {v}")
        return v.upper()

    @property
    def uploads_dir(self) -> str:
        """Temporary audio directory for user uploads"""
        return os.path.join(self.storage_dir, "uploads")

    @property
    def samples_dir(self) -> str:
        """Local sample audio directory"""
        return os.path.join("data", "samples")

    @property
    def pretrained_gptsovits_dir(self) -> str:
        """Pretrained GPT-SoVITS model directory"""
        return os.path.join(self.model_dir, "GPT-SoVITS")

    @property
    def max_upload_size_bytes(self) -> int:
        """Upload size limit (bytes)"""
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """
        Ensure that all required storage directories exist.
        Call this when the service starts.
        """
        dirs = [
            self.storage_dir,
            self.uploads_dir,
            self.models_dir,
            self.model_dir,
            self.samples_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# Global settings singleton
# Other modules reference this via `from app.config import settings`
settings = Settings()
