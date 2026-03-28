"""
配置管理模块
从 .env 文件和环境变量读取所有配置项，提供全局 settings 单例。
使用 pydantic-settings 自动完成类型转换和校验。
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    全局配置类，字段自动从 .env 文件 / 环境变量读取。

    优先级：环境变量 > .env 文件 > 字段默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── 服务配置 ────────────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="服务监听地址")
    port: int = Field(default=8000, description="服务监听端口")

    # ── 存储目录 ────────────────────────────────────────────────────────────
    model_dir: str = Field(
        default="storage/pretrained_models",
        description="预训练模型存储目录",
    )
    storage_dir: str = Field(
        default="storage",
        description="存储根目录（uploads / voicepacks 的父目录）",
    )

    # ── HuggingFace 配置 ────────────────────────────────────────────────────
    hf_endpoint: str = Field(
        default="https://hf-mirror.com",
        description="HuggingFace 镜像源 URL",
    )

    # ── 日志配置 ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="日志级别")

    # ── 上传限制 ────────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(
        default=100,
        description="单文件上传大小限制（MB）",
    )
    max_audio_duration_seconds: float = Field(
        default=60.0,
        description="音频最大允许时长（秒）",
    )
    min_audio_duration_seconds: float = Field(
        default=3.0,
        description="音频最小允许时长（秒）",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别合法性"""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level 必须为 {allowed} 之一，当前: {v}")
        return v.upper()

    @property
    def uploads_dir(self) -> str:
        """用户上传的临时音频目录"""
        return os.path.join(self.storage_dir, "uploads")

    @property
    def voicepacks_dir(self) -> str:
        """训练好的音色包存储目录"""
        return os.path.join(self.storage_dir, "voicepacks")

    @property
    def samples_dir(self) -> str:
        """本地样本音频目录"""
        return os.path.join("data", "samples")

    @property
    def max_upload_size_bytes(self) -> int:
        """上传大小限制（字节）"""
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """
        确保所有必要的存储目录存在。
        在服务启动时调用。
        """
        dirs = [
            self.storage_dir,
            self.uploads_dir,
            self.voicepacks_dir,
            self.model_dir,
            self.samples_dir,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


# 全局 settings 单例
# 其他模块通过 `from app.config import settings` 引用
settings = Settings()
