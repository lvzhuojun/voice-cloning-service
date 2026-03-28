"""
Pydantic 数据模型定义
定义所有 API 请求/响应的数据结构，以及内部传递的数据对象。
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# 枚举类型
# ══════════════════════════════════════════════════════════════════════════════

class Language(str, Enum):
    """支持的语言枚举"""
    ZH = "zh"   # 中文
    EN = "en"   # 英文


class TaskStatusEnum(str, Enum):
    """训练任务状态枚举"""
    PENDING = "pending"       # 等待执行
    PROCESSING = "processing" # 处理中
    DONE = "done"             # 已完成
    FAILED = "failed"         # 失败


# ══════════════════════════════════════════════════════════════════════════════
# 音频质量报告
# ══════════════════════════════════════════════════════════════════════════════

class QualityReport(BaseModel):
    """
    单段音频的质量检测报告。

    Attributes:
        duration_seconds: 音频时长（秒）
        duration_ok: 时长是否在允许范围内（3~60秒）
        snr_db: 估算的信噪比（dB），越高越好
        quality_score: 综合质量评分（0.0~1.0）
        has_long_silence: 是否存在超过2秒的连续静音
        sample_rate: 原始采样率（Hz）
        channels: 声道数
        warnings: 质量警告列表（人类可读）
    """
    duration_seconds: float = Field(description="音频时长（秒）")
    duration_ok: bool = Field(description="时长是否在允许范围内")
    snr_db: float = Field(description="估算信噪比（dB）")
    quality_score: float = Field(ge=0.0, le=1.0, description="综合质量评分（0.0~1.0）")
    has_long_silence: bool = Field(description="是否存在超过2秒的连续静音段")
    sample_rate: int = Field(description="原始采样率（Hz）")
    channels: int = Field(description="声道数（1=单声道，2=立体声）")
    warnings: List[str] = Field(default_factory=list, description="质量警告列表")


class BatchProcessReport(BaseModel):
    """
    批量音频处理的汇总报告。

    Attributes:
        total_files: 处理的音频文件总数
        success_count: 成功处理的数量
        failed_count: 处理失败的数量
        total_duration_seconds: 成功处理文件的总时长
        average_quality_score: 平均质量评分
        file_reports: 每个文件的详细质量报告（键为文件名）
        errors: 处理失败的文件及错误信息
    """
    total_files: int = Field(description="处理的文件总数")
    success_count: int = Field(description="成功处理数量")
    failed_count: int = Field(description="失败数量")
    total_duration_seconds: float = Field(description="成功文件总时长（秒）")
    average_quality_score: float = Field(ge=0.0, le=1.0, description="平均质量评分")
    file_reports: Dict[str, QualityReport] = Field(default_factory=dict, description="各文件质量报告")
    errors: Dict[str, str] = Field(default_factory=dict, description="失败文件及错误信息")


# ══════════════════════════════════════════════════════════════════════════════
# 训练相关
# ══════════════════════════════════════════════════════════════════════════════

class TrainFromFolderRequest(BaseModel):
    """
    从本地文件夹发起训练的请求体。

    Attributes:
        folder_name: data/samples/ 下的子文件夹名称
        voice_name: 用户自定义的音色名称
        language: 主要语言（zh/en）
    """
    folder_name: str = Field(description="data/samples/ 下的子文件夹名称", min_length=1, max_length=128)
    voice_name: str = Field(description="用户自定义音色名称", min_length=1, max_length=64)
    language: Language = Field(default=Language.ZH, description="主要语言")

    @field_validator("folder_name")
    @classmethod
    def validate_folder_name(cls, v: str) -> str:
        """禁止路径穿越攻击"""
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("folder_name 不能包含路径分隔符或 '..'")
        return v.strip()


class TrainResponse(BaseModel):
    """
    创建训练任务的响应体。

    Attributes:
        task_id: 唯一任务 ID，用于查询进度
        message: 任务创建说明
        status: 初始状态（通常为 pending）
    """
    task_id: str = Field(description="唯一任务 ID")
    message: str = Field(description="任务创建说明")
    status: TaskStatusEnum = Field(default=TaskStatusEnum.PENDING)


class TaskStatus(BaseModel):
    """
    训练任务的状态信息。

    Attributes:
        task_id: 任务 ID
        status: 当前状态
        progress: 进度百分比（0~100）
        message: 当前阶段描述
        voice_id: 训练完成后生成的音色 ID（仅 done 状态有值）
        error: 错误信息（仅 failed 状态有值）
        created_at: 任务创建时间
        updated_at: 最后更新时间
    """
    task_id: str = Field(description="任务 ID")
    status: TaskStatusEnum = Field(description="当前状态")
    progress: int = Field(ge=0, le=100, description="进度百分比（0~100）")
    message: str = Field(description="当前阶段描述")
    voice_id: Optional[str] = Field(default=None, description="训练完成后的音色 ID")
    error: Optional[str] = Field(default=None, description="错误信息（失败时）")
    created_at: datetime = Field(description="任务创建时间")
    updated_at: datetime = Field(description="最后更新时间")


# ══════════════════════════════════════════════════════════════════════════════
# 音色信息
# ══════════════════════════════════════════════════════════════════════════════

class VoiceInfo(BaseModel):
    """
    音色包的基本信息（不含 embedding 数据，用于列表展示）。

    Attributes:
        voice_id: UUID v4 唯一标识
        voice_name: 用户自定义名称
        created_at: 创建时间（ISO 8601）
        model_type: 模型类型（cosyvoice3）
        model_version: 模型版本
        language: 主要语言
        sample_count: 训练用参考音频数量
        total_duration_seconds: 参考音频总时长
        embedding_dim: embedding 维度
        quality_score: 综合音质评分（0.0~1.0）
        voicepack_size_bytes: .voicepack 文件大小（字节）
    """
    voice_id: str = Field(description="UUID v4 音色唯一标识")
    voice_name: str = Field(description="用户自定义名称")
    created_at: str = Field(description="创建时间（ISO 8601）")
    model_type: str = Field(description="模型类型")
    model_version: str = Field(description="模型版本")
    language: str = Field(description="主要语言")
    sample_count: int = Field(description="参考音频数量")
    total_duration_seconds: float = Field(description="参考音频总时长（秒）")
    embedding_dim: int = Field(description="Embedding 向量维度")
    quality_score: float = Field(ge=0.0, le=1.0, description="综合音质评分")
    voicepack_size_bytes: Optional[int] = Field(default=None, description=".voicepack 文件大小（字节）")


class VoiceListResponse(BaseModel):
    """
    音色列表响应。

    Attributes:
        total: 音色总数
        voices: 音色信息列表
    """
    total: int = Field(description="音色总数")
    voices: List[VoiceInfo] = Field(description="音色列表")


# ══════════════════════════════════════════════════════════════════════════════
# TTS 合成
# ══════════════════════════════════════════════════════════════════════════════

class TestVoiceRequest(BaseModel):
    """
    测试语音合成的请求体。

    Attributes:
        text: 要合成的文字（1~200字）
        speed: 语速（0.5=慢，1.0=正常，2.0=快）
        language: 合成语言
    """
    text: str = Field(
        description="要合成的文字",
        min_length=1,
        max_length=200,
        default="你好，这是一段语音克隆测试。"
    )
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速（0.5~2.0）")
    language: Language = Field(default=Language.ZH, description="合成语言")


# ══════════════════════════════════════════════════════════════════════════════
# 健康检查
# ══════════════════════════════════════════════════════════════════════════════

class GPUInfo(BaseModel):
    """
    GPU 信息。

    Attributes:
        name: GPU 型号名称
        compute_capability: CUDA 计算能力（如 "12.0"）
        total_memory_gb: 总显存（GB）
        cuda_available: CUDA 是否可用
    """
    name: str = Field(description="GPU 型号")
    compute_capability: str = Field(description="CUDA 计算能力")
    total_memory_gb: float = Field(description="总显存（GB）")
    cuda_available: bool = Field(description="CUDA 是否可用")


class HealthResponse(BaseModel):
    """
    健康检查响应。

    Attributes:
        status: 服务状态（ok/degraded/error）
        version: 服务版本
        model_loaded: 模型是否已加载
        gpu_info: GPU 信息
        voice_count: 已训练音色数量
        uptime_seconds: 服务运行时长（秒）
    """
    status: str = Field(description="服务状态（ok/degraded/error）")
    version: str = Field(default="1.0.0", description="服务版本")
    model_loaded: bool = Field(description="模型是否已加载")
    gpu_info: Optional[GPUInfo] = Field(default=None, description="GPU 信息")
    voice_count: int = Field(description="已训练音色数量")
    uptime_seconds: float = Field(description="服务运行时长（秒）")
