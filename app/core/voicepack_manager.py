"""
VoicePack 打包管理模块
负责 .voicepack 文件的创建、解包、验证和管理。

.voicepack 是一个标准 ZIP 文件，内部结构固定（详见 VOICEPACK_FORMAT.md）：
    {voice_id}/
    ├── speaker_embedding.pt     # torch.Tensor，speaker embedding 向量
    ├── reference_audio.wav      # 最优参考音频片段
    ├── model_config.json        # 模型配置
    └── metadata.json            # 元数据
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from app.models.schemas import VoiceInfo

logger = logging.getLogger(__name__)

# ── .voicepack 内部结构规范 ───────────────────────────────────────────────────
REQUIRED_FILES = frozenset({
    "speaker_embedding.pt",
    "reference_audio.wav",
    "metadata.json",
    "model_config.json",
})

MODEL_TYPE = "cosyvoice3"
MODEL_VERSION = "Fun-CosyVoice3-0.5B-2512"
REFERENCE_AUDIO_SAMPLE_RATE = 22050  # CosyVoice3 要求的参考音频采样率


class VoicePackManager:
    """
    音色包管理器，封装 .voicepack 文件的全生命周期操作。

    功能：
    - pack():      打包 embedding、参考音频、配置为 .voicepack
    - unpack():    解包并验证 .voicepack 文件
    - validate():  仅验证格式，不解包内容
    - list_all():  列出所有已有音色包的元数据
    - delete():    删除指定音色包
    - get_info():  读取元数据（不解包 embedding）

    使用方式:
        manager = VoicePackManager(storage_dir="storage")
        voice_id = manager.pack(embedding, ref_audio, sr, metadata_kwargs)
        info = manager.get_info(voice_id)
    """

    def __init__(self, storage_dir: str = "storage") -> None:
        """
        初始化 VoicePackManager。

        Args:
            storage_dir: 存储根目录，音色包保存在 {storage_dir}/voicepacks/

        Raises:
            OSError: 无法创建存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.voicepacks_dir = self.storage_dir / "voicepacks"

        # 确保存储目录存在
        try:
            self.voicepacks_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"无法创建音色包存储目录 ({self.voicepacks_dir}): {e}") from e

        logger.info(f"VoicePackManager 初始化 | 存储目录: {self.voicepacks_dir.absolute()}")

    def pack(
        self,
        embedding: torch.Tensor,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        voice_name: str,
        language: str = "zh",
        sample_count: int = 1,
        total_duration_seconds: float = 0.0,
        quality_score: float = 0.0,
        voice_id: Optional[str] = None,
    ) -> str:
        """
        将 speaker embedding 和参考音频打包为 .voicepack 文件。

        打包内容：
        - speaker_embedding.pt：torch.Tensor，形状 (embedding_dim,)
        - reference_audio.wav：16-bit PCM WAV，22050Hz 单声道
        - metadata.json：音色元数据
        - model_config.json：模型配置

        Args:
            embedding: speaker embedding 向量，形状 (embedding_dim,)
            reference_audio: 参考音频数组（float32）
            reference_sample_rate: 参考音频采样率（Hz）
            voice_name: 用户自定义的音色名称
            language: 主要语言（"zh"/"en"），默认 "zh"
            sample_count: 训练时使用的参考音频段数
            total_duration_seconds: 所有参考音频总时长（秒）
            quality_score: 综合音质评分（0.0~1.0）
            voice_id: 指定 voice_id（UUID v4），不传则自动生成

        Returns:
            str: 生成的 voice_id（UUID v4 字符串）

        Raises:
            ValueError: embedding 形状不合法
            IOError: 文件写入失败
        """
        # 参数验证
        if embedding.ndim != 1:
            raise ValueError(f"embedding 必须为 1D 张量，当前形状: {embedding.shape}")

        embedding_dim = embedding.shape[0]

        # 生成或使用指定的 voice_id
        if voice_id is None:
            voice_id = str(uuid.uuid4())

        output_path = self.voicepacks_dir / f"{voice_id}.voicepack"

        logger.info(f"打包音色包 | voice_id: {voice_id} | 名称: {voice_name}")

        try:
            with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
                # ── 写入 speaker embedding ──────────────────────────────────
                emb_buf = io.BytesIO()
                torch.save(embedding.cpu().float(), emb_buf)
                zf.writestr(
                    f"{voice_id}/speaker_embedding.pt",
                    emb_buf.getvalue(),
                )

                # ── 写入参考音频 ────────────────────────────────────────────
                # 确保采样率符合 CosyVoice3 要求
                if reference_sample_rate != REFERENCE_AUDIO_SAMPLE_RATE:
                    import librosa
                    reference_audio = librosa.resample(
                        reference_audio,
                        orig_sr=reference_sample_rate,
                        target_sr=REFERENCE_AUDIO_SAMPLE_RATE,
                    )

                audio_buf = io.BytesIO()
                sf.write(
                    audio_buf,
                    reference_audio,
                    REFERENCE_AUDIO_SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )
                zf.writestr(
                    f"{voice_id}/reference_audio.wav",
                    audio_buf.getvalue(),
                )

                # ── 写入 metadata.json ──────────────────────────────────────
                metadata = {
                    "voice_id": voice_id,
                    "voice_name": voice_name,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_type": MODEL_TYPE,
                    "model_version": MODEL_VERSION,
                    "language": language,
                    "sample_count": sample_count,
                    "total_duration_seconds": round(total_duration_seconds, 3),
                    "embedding_dim": embedding_dim,
                    "quality_score": round(quality_score, 4),
                }
                zf.writestr(
                    f"{voice_id}/metadata.json",
                    json.dumps(metadata, ensure_ascii=False, indent=2),
                )

                # ── 写入 model_config.json ──────────────────────────────────
                model_config = {
                    "sample_rate": REFERENCE_AUDIO_SAMPLE_RATE,
                    "model_type": MODEL_TYPE,
                    "embedding_method": "multi_audio_weighted_average",
                    "cosyvoice_config": {},
                }
                zf.writestr(
                    f"{voice_id}/model_config.json",
                    json.dumps(model_config, ensure_ascii=False, indent=2),
                )

        except zipfile.BadZipFile as e:
            raise IOError(f"ZIP 打包失败 ({output_path}): {e}") from e
        except Exception as e:
            # 打包失败时删除不完整的文件
            if output_path.exists():
                output_path.unlink()
            raise IOError(f"音色包打包失败: {e}") from e

        file_size = output_path.stat().st_size
        logger.info(
            f"音色包打包完成 | voice_id: {voice_id} | "
            f"大小: {file_size / 1024:.1f} KB | 路径: {output_path}"
        )
        return voice_id

    def unpack(
        self,
        voice_id: str,
    ) -> Tuple[torch.Tensor, np.ndarray, int, Dict, Dict]:
        """
        解包指定音色包，返回其中所有内容。

        Args:
            voice_id: 音色的 UUID v4 字符串

        Returns:
            Tuple 包含：
                - torch.Tensor: speaker embedding 向量 (embedding_dim,)
                - np.ndarray: 参考音频数组（float32）
                - int: 参考音频采样率（Hz）
                - Dict: metadata 字典
                - Dict: model_config 字典

        Raises:
            FileNotFoundError: 音色包文件不存在
            ValueError: 文件格式不合法
            IOError: 解包失败
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            raise FileNotFoundError(f"音色包不存在: {voice_id}")

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                # 验证内部结构
                self._validate_zipfile(zf, voice_id)

                # ── 读取 metadata ───────────────────────────────────────────
                with zf.open(f"{voice_id}/metadata.json") as f:
                    metadata = json.load(f)

                # ── 读取 model_config ───────────────────────────────────────
                with zf.open(f"{voice_id}/model_config.json") as f:
                    model_config = json.load(f)

                # ── 读取 speaker embedding ──────────────────────────────────
                with zf.open(f"{voice_id}/speaker_embedding.pt") as f:
                    emb_buf = io.BytesIO(f.read())
                    embedding = torch.load(emb_buf, map_location="cpu")

                # ── 读取参考音频 ────────────────────────────────────────────
                with zf.open(f"{voice_id}/reference_audio.wav") as f:
                    audio_buf = io.BytesIO(f.read())
                    reference_audio, ref_sr = sf.read(audio_buf, dtype="float32")

        except zipfile.BadZipFile as e:
            raise ValueError(f"音色包文件已损坏 ({voice_id}): {e}") from e
        except Exception as e:
            raise IOError(f"音色包解包失败 ({voice_id}): {e}") from e

        logger.info(f"音色包解包成功 | voice_id: {voice_id}")
        return embedding, reference_audio, ref_sr, metadata, model_config

    def validate(self, voice_id: str) -> Tuple[bool, List[str]]:
        """
        验证音色包文件格式是否符合规范（不完整解包内容）。

        Args:
            voice_id: 音色的 UUID v4 字符串

        Returns:
            Tuple[bool, List[str]]: (是否合法, 错误信息列表)
        """
        voicepack_path = self._get_voicepack_path(voice_id)
        errors = []

        if not voicepack_path.exists():
            return False, [f"音色包文件不存在: {voice_id}.voicepack"]

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                # 检查 ZIP 文件完整性
                bad_file = zf.testzip()
                if bad_file:
                    errors.append(f"ZIP 内部文件损坏: {bad_file}")
                    return False, errors

                # 检查内部文件结构
                names = zf.namelist()
                if not names:
                    errors.append("ZIP 文件为空")
                    return False, errors

                # 验证根目录名与 voice_id 一致
                root_dir = names[0].split("/")[0]
                if root_dir != voice_id:
                    errors.append(f"ZIP 内部根目录 ({root_dir}) 与 voice_id ({voice_id}) 不一致")

                # 检查必须文件
                inner_files = {n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1]}
                missing = REQUIRED_FILES - inner_files
                if missing:
                    errors.append(f"缺少必须文件: {missing}")

                # 验证 metadata.json 格式
                if "metadata.json" in inner_files:
                    try:
                        with zf.open(f"{voice_id}/metadata.json") as f:
                            metadata = json.load(f)
                        required_keys = {
                            "voice_id", "voice_name", "created_at", "model_type",
                            "model_version", "language", "sample_count",
                            "total_duration_seconds", "embedding_dim", "quality_score"
                        }
                        missing_keys = required_keys - set(metadata.keys())
                        if missing_keys:
                            errors.append(f"metadata.json 缺少字段: {missing_keys}")
                        if metadata.get("voice_id") != voice_id:
                            errors.append("metadata.json 中的 voice_id 与文件名不一致")
                    except (json.JSONDecodeError, Exception) as e:
                        errors.append(f"metadata.json 解析失败: {e}")

        except zipfile.BadZipFile as e:
            return False, [f"文件不是有效的 ZIP 格式: {e}"]
        except Exception as e:
            return False, [f"验证过程出错: {e}"]

        is_valid = len(errors) == 0
        if is_valid:
            logger.debug(f"音色包格式验证通过: {voice_id}")
        else:
            logger.warning(f"音色包格式验证失败: {voice_id} | 错误: {errors}")

        return is_valid, errors

    def list_all(self) -> List[VoiceInfo]:
        """
        扫描 voicepacks 目录，列出所有已有音色包的元数据。

        Returns:
            List[VoiceInfo]: 按创建时间降序排列的音色信息列表

        Note:
            损坏的音色包文件会被跳过并记录警告日志。
        """
        voiceinfos: List[VoiceInfo] = []

        if not self.voicepacks_dir.exists():
            return []

        for voicepack_file in self.voicepacks_dir.glob("*.voicepack"):
            voice_id = voicepack_file.stem
            try:
                info = self.get_info(voice_id)
                voiceinfos.append(info)
            except Exception as e:
                logger.warning(f"读取音色包元数据失败，跳过 ({voice_id}): {e}")

        # 按创建时间降序排序（最新的在前）
        voiceinfos.sort(key=lambda x: x.created_at, reverse=True)
        logger.info(f"扫描到 {len(voiceinfos)} 个音色包")
        return voiceinfos

    def delete(self, voice_id: str) -> bool:
        """
        删除指定音色包文件。

        Args:
            voice_id: 音色的 UUID v4 字符串

        Returns:
            bool: 删除成功返回 True，文件不存在返回 False

        Raises:
            OSError: 文件删除失败（权限问题等）
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            logger.warning(f"要删除的音色包不存在: {voice_id}")
            return False

        try:
            voicepack_path.unlink()
            logger.info(f"音色包已删除: {voice_id}")
            return True
        except OSError as e:
            raise OSError(f"删除音色包失败 ({voice_id}): {e}") from e

    def get_info(self, voice_id: str) -> VoiceInfo:
        """
        读取音色包的元数据，不解包 embedding 和音频（速度更快）。

        Args:
            voice_id: 音色的 UUID v4 字符串

        Returns:
            VoiceInfo: 音色信息对象

        Raises:
            FileNotFoundError: 音色包不存在
            ValueError: metadata.json 格式错误
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            raise FileNotFoundError(f"音色包不存在: {voice_id}")

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                with zf.open(f"{voice_id}/metadata.json") as f:
                    metadata = json.load(f)
        except zipfile.BadZipFile as e:
            raise ValueError(f"音色包文件已损坏: {voice_id}") from e
        except Exception as e:
            raise ValueError(f"读取元数据失败 ({voice_id}): {e}") from e

        file_size = voicepack_path.stat().st_size

        return VoiceInfo(
            voice_id=metadata["voice_id"],
            voice_name=metadata["voice_name"],
            created_at=metadata["created_at"],
            model_type=metadata["model_type"],
            model_version=metadata["model_version"],
            language=metadata["language"],
            sample_count=metadata["sample_count"],
            total_duration_seconds=metadata["total_duration_seconds"],
            embedding_dim=metadata["embedding_dim"],
            quality_score=metadata["quality_score"],
            voicepack_size_bytes=file_size,
        )

    def get_voicepack_path(self, voice_id: str) -> str:
        """
        获取音色包文件的绝对路径字符串（供外部使用）。

        Args:
            voice_id: 音色的 UUID v4 字符串

        Returns:
            str: .voicepack 文件的绝对路径
        """
        return str(self._get_voicepack_path(voice_id).absolute())

    # ══════════════════════════════════════════════════════════════════════════
    # 私有辅助方法
    # ══════════════════════════════════════════════════════════════════════════

    def _get_voicepack_path(self, voice_id: str) -> Path:
        """
        构造音色包文件路径。

        Args:
            voice_id: 音色 UUID

        Returns:
            Path: 文件路径对象
        """
        return self.voicepacks_dir / f"{voice_id}.voicepack"

    def _validate_zipfile(self, zf: zipfile.ZipFile, voice_id: str) -> None:
        """
        验证已打开的 ZipFile 内部结构是否合法。

        Args:
            zf: 已打开的 ZipFile 对象
            voice_id: 预期的 voice_id

        Raises:
            ValueError: 结构不合法
        """
        names = zf.namelist()
        if not names:
            raise ValueError("音色包 ZIP 文件为空")

        root_dir = names[0].split("/")[0]
        if root_dir != voice_id:
            raise ValueError(
                f"ZIP 内部根目录 ({root_dir}) 与 voice_id ({voice_id}) 不一致"
            )

        inner_files = {n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1]}
        missing = REQUIRED_FILES - inner_files
        if missing:
            raise ValueError(f"音色包缺少必须文件: {missing}")
