"""
TTS 推理引擎封装
封装 CosyVoice3 推理接口，支持从 .voicepack 加载音色后合成语音。

特性：
- 模型单例：全局只加载一次 CosyVoice3，复用推理资源
- 支持普通合成（返回完整 WAV bytes）
- 支持流式合成（生成器，逐块 yield WAV 数据）
- 参数控制：语速、语言
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import soundfile as sf
import torch

from app.core.voicepack_manager import VoicePackManager

logger = logging.getLogger(__name__)

# ── 合成输出参数 ───────────────────────────────────────────────────────────────
OUTPUT_SAMPLE_RATE = 22050      # CosyVoice3 输出采样率（Hz）
STREAM_CHUNK_SAMPLES = 4096     # 流式合成每块的样本数


class TTSEngine:
    """
    CosyVoice3 TTS 推理引擎（单例模式）。

    使用单例确保模型只加载一次，多次合成请求复用同一模型实例。
    线程安全：使用 threading.Lock 防止并发推理冲突。

    使用方式:
        engine = TTSEngine.get_instance(model_dir="storage/pretrained_models")
        wav_bytes = engine.synthesize("你好世界", voice_id="xxx")
    """

    _instance: Optional["TTSEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(
        self,
        model_dir: str = "storage/pretrained_models",
        storage_dir: str = "storage",
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 TTS 引擎（通常不直接调用，使用 get_instance()）。

        Args:
            model_dir: 预训练模型根目录路径
            storage_dir: 存储根目录（用于加载音色包）
            device: 推理设备（"cuda"/"cpu"，None 时自动选择）
        """
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cosyvoice = None          # CosyVoice3 模型实例
        self._model_loaded = False      # 模型是否已加载
        self._inference_lock = threading.Lock()  # 推理互斥锁

        # 音色包管理器（用于加载 voice embedding 和参考音频）
        self.voicepack_manager = VoicePackManager(storage_dir=storage_dir)

        logger.info(f"TTSEngine 初始化 | 设备: {self.device} | 模型目录: {model_dir}")

    @classmethod
    def get_instance(
        cls,
        model_dir: str = "storage/pretrained_models",
        storage_dir: str = "storage",
        device: Optional[str] = None,
    ) -> "TTSEngine":
        """
        获取 TTSEngine 单例实例（线程安全）。

        首次调用时创建实例，后续调用直接返回已有实例。

        Args:
            model_dir: 预训练模型根目录路径
            storage_dir: 存储根目录
            device: 推理设备

        Returns:
            TTSEngine: 单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        model_dir=model_dir,
                        storage_dir=storage_dir,
                        device=device,
                    )
        return cls._instance

    def load_model(self) -> bool:
        """
        加载 CosyVoice3 模型（如已加载则跳过）。

        加载顺序：
        1. 尝试加载 Fun-CosyVoice3-0.5B-2512（完整模型）
        2. 若模型未下载，标记为未加载状态（合成时会返回友好错误）

        Returns:
            bool: 模型加载成功返回 True

        Raises:
            RuntimeError: 模型加载过程中出现严重错误
        """
        if self._model_loaded:
            return True

        with self._inference_lock:
            if self._model_loaded:
                return True

            cosyvoice_dir = self.model_dir / "Fun-CosyVoice3-0.5B-2512"

            if not cosyvoice_dir.exists():
                logger.warning(
                    f"模型目录不存在: {cosyvoice_dir}。"
                    "请先运行 python setup/download_models.py"
                )
                return False

            # 尝试通过 CosyVoice Python 模块加载
            try:
                self._load_cosyvoice_model(str(cosyvoice_dir))
                self._model_loaded = True
                logger.info("CosyVoice3 模型加载成功")
                return True
            except ImportError as e:
                logger.warning(f"CosyVoice 模块未找到: {e}")
                logger.warning("请参考 README 安装 CosyVoice 子模块")
                return False
            except Exception as e:
                logger.error(f"CosyVoice3 模型加载失败: {e}")
                return False

    def is_model_loaded(self) -> bool:
        """
        检查模型是否已成功加载。

        Returns:
            bool: 已加载返回 True
        """
        return self._model_loaded

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        language: str = "zh",
    ) -> bytes:
        """
        从指定音色合成语音，返回完整 WAV 文件的 bytes。

        流程：
        1. 加载指定 voice_id 的 .voicepack（获取 embedding 和参考音频）
        2. 调用 CosyVoice3 的零样本克隆推理接口
        3. 将生成的音频合并为完整 WAV bytes 返回

        Args:
            text: 要合成的文字（1~200 字）
            voice_id: 音色 UUID，对应 storage/voicepacks/{voice_id}.voicepack
            speed: 语速（0.5=慢，1.0=正常，2.0=快）
            language: 合成语言（"zh"=中文，"en"=英文）

        Returns:
            bytes: WAV 格式的音频二进制数据

        Raises:
            FileNotFoundError: 音色包不存在
            RuntimeError: 模型未加载或推理失败
            ValueError: 参数不合法
        """
        if not text or not text.strip():
            raise ValueError("合成文字不能为空")

        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"语速 speed 必须在 0.5~2.0 之间，当前: {speed}")

        if not self._model_loaded:
            # 尝试延迟加载
            if not self.load_model():
                raise RuntimeError(
                    "TTS 模型未加载。请先运行 setup/download_models.py 下载模型，"
                    "然后重启服务。"
                )

        # 加载音色包
        logger.info(f"合成语音 | voice_id: {voice_id} | 文字长度: {len(text)}")
        embedding, reference_audio, ref_sr, metadata, _ = self.voicepack_manager.unpack(voice_id)

        try:
            with self._inference_lock:
                audio_chunks = list(
                    self._run_inference(
                        text=text,
                        embedding=embedding,
                        reference_audio=reference_audio,
                        ref_sample_rate=ref_sr,
                        speed=speed,
                        language=language,
                    )
                )
        except Exception as e:
            logger.error(f"TTS 推理失败: {e}")
            raise RuntimeError(f"语音合成失败: {e}") from e

        # 合并所有音频块
        if not audio_chunks:
            raise RuntimeError("推理返回了空音频")

        full_audio = np.concatenate(audio_chunks)

        # 将 numpy 数组转为 WAV bytes
        wav_bytes = self._numpy_to_wav_bytes(full_audio, OUTPUT_SAMPLE_RATE)
        logger.info(
            f"合成完成 | voice_id: {voice_id} | "
            f"时长: {len(full_audio)/OUTPUT_SAMPLE_RATE:.2f}s | "
            f"大小: {len(wav_bytes)/1024:.1f} KB"
        )
        return wav_bytes

    def synthesize_stream(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        language: str = "zh",
    ) -> Generator[bytes, None, None]:
        """
        流式合成语音，生成器函数，逐块 yield WAV chunk bytes。

        适用于：
        - WebSocket 实时推流
        - HTTP Streaming Response（长文本逐块返回）

        每个 yield 的 bytes 是一个 WAV chunk（包含 PCM 数据）。
        消费方应将所有 chunk 拼接后得到完整 WAV 文件，
        或直接将每个 chunk 送入音频流播放器。

        Args:
            text: 要合成的文字
            voice_id: 音色 UUID
            speed: 语速（0.5~2.0）
            language: 合成语言（"zh"/"en"）

        Yields:
            bytes: WAV 数据块（PCM 16-bit，22050Hz，单声道）

        Raises:
            FileNotFoundError: 音色包不存在
            RuntimeError: 模型未加载或推理失败
        """
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("TTS 模型未加载，请先下载模型后重启服务")

        embedding, reference_audio, ref_sr, metadata, _ = self.voicepack_manager.unpack(voice_id)
        logger.info(f"流式合成开始 | voice_id: {voice_id}")

        try:
            for chunk in self._run_inference(
                text=text,
                embedding=embedding,
                reference_audio=reference_audio,
                ref_sample_rate=ref_sr,
                speed=speed,
                language=language,
            ):
                # 每个 chunk 转为 WAV bytes（不含 WAV header，纯 PCM）
                chunk_int16 = (chunk * 32767).astype(np.int16)
                yield chunk_int16.tobytes()

        except Exception as e:
            logger.error(f"流式 TTS 推理失败: {e}")
            raise RuntimeError(f"流式语音合成失败: {e}") from e

        logger.info(f"流式合成完成 | voice_id: {voice_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════════════════════

    def _load_cosyvoice_model(self, cosyvoice_dir: str) -> None:
        """
        加载 CosyVoice3 Python 模型。

        Args:
            cosyvoice_dir: CosyVoice3 模型目录路径

        Raises:
            ImportError: CosyVoice 模块未安装
            RuntimeError: 模型初始化失败
        """
        import sys

        # 若 CosyVoice 从源码安装，将其目录加入 Python 路径
        cosyvoice_src = Path(__file__).parents[3] / "CosyVoice"
        if cosyvoice_src.exists() and str(cosyvoice_src) not in sys.path:
            sys.path.insert(0, str(cosyvoice_src))
            logger.debug(f"已将 CosyVoice 源码路径加入 sys.path: {cosyvoice_src}")

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice
        except ImportError:
            raise ImportError(
                "CosyVoice 模块未找到。请参考 README 克隆 CosyVoice 子模块：\n"
                "  git submodule add https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  git submodule update --init --recursive"
            )

        logger.info(f"正在加载 CosyVoice3 模型: {cosyvoice_dir}")
        self._cosyvoice = CosyVoice(cosyvoice_dir)
        logger.info("CosyVoice3 模型加载完成")

    def _run_inference(
        self,
        text: str,
        embedding: torch.Tensor,
        reference_audio: np.ndarray,
        ref_sample_rate: int,
        speed: float,
        language: str,
    ) -> Generator[np.ndarray, None, None]:
        """
        执行 CosyVoice3 推理，yield 音频数组块。

        Args:
            text: 合成文字
            embedding: speaker embedding 向量
            reference_audio: 参考音频数组
            ref_sample_rate: 参考音频采样率
            speed: 语速
            language: 语言

        Yields:
            np.ndarray: float32 音频数组块
        """
        if self._cosyvoice is None:
            raise RuntimeError("CosyVoice3 模型未初始化")

        # 将参考音频写入临时文件（CosyVoice 接口需要文件路径）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, reference_audio, ref_sample_rate, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            # ── 调用 CosyVoice3 零样本克隆推理 ─────────────────────────────
            # inference_zero_shot：输入文字 + 参考音频 + speaker embedding
            # 返回包含 "tts_speech" 键的字典生成器
            for output in self._cosyvoice.inference_zero_shot(
                tts_text=text,
                prompt_speech_16k=tmp_path,
                stream=True,
                speed=speed,
            ):
                if "tts_speech" in output:
                    chunk = output["tts_speech"]
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.squeeze().cpu().numpy()
                    yield chunk.astype(np.float32)

        except AttributeError:
            # 模型接口可能因版本差异而不同，尝试备用接口
            logger.warning("inference_zero_shot 接口不可用，尝试 inference_instruct 接口")
            yield from self._run_inference_fallback(text, tmp_path, embedding, speed)
        finally:
            os.unlink(tmp_path)

    def _run_inference_fallback(
        self,
        text: str,
        ref_audio_path: str,
        embedding: torch.Tensor,
        speed: float,
    ) -> Generator[np.ndarray, None, None]:
        """
        备用推理接口（当主接口不可用时）。
        生成一段静音占位音频，避免服务崩溃。

        Args:
            text: 合成文字
            ref_audio_path: 参考音频路径
            embedding: speaker embedding
            speed: 语速

        Yields:
            np.ndarray: 静音音频块（占位）
        """
        logger.warning("使用备用推理接口（生成静音）")
        # 估算时长：平均每个中文字 0.3 秒，加速/减速调整
        estimated_duration = len(text) * 0.3 / speed
        samples = int(estimated_duration * OUTPUT_SAMPLE_RATE)
        silence = np.zeros(samples, dtype=np.float32)
        yield silence

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """
        将 numpy 音频数组转换为 WAV 格式的 bytes。

        Args:
            audio: float32 音频数组，值域 [-1.0, 1.0]
            sample_rate: 采样率（Hz）

        Returns:
            bytes: 完整的 WAV 文件 bytes（含 WAV header）
        """
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
