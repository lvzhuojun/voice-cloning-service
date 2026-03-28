"""
Speaker Embedding 提取模块
核心改进点：支持多段音频加权融合，提取更稳定的说话人特征向量。

与原始 CosyVoice 相比，本模块的关键改进：
1. 从多个参考音频（3~10段）分别提取 embedding
2. 按各段音频的信噪比（SNR）计算归一化权重
3. 加权平均融合所有 embedding，得到更稳定的说话人表示
4. 自动选择最适合推理参考的音频片段
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.audio_processor import AudioProcessor
from app.models.schemas import QualityReport

logger = logging.getLogger(__name__)

# ── CosyVoice3 speaker embedding 的标准维度 ───────────────────────────────────
# Fun-CosyVoice3-0.5B-2512 使用 CAM++ 提取 192 维 speaker embedding
COSYVOICE_EMBEDDING_DIM = 192

# 最优参考音频的目标时长范围（秒）
REFERENCE_MIN_DURATION = 5.0
REFERENCE_MAX_DURATION = 15.0


class EmbeddingExtractor:
    """
    说话人 Embedding 提取器。

    封装 CosyVoice3 的 CAM++ 说话人编码器，支持：
    - 单段音频 embedding 提取
    - 多段音频加权融合
    - 最优参考音频自动选取

    使用方式:
        extractor = EmbeddingExtractor(model_dir="storage/pretrained_models")
        embedding, weights = extractor.extract_and_fuse(audio_dict, sample_rate)
    """

    def __init__(
        self,
        model_dir: str = "storage/pretrained_models",
        device: Optional[str] = None,
    ) -> None:
        """
        初始化 Embedding 提取器，加载 CosyVoice3 模型。

        Args:
            model_dir: 预训练模型根目录路径
            device: 推理设备（"cuda"/"cpu"，None 时自动选择）

        Raises:
            RuntimeError: 模型加载失败
        """
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cosyvoice_model = None  # 延迟加载
        self._campplus_model = None   # CAM++ speaker encoder
        self._feature_extractor = None

        logger.info(f"EmbeddingExtractor 初始化 | 设备: {self.device} | 模型目录: {model_dir}")

    def _load_model(self) -> None:
        """
        延迟加载 CosyVoice3 模型（首次调用时才加载，避免启动过慢）。

        尝试顺序：
        1. 从 storage/pretrained_models 加载完整 CosyVoice3 模型（含 CAM++）
        2. 若模型未下载，使用预置的 ONNX CAM++ 模型（备用）
        3. 若均不可用，降级到基于 spectral 特征的近似 embedding

        Raises:
            RuntimeError: 所有加载方式均失败
        """
        if self._campplus_model is not None:
            return  # 已加载，跳过

        cosyvoice_dir = Path(self.model_dir) / "Fun-CosyVoice3-0.5B-2512"
        campplus_onnx = cosyvoice_dir / "campplus.onnx"

        # ── 方式1：使用 ONNX CAM++ 模型提取 embedding ──────────────────────
        if campplus_onnx.exists():
            try:
                import onnxruntime as ort
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                self._campplus_model = ort.InferenceSession(
                    str(campplus_onnx),
                    providers=providers,
                )
                self._load_feature_extractor()
                logger.info(f"CAM++ ONNX 模型加载成功: {campplus_onnx}")
                return
            except Exception as e:
                logger.warning(f"ONNX 加载失败，尝试备用方案: {e}")

        # ── 方式2：尝试通过 CosyVoice Python 模块加载 ──────────────────────
        try:
            self._load_cosyvoice_model(cosyvoice_dir)
            return
        except Exception as e:
            logger.warning(f"CosyVoice 模型加载失败，使用近似 embedding: {e}")

        # ── 方式3：降级到近似 embedding（开发/测试用，不依赖预训练模型）──────
        logger.warning(
            "未找到预训练模型，使用基于 MFCC 的近似 Embedding（效果有限，请下载模型后使用）"
        )
        self._campplus_model = "fallback"

    def _load_cosyvoice_model(self, cosyvoice_dir: Path) -> None:
        """
        尝试通过 CosyVoice Python 模块加载完整模型。

        Args:
            cosyvoice_dir: CosyVoice3 模型目录路径

        Raises:
            ImportError: CosyVoice 模块未安装
            RuntimeError: 模型加载失败
        """
        try:
            # CosyVoice 从源码安装时，需要将其路径加入 sys.path
            import sys
            cosyvoice_src = Path(__file__).parents[3] / "CosyVoice"
            if cosyvoice_src.exists():
                sys.path.insert(0, str(cosyvoice_src))

            from cosyvoice.cli.cosyvoice import CosyVoice
            self._cosyvoice_model = CosyVoice(str(cosyvoice_dir))
            self._campplus_model = "cosyvoice"
            logger.info("CosyVoice3 完整模型加载成功")
        except ImportError as e:
            raise ImportError(f"CosyVoice 模块未找到: {e}")

    def _load_feature_extractor(self) -> None:
        """
        加载 Kaldi/torchaudio 特征提取器（用于 ONNX 推理前的特征计算）。
        """
        try:
            import torchaudio.compliance.kaldi as kaldi
            self._feature_extractor = kaldi
            logger.debug("Kaldi 特征提取器加载成功")
        except ImportError:
            logger.debug("torchaudio 未安装，将使用 librosa 提取特征")
            self._feature_extractor = None

    def extract_single(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        从单段音频提取 speaker embedding 向量。

        Args:
            audio: 音频数组（float32，单声道）
            sample_rate: 采样率（Hz），建议 16000 或 22050

        Returns:
            torch.Tensor: 形状为 (embedding_dim,) 的 float32 embedding 向量

        Raises:
            RuntimeError: 模型未加载或推理失败
        """
        self._load_model()

        if self._campplus_model == "cosyvoice":
            return self._extract_via_cosyvoice(audio, sample_rate)
        elif self._campplus_model == "fallback":
            return self._extract_fallback(audio, sample_rate)
        else:
            # ONNX 模式
            return self._extract_via_onnx(audio, sample_rate)

    def extract_and_fuse(
        self,
        audio_quality_pairs: List[Tuple[np.ndarray, int, QualityReport]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        核心改进：从多段音频提取 embedding 后，按质量评分加权融合。

        融合策略：
        - 对每段音频独立提取 speaker embedding
        - 以各段的 quality_score 作为权重（SNR 越高权重越大）
        - 对所有 embedding 做归一化加权平均
        - 对最终 embedding 做 L2 归一化（保持与模型一致的向量空间）

        Args:
            audio_quality_pairs: 列表，每元素为 (音频数组, 采样率, 质量报告)

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - 融合后的 embedding 向量，形状 (embedding_dim,)
                - 各片段的权重字典 {"segment_0": 0.3, "segment_1": 0.7, ...}

        Raises:
            ValueError: 输入为空列表
            RuntimeError: embedding 提取失败
        """
        if not audio_quality_pairs:
            raise ValueError("audio_quality_pairs 不能为空列表")

        self._load_model()
        logger.info(f"开始提取 embedding | 音频段数: {len(audio_quality_pairs)}")

        embeddings = []
        quality_scores = []
        weight_records: Dict[str, float] = {}

        for i, (audio, sr, quality_report) in enumerate(audio_quality_pairs):
            segment_key = f"segment_{i}"
            try:
                # 提取单段 embedding
                emb = self.extract_single(audio, sr)
                embeddings.append(emb)
                quality_scores.append(quality_report.quality_score)
                logger.debug(
                    f"  [{i+1}/{len(audio_quality_pairs)}] 提取成功 | "
                    f"时长: {quality_report.duration_seconds:.1f}s | "
                    f"质量: {quality_report.quality_score:.3f}"
                )
            except Exception as e:
                logger.warning(f"  [{i+1}] 提取失败，跳过: {e}")
                quality_scores.append(0.0)  # 失败的段权重为0

        if not embeddings:
            raise RuntimeError("所有音频段的 embedding 提取均失败")

        # ── 计算归一化权重 ───────────────────────────────────────────────────
        # 只对成功提取 embedding 的段计算权重
        successful_scores = [quality_scores[i] for i in range(len(audio_quality_pairs)) if i < len(embeddings)]
        total_score = sum(successful_scores)

        if total_score < 1e-8:
            # 所有段质量分均为0，退化为均等权重
            weights = [1.0 / len(embeddings)] * len(embeddings)
        else:
            weights = [s / total_score for s in successful_scores]

        # 记录每段权重
        emb_idx = 0
        for i in range(len(audio_quality_pairs)):
            if emb_idx < len(embeddings):
                weight_records[f"segment_{i}"] = round(weights[emb_idx], 4)
                emb_idx += 1

        # ── 加权平均融合 ─────────────────────────────────────────────────────
        stacked = torch.stack(embeddings, dim=0)  # (n, embedding_dim)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)  # (n,)
        weight_tensor = weight_tensor.unsqueeze(1)                    # (n, 1)

        fused_embedding = (stacked * weight_tensor).sum(dim=0)  # (embedding_dim,)

        # ── L2 归一化（保持与模型 embedding 空间一致）───────────────────────
        norm = torch.norm(fused_embedding, p=2)
        if norm > 1e-8:
            fused_embedding = fused_embedding / norm

        logger.info(
            f"Embedding 融合完成 | 有效段: {len(embeddings)} | "
            f"维度: {fused_embedding.shape[0]} | 权重分布: {weight_records}"
        )

        return fused_embedding, weight_records

    def select_reference_audio(
        self,
        audio_quality_pairs: List[Tuple[np.ndarray, int, QualityReport]],
    ) -> Tuple[np.ndarray, int, QualityReport]:
        """
        从多段音频中选取最适合作为推理参考的一段。

        选取策略（按优先级）：
        1. 时长在 REFERENCE_MIN~MAX_DURATION 范围内
        2. 在满足时长条件的段中，选 quality_score 最高的
        3. 若无时长合适的段，则直接选质量最高的

        Args:
            audio_quality_pairs: 列表，每元素为 (音频数组, 采样率, 质量报告)

        Returns:
            Tuple[np.ndarray, int, QualityReport]: 最优 (音频, 采样率, 质量报告)

        Raises:
            ValueError: 输入为空列表
        """
        if not audio_quality_pairs:
            raise ValueError("audio_quality_pairs 不能为空列表")

        # 按时长和质量联合筛选
        candidates_with_duration = [
            (audio, sr, report)
            for audio, sr, report in audio_quality_pairs
            if REFERENCE_MIN_DURATION <= report.duration_seconds <= REFERENCE_MAX_DURATION
        ]

        if candidates_with_duration:
            # 从时长合适的候选中选质量最高的
            best = max(candidates_with_duration, key=lambda x: x[2].quality_score)
        else:
            # 无时长合适的候选，直接选质量最高的
            best = max(audio_quality_pairs, key=lambda x: x[2].quality_score)

        logger.info(
            f"参考音频选取完成 | 时长: {best[2].duration_seconds:.1f}s | "
            f"质量: {best[2].quality_score:.3f}"
        )
        return best

    # ══════════════════════════════════════════════════════════════════════════
    # 内部推理方法
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_via_cosyvoice(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        通过完整 CosyVoice3 模型提取 embedding（最高精度）。

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            torch.Tensor: (embedding_dim,) embedding 向量
        """
        try:
            # 将 numpy 音频写入临时文件供 CosyVoice 读取
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, sample_rate)
                tmp_path = tmp.name

            # 使用 CosyVoice 内置的 speaker encoder 提取
            spk_emb = self._cosyvoice_model.frontend.get_speaker_embedding(tmp_path)
            os.unlink(tmp_path)

            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)

            return spk_emb.float().squeeze()

        except Exception as e:
            logger.error(f"CosyVoice embedding 提取失败: {e}")
            raise RuntimeError(f"embedding 提取失败: {e}") from e

    def _extract_via_onnx(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        通过 ONNX CAM++ 模型提取 embedding。

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            torch.Tensor: (embedding_dim,) embedding 向量
        """
        # 重采样到 16kHz（CAM++ 标准输入）
        if sample_rate != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio

        # 提取 Fbank 特征（80 dim，与 CAM++ 训练时一致）
        features = self._extract_fbank_features(audio_16k, 16000)  # (T, 80)

        # ONNX 推理
        input_name = self._campplus_model.get_inputs()[0].name
        features_input = features[np.newaxis, :, :].astype(np.float32)  # (1, T, 80)

        try:
            outputs = self._campplus_model.run(None, {input_name: features_input})
            embedding = torch.from_numpy(outputs[0]).squeeze()  # (192,)
            return embedding.float()
        except Exception as e:
            raise RuntimeError(f"ONNX 推理失败: {e}") from e

    def _extract_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        降级方案：基于 MFCC 的近似 embedding（开发/测试用）。

        注意：此方案仅用于在未下载预训练模型时保持代码可运行，
        效果远不如真正的 speaker encoder。

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            torch.Tensor: (COSYVOICE_EMBEDDING_DIM,) 近似 embedding
        """
        import librosa

        # 提取 MFCC（40 维）及其 delta 特征
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)   # (40, T)
        delta = librosa.feature.delta(mfcc)                                 # (40, T)
        delta2 = librosa.feature.delta(mfcc, order=2)                       # (40, T)

        # 拼接并时间维度均值池化 -> (120,)
        features = np.concatenate([
            mfcc.mean(axis=1),
            delta.mean(axis=1),
            delta2.mean(axis=1),
        ])  # (120,)

        # 补零或截断到 COSYVOICE_EMBEDDING_DIM
        if len(features) < COSYVOICE_EMBEDDING_DIM:
            features = np.pad(features, (0, COSYVOICE_EMBEDDING_DIM - len(features)))
        else:
            features = features[:COSYVOICE_EMBEDDING_DIM]

        embedding = torch.from_numpy(features).float()

        # L2 归一化
        norm = torch.norm(embedding, p=2)
        if norm > 1e-8:
            embedding = embedding / norm

        logger.debug("使用 MFCC 近似 embedding（降级模式）")
        return embedding

    def _extract_fbank_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_mel_bins: int = 80,
    ) -> np.ndarray:
        """
        提取 Filter Bank 特征（用于 ONNX CAM++ 推理）。

        Args:
            audio: 音频数组（float32，16kHz）
            sample_rate: 采样率（Hz），应为 16000
            num_mel_bins: Mel 滤波器组数量（默认 80）

        Returns:
            np.ndarray: 形状 (T, num_mel_bins) 的 Fbank 特征矩阵
        """
        if self._feature_extractor is not None:
            # 使用 torchaudio/kaldi 提取（与训练时一致）
            try:
                waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
                fbank = self._feature_extractor.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=25,  # ms
                    frame_shift=10,   # ms
                    sample_frequency=sample_rate,
                    use_energy=False,
                )
                return fbank.numpy()  # (T, 80)
            except Exception as e:
                logger.warning(f"Kaldi fbank 提取失败，降级到 librosa: {e}")

        # 降级：使用 librosa 的 melspectrogram
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=num_mel_bins,
            hop_length=160,   # 10ms at 16kHz
            win_length=400,   # 25ms at 16kHz
            fmin=20,
            fmax=8000,
        )
        log_mel = np.log(mel + 1e-8).T.astype(np.float32)  # (T, 80)
        return log_mel
