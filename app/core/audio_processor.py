"""
音频预处理模块
负责将用户上传的各种格式音频转换为模型可用的标准格式，
并进行质量检测、降噪等预处理操作。
"""

from __future__ import annotations

import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa
import soundfile as sf

from app.models.schemas import QualityReport, BatchProcessReport

logger = logging.getLogger(__name__)

# ── 音频标准参数 ───────────────────────────────────────────────────────────────
# CosyVoice3 要求的标准采样率
STANDARD_SAMPLE_RATE = 22050
# 提取 embedding 时用于质量检测的采样率（16kHz SNR 估算更准确）
ANALYSIS_SAMPLE_RATE = 16000
# 高通滤波截止频率（Hz），去除低频噪声（如空调、振动）
HIGHPASS_CUTOFF_HZ = 80
# 静音检测阈值（RMS能量，低于此值视为静音）
SILENCE_THRESHOLD = 0.01
# 连续静音超过此时长（秒）则标记为"有长静音"
LONG_SILENCE_SECONDS = 2.0
# 信噪比估算：将音频分成若干帧，最低 10% 能量帧视为噪声底
NOISE_PERCENTILE = 10


class AudioProcessor:
    """
    音频预处理器，封装音频质量检测、格式转换和降噪功能。

    使用方式:
        processor = AudioProcessor(min_duration=3, max_duration=60)
        audio, sr = processor.convert_to_standard("input.mp3")
        report = processor.check_quality(audio, sr)
    """

    def __init__(
        self,
        min_duration: float = 3.0,
        max_duration: float = 60.0,
        target_sample_rate: int = STANDARD_SAMPLE_RATE,
    ) -> None:
        """
        初始化音频处理器。

        Args:
            min_duration: 最短允许时长（秒），默认 3 秒
            max_duration: 最长允许时长（秒），默认 60 秒
            target_sample_rate: 目标采样率（Hz），默认 22050
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        logger.info(
            f"AudioProcessor 初始化完成 | 时长限制: {min_duration}~{max_duration}s | "
            f"目标采样率: {target_sample_rate}Hz"
        )

    def convert_to_standard(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        将任意格式的音频转换为标准格式（单声道、目标采样率 WAV）。

        使用 librosa.load() 自动处理格式转换，支持 WAV/MP3/M4A/FLAC/OGG 等。

        Args:
            input_path: 输入音频文件路径
            output_path: 输出 WAV 文件路径（可选，不传则只返回数组不保存）

        Returns:
            Tuple[np.ndarray, int]: (音频数组 float32, 采样率)

        Raises:
            FileNotFoundError: 输入文件不存在
            ValueError: 音频加载失败
            IOError: 文件读取或写入失败
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"音频文件不存在: {input_path}")

        try:
            # librosa.load 自动处理：
            # 1. 格式解码（MP3/M4A 需要系统 ffmpeg）
            # 2. 重采样到目标采样率
            # 3. 立体声转单声道（取平均）
            # 4. 归一化到 [-1.0, 1.0] float32
            audio, sr = librosa.load(
                input_path,
                sr=self.target_sample_rate,
                mono=True,
            )
            logger.debug(f"音频加载成功: {input_path} | 时长: {len(audio)/sr:.2f}s | sr: {sr}Hz")
        except Exception as e:
            raise ValueError(f"音频文件加载失败 ({input_path}): {e}") from e

        # 如果指定了输出路径，保存为 WAV
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, audio, sr, subtype="PCM_16")
                logger.debug(f"标准化音频已保存: {output_path}")
            except Exception as e:
                raise IOError(f"音频文件保存失败 ({output_path}): {e}") from e

        return audio, sr

    def check_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> QualityReport:
        """
        检测音频质量，返回详细质量报告。

        检测项目：
        - 时长是否在允许范围内
        - 信噪比估算（基于能量分布）
        - 长静音段检测
        - 综合质量评分

        Args:
            audio: 音频数组（float32，单声道，[-1, 1]）
            sample_rate: 采样率（Hz）

        Returns:
            QualityReport: 质量报告对象
        """
        warnings = []
        duration = len(audio) / sample_rate

        # ── 时长检测 ────────────────────────────────────────────────────────
        duration_ok = self.min_duration <= duration <= self.max_duration
        if duration < self.min_duration:
            warnings.append(f"音频时长 {duration:.1f}s 过短，建议不低于 {self.min_duration}s")
        elif duration > self.max_duration:
            warnings.append(f"音频时长 {duration:.1f}s 过长，建议不超过 {self.max_duration}s")

        # ── 信噪比估算 ──────────────────────────────────────────────────────
        snr_db = self._estimate_snr(audio, sample_rate)
        if snr_db < 10.0:
            warnings.append(f"信噪比偏低 ({snr_db:.1f}dB)，建议在安静环境中录音")
        elif snr_db < 20.0:
            warnings.append(f"信噪比一般 ({snr_db:.1f}dB)，可能影响音色克隆效果")

        # ── 长静音检测 ──────────────────────────────────────────────────────
        has_long_silence = self._detect_long_silence(audio, sample_rate)
        if has_long_silence:
            warnings.append(f"检测到超过 {LONG_SILENCE_SECONDS}s 的连续静音段，建议裁剪")

        # ── 综合质量评分 ────────────────────────────────────────────────────
        quality_score = self._compute_quality_score(
            duration_ok=duration_ok,
            snr_db=snr_db,
            has_long_silence=has_long_silence,
        )

        return QualityReport(
            duration_seconds=round(duration, 3),
            duration_ok=duration_ok,
            snr_db=round(snr_db, 2),
            quality_score=round(quality_score, 4),
            has_long_silence=has_long_silence,
            sample_rate=sample_rate,
            channels=1,  # 已经转为单声道
            warnings=warnings,
        )

    def denoise_simple(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        简单降噪：高通滤波去除低频噪声（如空调、麦克风低频噪声）。

        使用 librosa 的 butter 高通滤波（截止频率 80Hz），
        低频成分（< 80Hz）在自然语音中通常是噪声而非语音内容。

        Args:
            audio: 输入音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            np.ndarray: 降噪后的音频数组（同形状，float32）
        """
        from scipy import signal as scipy_signal

        # 设计 Butterworth 高通滤波器（4阶）
        nyquist = sample_rate / 2.0
        cutoff_normalized = HIGHPASS_CUTOFF_HZ / nyquist

        # 截止频率必须在 (0, 1) 范围内
        if cutoff_normalized >= 1.0:
            logger.warning(f"采样率 {sample_rate}Hz 过低，跳过高通滤波")
            return audio

        b, a = scipy_signal.butter(4, cutoff_normalized, btype="high")
        denoised = scipy_signal.filtfilt(b, a, audio).astype(np.float32)

        logger.debug(f"高通滤波完成 | 截止频率: {HIGHPASS_CUTOFF_HZ}Hz")
        return denoised

    def select_best_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_duration: float = 10.0,
        step_seconds: float = 1.0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        从长音频中自动截取信噪比最高的片段（滑动窗口法）。

        适用于用户上传了较长的录音，需要自动提取最优片段作为参考音频。

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）
            target_duration: 目标片段时长（秒），默认 10 秒
            step_seconds: 滑动步长（秒），默认 1 秒

        Returns:
            Tuple[np.ndarray, float, float]:
                - 最优音频片段数组
                - 片段开始时间（秒）
                - 片段的信噪比（dB）
        """
        duration = len(audio) / sample_rate

        # 如果音频时长不超过目标，直接返回全段
        if duration <= target_duration:
            snr = self._estimate_snr(audio, sample_rate)
            return audio, 0.0, snr

        window_samples = int(target_duration * sample_rate)
        step_samples = int(step_seconds * sample_rate)

        best_snr = -float("inf")
        best_start = 0

        # 滑动窗口遍历，找信噪比最高的片段
        for start in range(0, len(audio) - window_samples + 1, step_samples):
            segment = audio[start : start + window_samples]
            snr = self._estimate_snr(segment, sample_rate)
            if snr > best_snr:
                best_snr = snr
                best_start = start

        best_segment = audio[best_start : best_start + window_samples]
        best_start_seconds = best_start / sample_rate

        logger.info(
            f"最优片段选取完成 | 起始: {best_start_seconds:.1f}s | "
            f"时长: {target_duration}s | SNR: {best_snr:.1f}dB"
        )
        return best_segment, best_start_seconds, best_snr

    def process_batch(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, int]], BatchProcessReport]:
        """
        批量处理多个音频文件：转换格式 + 质量检测 + 降噪。

        Args:
            input_paths: 输入音频文件路径列表
            output_dir: 处理后文件的保存目录（可选）

        Returns:
            Tuple[Dict, BatchProcessReport]:
                - 字典：{原文件名: (音频数组, 采样率)}（仅成功处理的文件）
                - 批量处理汇总报告

        Note:
            处理失败的文件会记录在 report.errors 中，不会中断整个批处理。
        """
        processed: Dict[str, Tuple[np.ndarray, int]] = {}
        file_reports: Dict = {}
        errors: Dict[str, str] = {}

        for input_path in input_paths:
            filename = os.path.basename(input_path)
            logger.info(f"处理音频: {filename}")

            try:
                # 1. 格式转换
                if output_dir:
                    stem = Path(input_path).stem
                    out_path = os.path.join(output_dir, f"{stem}_processed.wav")
                else:
                    out_path = None

                audio, sr = self.convert_to_standard(input_path, out_path)

                # 2. 简单降噪
                audio = self.denoise_simple(audio, sr)

                # 3. 质量检测
                report = self.check_quality(audio, sr)
                file_reports[filename] = report
                processed[filename] = (audio, sr)

                logger.info(
                    f"  ✓ {filename} | 时长: {report.duration_seconds:.1f}s | "
                    f"SNR: {report.snr_db:.1f}dB | 质量分: {report.quality_score:.2f}"
                )

            except Exception as e:
                errors[filename] = str(e)
                logger.warning(f"  ✗ {filename} 处理失败: {e}")

        # ── 计算汇总统计 ────────────────────────────────────────────────────
        success_count = len(processed)
        failed_count = len(errors)
        total_duration = sum(r.duration_seconds for r in file_reports.values())
        avg_quality = (
            sum(r.quality_score for r in file_reports.values()) / success_count
            if success_count > 0
            else 0.0
        )

        report = BatchProcessReport(
            total_files=len(input_paths),
            success_count=success_count,
            failed_count=failed_count,
            total_duration_seconds=round(total_duration, 3),
            average_quality_score=round(avg_quality, 4),
            file_reports=file_reports,
            errors=errors,
        )

        logger.info(
            f"批量处理完成 | 共 {len(input_paths)} 个文件 | "
            f"成功: {success_count} | 失败: {failed_count}"
        )
        return processed, report

    # ══════════════════════════════════════════════════════════════════════════
    # 私有辅助方法
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_snr(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        估算信噪比（Signal-to-Noise Ratio，单位 dB）。

        方法：将音频分为短帧，计算每帧 RMS 能量。
        将最低 10% 能量的帧视为噪声底，其余帧视为信号。
        SNR = 10 * log10(signal_power / noise_power)

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            float: 估算的 SNR（dB），典型范围 0~40dB
        """
        # 帧长：25ms，步长：10ms
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        # 计算短时 RMS 能量
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        if len(rms) == 0 or np.all(rms == 0):
            return 0.0

        # 过滤掉能量极低的帧（纯静音帧不参与计算）
        rms_nonzero = rms[rms > 1e-8]
        if len(rms_nonzero) == 0:
            return 0.0

        # 噪声能量 = 最低 NOISE_PERCENTILE% 帧的能量均值
        noise_rms = np.percentile(rms_nonzero, NOISE_PERCENTILE)
        # 信号能量 = 最高 50% 帧的能量均值（信号主体）
        signal_rms = np.percentile(rms_nonzero, 75)

        if noise_rms < 1e-10:
            return 40.0  # 噪声极低，返回上限

        snr_db = 20.0 * np.log10(signal_rms / noise_rms + 1e-10)
        return float(np.clip(snr_db, -10.0, 60.0))

    def _detect_long_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """
        检测音频中是否存在超过阈值时长的连续静音段。

        Args:
            audio: 音频数组（float32）
            sample_rate: 采样率（Hz）

        Returns:
            bool: 存在长静音段返回 True
        """
        # 帧级 RMS 能量
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # 静音帧判断
        silence_mask = rms < SILENCE_THRESHOLD
        # 连续静音帧数阈值
        max_silence_frames = int(LONG_SILENCE_SECONDS / (hop_length / sample_rate))

        # 计算最长连续静音段长度
        max_consecutive = 0
        current_consecutive = 0
        for is_silence in silence_mask:
            if is_silence:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive > max_silence_frames

    def _compute_quality_score(
        self,
        duration_ok: bool,
        snr_db: float,
        has_long_silence: bool,
    ) -> float:
        """
        综合计算质量评分（0.0~1.0）。

        评分规则：
        - 时长不达标：-0.3 扣分
        - 有长静音：-0.1 扣分
        - SNR 贡献：0~0.7（按 SNR 线性映射到 0~40dB 范围）

        Args:
            duration_ok: 时长是否合规
            snr_db: 信噪比（dB）
            has_long_silence: 是否有长静音

        Returns:
            float: 质量评分（0.0~1.0）
        """
        score = 0.0

        # SNR 分数：SNR 40dB 对应满分 0.7
        snr_clamped = float(np.clip(snr_db, 0.0, 40.0))
        score += (snr_clamped / 40.0) * 0.7

        # 时长合规加分
        if duration_ok:
            score += 0.2

        # 无长静音加分
        if not has_long_silence:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))
