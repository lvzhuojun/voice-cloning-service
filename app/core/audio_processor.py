"""
Audio preprocessing module
Converts user-uploaded audio in various formats to a standard format usable by the model,
and performs quality detection, denoising, and other preprocessing operations.
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

# ── Standard audio parameters ──────────────────────────────────────────────────
# Standard sample rate required by CosyVoice3
STANDARD_SAMPLE_RATE = 22050
# Sample rate used for quality detection when extracting embeddings (16 kHz gives more accurate SNR estimates)
ANALYSIS_SAMPLE_RATE = 16000
# High-pass filter cutoff frequency (Hz); removes low-frequency noise (e.g. AC hum, vibration)
HIGHPASS_CUTOFF_HZ = 80
# Silence detection threshold (RMS energy; values below this are treated as silence)
SILENCE_THRESHOLD = 0.01
# Continuous silence exceeding this duration (seconds) is flagged as "long silence"
LONG_SILENCE_SECONDS = 2.0
# SNR estimation: divide audio into frames; the lowest 10% energy frames are treated as noise floor
NOISE_PERCENTILE = 10


class AudioProcessor:
    """
    Audio preprocessor encapsulating audio quality detection, format conversion, and denoising.

    Usage:
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
        Initialize the audio processor.

        Args:
            min_duration: Minimum allowed duration (seconds), default 3
            max_duration: Maximum allowed duration (seconds), default 60
            target_sample_rate: Target sample rate (Hz), default 22050
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sample_rate = target_sample_rate
        logger.info(
            f"AudioProcessor initialized | Duration limit: {min_duration}~{max_duration}s | "
            f"Target sample rate: {target_sample_rate}Hz"
        )

    def convert_to_standard(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Convert audio in any format to a standard format (mono, target sample rate, WAV).

        Uses librosa.load() to handle format conversion automatically; supports WAV/MP3/M4A/FLAC/OGG etc.

        Args:
            input_path: Input audio file path
            output_path: Output WAV file path (optional; if not provided, returns array only without saving)

        Returns:
            Tuple[np.ndarray, int]: (audio array float32, sample rate)

        Raises:
            FileNotFoundError: Input file does not exist
            ValueError: Audio loading failed
            IOError: File read or write failed
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Audio file does not exist: {input_path}")

        try:
            # librosa.load automatically handles:
            # 1. Format decoding (MP3/M4A requires system ffmpeg)
            # 2. Resampling to the target sample rate
            # 3. Stereo to mono conversion (average channels)
            # 4. Normalization to [-1.0, 1.0] float32
            audio, sr = librosa.load(
                input_path,
                sr=self.target_sample_rate,
                mono=True,
            )
            logger.debug(f"Audio loaded successfully: {input_path} | Duration: {len(audio)/sr:.2f}s | sr: {sr}Hz")
        except Exception as e:
            raise ValueError(f"Failed to load audio file ({input_path}): {e}") from e

        # If an output path is specified, save as WAV
        if output_path:
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sf.write(output_path, audio, sr, subtype="PCM_16")
                logger.debug(f"Standardized audio saved: {output_path}")
            except Exception as e:
                raise IOError(f"Failed to save audio file ({output_path}): {e}") from e

        return audio, sr

    def check_quality(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> QualityReport:
        """
        Inspect audio quality and return a detailed quality report.

        Checks:
        - Whether duration is within the allowed range
        - SNR estimate (based on energy distribution)
        - Long silence segment detection
        - Overall quality score

        Args:
            audio: Audio array (float32, mono, [-1, 1])
            sample_rate: Sample rate (Hz)

        Returns:
            QualityReport: Quality report object
        """
        warnings = []
        duration = len(audio) / sample_rate

        # ── Duration check ───────────────────────────────────────────────────
        duration_ok = self.min_duration <= duration <= self.max_duration
        if duration < self.min_duration:
            warnings.append(f"Audio duration {duration:.1f}s is too short; recommended minimum is {self.min_duration}s")
        elif duration > self.max_duration:
            warnings.append(f"Audio duration {duration:.1f}s is too long; recommended maximum is {self.max_duration}s")

        # ── SNR estimation ───────────────────────────────────────────────────
        snr_db = self._estimate_snr(audio, sample_rate)
        if snr_db < 10.0:
            warnings.append(f"SNR is low ({snr_db:.1f}dB); recording in a quiet environment is recommended")
        elif snr_db < 20.0:
            warnings.append(f"SNR is moderate ({snr_db:.1f}dB); may affect voice cloning quality")

        # ── Long silence detection ───────────────────────────────────────────
        has_long_silence = self._detect_long_silence(audio, sample_rate)
        if has_long_silence:
            warnings.append(f"Continuous silence longer than {LONG_SILENCE_SECONDS}s detected; trimming is recommended")

        # ── Overall quality score ────────────────────────────────────────────
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
            channels=1,  # Already converted to mono
            warnings=warnings,
        )

    def denoise_simple(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Simple denoising: high-pass filter to remove low-frequency noise
        (e.g. AC hum, microphone low-frequency noise).

        Uses librosa's Butterworth high-pass filter (cutoff 80 Hz).
        Low-frequency components (< 80 Hz) in natural speech are typically noise rather than speech content.

        Args:
            audio: Input audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            np.ndarray: Denoised audio array (same shape, float32)
        """
        from scipy import signal as scipy_signal

        # Design a Butterworth high-pass filter (4th order)
        nyquist = sample_rate / 2.0
        cutoff_normalized = HIGHPASS_CUTOFF_HZ / nyquist

        # Cutoff frequency must be in the range (0, 1)
        if cutoff_normalized >= 1.0:
            logger.warning(f"Sample rate {sample_rate}Hz is too low; skipping high-pass filter")
            return audio

        b, a = scipy_signal.butter(4, cutoff_normalized, btype="high")
        denoised = scipy_signal.filtfilt(b, a, audio).astype(np.float32)

        logger.debug(f"High-pass filter applied | Cutoff: {HIGHPASS_CUTOFF_HZ}Hz")
        return denoised

    def select_best_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_duration: float = 10.0,
        step_seconds: float = 1.0,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Automatically extract the segment with the highest SNR from a long audio clip
        using a sliding window approach.

        Useful when the user uploads a long recording and the best segment needs to be
        automatically selected as the reference audio.

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)
            target_duration: Target segment duration (seconds), default 10
            step_seconds: Sliding step size (seconds), default 1

        Returns:
            Tuple[np.ndarray, float, float]:
                - Best audio segment array
                - Segment start time (seconds)
                - Segment SNR (dB)
        """
        duration = len(audio) / sample_rate

        # If audio is shorter than or equal to target duration, return the whole clip
        if duration <= target_duration:
            snr = self._estimate_snr(audio, sample_rate)
            return audio, 0.0, snr

        window_samples = int(target_duration * sample_rate)
        step_samples = int(step_seconds * sample_rate)

        best_snr = -float("inf")
        best_start = 0

        # Slide the window to find the segment with the highest SNR
        for start in range(0, len(audio) - window_samples + 1, step_samples):
            segment = audio[start : start + window_samples]
            snr = self._estimate_snr(segment, sample_rate)
            if snr > best_snr:
                best_snr = snr
                best_start = start

        best_segment = audio[best_start : best_start + window_samples]
        best_start_seconds = best_start / sample_rate

        logger.info(
            f"Best segment selected | Start: {best_start_seconds:.1f}s | "
            f"Duration: {target_duration}s | SNR: {best_snr:.1f}dB"
        )
        return best_segment, best_start_seconds, best_snr

    def process_batch(
        self,
        input_paths: List[str],
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, int]], BatchProcessReport]:
        """
        Batch-process multiple audio files: format conversion + quality detection + denoising.

        Args:
            input_paths: List of input audio file paths
            output_dir: Directory to save processed files (optional)

        Returns:
            Tuple[Dict, BatchProcessReport]:
                - Dictionary: {original filename: (audio array, sample rate)} (successfully processed only)
                - Batch processing summary report

        Note:
            Files that fail processing are recorded in report.errors and do not interrupt the batch.
        """
        processed: Dict[str, Tuple[np.ndarray, int]] = {}
        file_reports: Dict = {}
        errors: Dict[str, str] = {}

        for input_path in input_paths:
            filename = os.path.basename(input_path)
            logger.info(f"Processing audio: {filename}")

            try:
                # 1. Format conversion
                if output_dir:
                    stem = Path(input_path).stem
                    out_path = os.path.join(output_dir, f"{stem}_processed.wav")
                else:
                    out_path = None

                audio, sr = self.convert_to_standard(input_path, out_path)

                # 2. Simple denoising
                audio = self.denoise_simple(audio, sr)

                # 3. Quality detection
                report = self.check_quality(audio, sr)
                file_reports[filename] = report
                processed[filename] = (audio, sr)

                logger.info(
                    f"  ✓ {filename} | Duration: {report.duration_seconds:.1f}s | "
                    f"SNR: {report.snr_db:.1f}dB | Quality: {report.quality_score:.2f}"
                )

            except Exception as e:
                errors[filename] = str(e)
                logger.warning(f"  ✗ {filename} processing failed: {e}")

        # ── Calculate summary statistics ─────────────────────────────────────
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
            f"Batch processing complete | {len(input_paths)} files total | "
            f"Succeeded: {success_count} | Failed: {failed_count}"
        )
        return processed, report

    # ══════════════════════════════════════════════════════════════════════════
    # Private helper methods
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_snr(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Estimate the signal-to-noise ratio (SNR) in dB.

        Method: divide the audio into short frames and compute the RMS energy of each frame.
        The lowest 10% energy frames are treated as the noise floor; the rest are treated as signal.
        SNR = 10 * log10(signal_power / noise_power)

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            float: Estimated SNR (dB), typical range 0~40 dB
        """
        # Frame length: 25 ms, hop length: 10 ms
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)

        # Compute short-time RMS energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        if len(rms) == 0 or np.all(rms == 0):
            return 0.0

        # Filter out near-zero energy frames (pure silence frames are excluded)
        rms_nonzero = rms[rms > 1e-8]
        if len(rms_nonzero) == 0:
            return 0.0

        # Noise energy = mean energy of the lowest NOISE_PERCENTILE% frames
        noise_rms = np.percentile(rms_nonzero, NOISE_PERCENTILE)
        # Signal energy = mean energy of the top 50% frames (main signal)
        signal_rms = np.percentile(rms_nonzero, 75)

        if noise_rms < 1e-10:
            return 40.0  # Noise is very low; return upper bound

        snr_db = 20.0 * np.log10(signal_rms / noise_rms + 1e-10)
        return float(np.clip(snr_db, -10.0, 60.0))

    def _detect_long_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """
        Detect whether the audio contains a continuous silence segment longer than the threshold.

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            bool: True if a long silence segment exists
        """
        # Frame-level RMS energy
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.010 * sample_rate)
        rms = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]

        # Silence frame detection
        silence_mask = rms < SILENCE_THRESHOLD
        # Maximum allowed consecutive silence frames
        max_silence_frames = int(LONG_SILENCE_SECONDS / (hop_length / sample_rate))

        # Find the longest consecutive silence run
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
        Compute an overall quality score (0.0~1.0).

        Scoring rules:
        - Duration out of range: -0.3 penalty
        - Has long silence: -0.1 penalty
        - SNR contribution: 0~0.7 (SNR linearly mapped from 0~40 dB range)

        Args:
            duration_ok: Whether the duration is within the allowed range
            snr_db: Signal-to-noise ratio (dB)
            has_long_silence: Whether long silence exists

        Returns:
            float: Quality score (0.0~1.0)
        """
        score = 0.0

        # SNR score: SNR of 40 dB corresponds to full score 0.7
        snr_clamped = float(np.clip(snr_db, 0.0, 40.0))
        score += (snr_clamped / 40.0) * 0.7

        # Duration compliance bonus
        if duration_ok:
            score += 0.2

        # No long silence bonus
        if not has_long_silence:
            score += 0.1

        return float(np.clip(score, 0.0, 1.0))
