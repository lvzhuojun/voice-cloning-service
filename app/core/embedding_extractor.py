"""
Speaker Embedding extraction module
Key improvement: supports weighted fusion of multiple audio segments for more stable speaker feature vectors.

Compared with the original CosyVoice, the key improvements in this module are:
1. Extract embeddings from multiple reference audio clips (3~10 segments) individually
2. Compute normalized weights based on the SNR of each segment
3. Fuse all embeddings via weighted averaging for a more stable speaker representation
4. Automatically select the most suitable audio segment as the inference reference
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

# ── Standard dimension of CosyVoice3 speaker embeddings ──────────────────────
# Fun-CosyVoice3-0.5B-2512 uses CAM++ to extract 192-dimensional speaker embeddings
COSYVOICE_EMBEDDING_DIM = 192

# Target duration range for the optimal reference audio (seconds)
REFERENCE_MIN_DURATION = 5.0
REFERENCE_MAX_DURATION = 15.0


class EmbeddingExtractor:
    """
    Speaker embedding extractor.

    Wraps the CosyVoice3 CAM++ speaker encoder and supports:
    - Single-segment embedding extraction
    - Multi-segment weighted fusion
    - Automatic selection of the optimal reference audio

    Usage:
        extractor = EmbeddingExtractor(model_dir="storage/pretrained_models")
        embedding, weights = extractor.extract_and_fuse(audio_dict, sample_rate)
    """

    def __init__(
        self,
        model_dir: str = "storage/pretrained_models",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the embedding extractor and load the CosyVoice3 model.

        Args:
            model_dir: Pretrained model root directory path
            device: Inference device ("cuda"/"cpu"; auto-selected if None)

        Raises:
            RuntimeError: Model loading failed
        """
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cosyvoice_model = None  # Lazy load
        self._campplus_model = None   # CAM++ speaker encoder
        self._feature_extractor = None

        logger.info(f"EmbeddingExtractor initialized | Device: {self.device} | Model dir: {model_dir}")

    def _load_model(self) -> None:
        """
        Lazily load the CosyVoice3 model (loaded on first call to avoid slow startup).

        Attempt order:
        1. Load the full CosyVoice3 model (including CAM++) from storage/pretrained_models
        2. If the model has not been downloaded, use the bundled ONNX CAM++ model (fallback)
        3. If neither is available, fall back to spectral-feature-based approximate embeddings

        Raises:
            RuntimeError: All loading methods failed
        """
        if self._campplus_model is not None:
            return  # Already loaded; skip

        cosyvoice_dir = Path(self.model_dir) / "Fun-CosyVoice3-0.5B-2512"
        campplus_onnx = cosyvoice_dir / "campplus.onnx"

        # ── Method 1: extract embeddings using the ONNX CAM++ model ──────────
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
                logger.info(f"CAM++ ONNX model loaded successfully: {campplus_onnx}")
                return
            except Exception as e:
                logger.warning(f"ONNX loading failed; trying fallback: {e}")

        # ── Method 2: try loading via the CosyVoice Python module ────────────
        try:
            self._load_cosyvoice_model(cosyvoice_dir)
            return
        except Exception as e:
            logger.warning(f"CosyVoice model loading failed; using approximate embeddings: {e}")

        # ── Method 3: fall back to approximate embeddings (for dev/testing; no pretrained model needed) ──
        logger.warning(
            "No pretrained model found; using MFCC-based approximate embeddings "
            "(limited quality — please download the model for production use)"
        )
        self._campplus_model = "fallback"

    def _load_cosyvoice_model(self, cosyvoice_dir: Path) -> None:
        """
        Try to load the full model via the CosyVoice Python module.

        Args:
            cosyvoice_dir: CosyVoice3 model directory path

        Raises:
            ImportError: CosyVoice module not installed
            RuntimeError: Model loading failed
        """
        try:
            # If CosyVoice is installed from source, add its path to sys.path
            import sys
            cosyvoice_src = Path(__file__).parents[3] / "CosyVoice"
            if cosyvoice_src.exists():
                sys.path.insert(0, str(cosyvoice_src))

            from cosyvoice.cli.cosyvoice import CosyVoice
            self._cosyvoice_model = CosyVoice(str(cosyvoice_dir))
            self._campplus_model = "cosyvoice"
            logger.info("CosyVoice3 full model loaded successfully")
        except ImportError as e:
            raise ImportError(f"CosyVoice module not found: {e}")

    def _load_feature_extractor(self) -> None:
        """
        Load the Kaldi/torchaudio feature extractor (used for feature computation before ONNX inference).
        """
        try:
            import torchaudio.compliance.kaldi as kaldi
            self._feature_extractor = kaldi
            logger.debug("Kaldi feature extractor loaded successfully")
        except ImportError:
            logger.debug("torchaudio not installed; will use librosa for feature extraction")
            self._feature_extractor = None

    def extract_single(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Extract a speaker embedding vector from a single audio segment.

        Args:
            audio: Audio array (float32, mono)
            sample_rate: Sample rate (Hz); 16000 or 22050 recommended

        Returns:
            torch.Tensor: float32 embedding vector with shape (embedding_dim,)

        Raises:
            RuntimeError: Model not loaded or inference failed
        """
        self._load_model()

        if self._campplus_model == "cosyvoice":
            return self._extract_via_cosyvoice(audio, sample_rate)
        elif self._campplus_model == "fallback":
            return self._extract_fallback(audio, sample_rate)
        else:
            # ONNX mode
            return self._extract_via_onnx(audio, sample_rate)

    def extract_and_fuse(
        self,
        audio_quality_pairs: List[Tuple[np.ndarray, int, QualityReport]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Core improvement: extract embeddings from multiple audio segments and fuse them
        with quality-score-based weighting.

        Fusion strategy:
        - Extract speaker embeddings from each segment independently
        - Use each segment's quality_score as its weight (higher SNR = higher weight)
        - Compute a normalized weighted average of all embeddings
        - Apply L2 normalization to the final embedding (consistent with the model's vector space)

        Args:
            audio_quality_pairs: List of (audio array, sample rate, quality report) tuples

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - Fused embedding vector with shape (embedding_dim,)
                - Weight dictionary for each segment: {"segment_0": 0.3, "segment_1": 0.7, ...}

        Raises:
            ValueError: Input is an empty list
            RuntimeError: Embedding extraction failed
        """
        if not audio_quality_pairs:
            raise ValueError("audio_quality_pairs must not be an empty list")

        self._load_model()
        logger.info(f"Starting embedding extraction | Number of audio segments: {len(audio_quality_pairs)}")

        embeddings = []
        quality_scores = []
        weight_records: Dict[str, float] = {}

        for i, (audio, sr, quality_report) in enumerate(audio_quality_pairs):
            segment_key = f"segment_{i}"
            try:
                # Extract single-segment embedding
                emb = self.extract_single(audio, sr)
                embeddings.append(emb)
                quality_scores.append(quality_report.quality_score)
                logger.debug(
                    f"  [{i+1}/{len(audio_quality_pairs)}] Extraction successful | "
                    f"Duration: {quality_report.duration_seconds:.1f}s | "
                    f"Quality: {quality_report.quality_score:.3f}"
                )
            except Exception as e:
                logger.warning(f"  [{i+1}] Extraction failed; skipping: {e}")
                quality_scores.append(0.0)  # Failed segments get weight 0

        if not embeddings:
            raise RuntimeError("Embedding extraction failed for all audio segments")

        # ── Compute normalized weights ────────────────────────────────────────
        # Compute weights only for segments that were successfully extracted
        successful_scores = [quality_scores[i] for i in range(len(audio_quality_pairs)) if i < len(embeddings)]
        total_score = sum(successful_scores)

        if total_score < 1e-8:
            # All segment scores are 0; fall back to equal weights
            weights = [1.0 / len(embeddings)] * len(embeddings)
        else:
            weights = [s / total_score for s in successful_scores]

        # Record per-segment weights
        emb_idx = 0
        for i in range(len(audio_quality_pairs)):
            if emb_idx < len(embeddings):
                weight_records[f"segment_{i}"] = round(weights[emb_idx], 4)
                emb_idx += 1

        # ── Weighted average fusion ───────────────────────────────────────────
        stacked = torch.stack(embeddings, dim=0)  # (n, embedding_dim)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)  # (n,)
        weight_tensor = weight_tensor.unsqueeze(1)                    # (n, 1)

        fused_embedding = (stacked * weight_tensor).sum(dim=0)  # (embedding_dim,)

        # ── L2 normalization (consistent with the model's embedding space) ───
        norm = torch.norm(fused_embedding, p=2)
        if norm > 1e-8:
            fused_embedding = fused_embedding / norm

        logger.info(
            f"Embedding fusion complete | Valid segments: {len(embeddings)} | "
            f"Dimension: {fused_embedding.shape[0]} | Weight distribution: {weight_records}"
        )

        return fused_embedding, weight_records

    def select_reference_audio(
        self,
        audio_quality_pairs: List[Tuple[np.ndarray, int, QualityReport]],
    ) -> Tuple[np.ndarray, int, QualityReport]:
        """
        Select the most suitable segment from multiple audio clips as the inference reference.

        Selection strategy (by priority):
        1. Duration within the REFERENCE_MIN~MAX_DURATION range
        2. Among duration-eligible segments, choose the one with the highest quality_score
        3. If no duration-eligible segment exists, choose the one with the highest quality score overall

        Args:
            audio_quality_pairs: List of (audio array, sample rate, quality report) tuples

        Returns:
            Tuple[np.ndarray, int, QualityReport]: Best (audio, sample rate, quality report)

        Raises:
            ValueError: Input is an empty list
        """
        if not audio_quality_pairs:
            raise ValueError("audio_quality_pairs must not be an empty list")

        # Filter candidates by duration and quality
        candidates_with_duration = [
            (audio, sr, report)
            for audio, sr, report in audio_quality_pairs
            if REFERENCE_MIN_DURATION <= report.duration_seconds <= REFERENCE_MAX_DURATION
        ]

        if candidates_with_duration:
            # Among duration-eligible candidates, choose the one with the highest quality
            best = max(candidates_with_duration, key=lambda x: x[2].quality_score)
        else:
            # No duration-eligible candidates; choose the highest quality overall
            best = max(audio_quality_pairs, key=lambda x: x[2].quality_score)

        logger.info(
            f"Reference audio selected | Duration: {best[2].duration_seconds:.1f}s | "
            f"Quality: {best[2].quality_score:.3f}"
        )
        return best

    # ══════════════════════════════════════════════════════════════════════════
    # Internal inference methods
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_via_cosyvoice(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Extract embedding via the full CosyVoice3 model (highest accuracy).

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            torch.Tensor: (embedding_dim,) embedding vector
        """
        try:
            # Write numpy audio to a temporary file for CosyVoice to read
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                import soundfile as sf
                sf.write(tmp.name, audio, sample_rate)
                tmp_path = tmp.name

            # Use CosyVoice's built-in speaker encoder to extract the embedding
            spk_emb = self._cosyvoice_model.frontend.get_speaker_embedding(tmp_path)
            os.unlink(tmp_path)

            if isinstance(spk_emb, np.ndarray):
                spk_emb = torch.from_numpy(spk_emb)

            return spk_emb.float().squeeze()

        except Exception as e:
            logger.error(f"CosyVoice embedding extraction failed: {e}")
            raise RuntimeError(f"Embedding extraction failed: {e}") from e

    def _extract_via_onnx(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Extract embedding via the ONNX CAM++ model.

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            torch.Tensor: (embedding_dim,) embedding vector
        """
        # Resample to 16 kHz (CAM++ standard input)
        if sample_rate != 16000:
            import librosa
            audio_16k = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        else:
            audio_16k = audio

        # Extract Fbank features (80-dim, consistent with CAM++ training)
        features = self._extract_fbank_features(audio_16k, 16000)  # (T, 80)

        # ONNX inference
        input_name = self._campplus_model.get_inputs()[0].name
        features_input = features[np.newaxis, :, :].astype(np.float32)  # (1, T, 80)

        try:
            outputs = self._campplus_model.run(None, {input_name: features_input})
            embedding = torch.from_numpy(outputs[0]).squeeze()  # (192,)
            return embedding.float()
        except Exception as e:
            raise RuntimeError(f"ONNX inference failed: {e}") from e

    def _extract_fallback(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> torch.Tensor:
        """
        Fallback method: MFCC-based approximate embeddings (for dev/testing only).

        Note: This method is only used to keep the code runnable when no pretrained model
        has been downloaded. Its quality is far inferior to a real speaker encoder.

        Args:
            audio: Audio array (float32)
            sample_rate: Sample rate (Hz)

        Returns:
            torch.Tensor: (COSYVOICE_EMBEDDING_DIM,) approximate embedding
        """
        import librosa

        # Extract MFCC (40-dim) and its delta features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)   # (40, T)
        delta = librosa.feature.delta(mfcc)                                 # (40, T)
        delta2 = librosa.feature.delta(mfcc, order=2)                       # (40, T)

        # Concatenate and apply temporal mean pooling -> (120,)
        features = np.concatenate([
            mfcc.mean(axis=1),
            delta.mean(axis=1),
            delta2.mean(axis=1),
        ])  # (120,)

        # Zero-pad or truncate to COSYVOICE_EMBEDDING_DIM
        if len(features) < COSYVOICE_EMBEDDING_DIM:
            features = np.pad(features, (0, COSYVOICE_EMBEDDING_DIM - len(features)))
        else:
            features = features[:COSYVOICE_EMBEDDING_DIM]

        embedding = torch.from_numpy(features).float()

        # L2 normalization
        norm = torch.norm(embedding, p=2)
        if norm > 1e-8:
            embedding = embedding / norm

        logger.debug("Using MFCC approximate embeddings (fallback mode)")
        return embedding

    def _extract_fbank_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
        num_mel_bins: int = 80,
    ) -> np.ndarray:
        """
        Extract filter bank features (for ONNX CAM++ inference).

        Args:
            audio: Audio array (float32, 16 kHz)
            sample_rate: Sample rate (Hz); should be 16000
            num_mel_bins: Number of Mel filter banks (default 80)

        Returns:
            np.ndarray: Fbank feature matrix with shape (T, num_mel_bins)
        """
        if self._feature_extractor is not None:
            # Use torchaudio/kaldi for extraction (consistent with training)
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
                logger.warning(f"Kaldi fbank extraction failed; falling back to librosa: {e}")

        # Fallback: use librosa melspectrogram
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
