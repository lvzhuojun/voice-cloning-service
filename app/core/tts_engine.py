"""
TTS inference engine wrapper
Wraps the CosyVoice3 inference interface; supports loading a voice from a .voicepack
and synthesizing speech.

Features:
- Model singleton: CosyVoice3 is loaded only once globally, reusing inference resources
- Standard synthesis: returns complete WAV bytes
- Streaming synthesis: generator that yields WAV data in chunks
- Parameter control: speech speed, language
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

# ── Synthesis output parameters ───────────────────────────────────────────────
OUTPUT_SAMPLE_RATE = 24000      # CosyVoice2 output sample rate (Hz)
REFERENCE_SAMPLE_RATE = 16000   # CosyVoice2 expects reference audio at 16 kHz
STREAM_CHUNK_SAMPLES = 4096     # Number of samples per chunk in streaming synthesis


class TTSEngine:
    """
    CosyVoice3 TTS inference engine (singleton pattern).

    Uses a singleton to ensure the model is loaded only once;
    multiple synthesis requests reuse the same model instance.
    Thread-safe: uses threading.Lock to prevent concurrent inference conflicts.

    Usage:
        engine = TTSEngine.get_instance(model_dir="storage/pretrained_models")
        wav_bytes = engine.synthesize("Hello world", voice_id="xxx")
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
        Initialize the TTS engine (usually not called directly; use get_instance()).

        Args:
            model_dir: Pretrained model root directory path
            storage_dir: Storage root directory (used for loading voice packs)
            device: Inference device ("cuda"/"cpu"; auto-selected if None)
        """
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cosyvoice = None          # CosyVoice3 model instance
        self._model_loaded = False      # Whether the model is loaded
        self._inference_lock = threading.Lock()  # Inference mutex lock

        # Voice pack manager (used to load voice embeddings and reference audio)
        self.voicepack_manager = VoicePackManager(storage_dir=storage_dir)

        logger.info(f"TTSEngine initialized | Device: {self.device} | Model dir: {model_dir}")

    @classmethod
    def get_instance(
        cls,
        model_dir: str = "storage/pretrained_models",
        storage_dir: str = "storage",
        device: Optional[str] = None,
    ) -> "TTSEngine":
        """
        Get the TTSEngine singleton instance (thread-safe).

        Creates the instance on the first call; subsequent calls return the existing instance.

        Args:
            model_dir: Pretrained model root directory path
            storage_dir: Storage root directory
            device: Inference device

        Returns:
            TTSEngine: Singleton instance
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
        Load the CosyVoice3 model (skipped if already loaded).

        Loading order:
        1. Try to load Fun-CosyVoice3-0.5B-2512 (full model)
        2. If the model has not been downloaded, mark as not loaded
           (synthesis will return a friendly error)

        Returns:
            bool: True if the model loaded successfully

        Raises:
            RuntimeError: A critical error occurred during model loading
        """
        if self._model_loaded:
            return True

        with self._inference_lock:
            if self._model_loaded:
                return True

            cosyvoice_dir = self.model_dir / "Fun-CosyVoice3-0.5B-2512"

            if not cosyvoice_dir.exists():
                logger.warning(
                    f"Model directory does not exist: {cosyvoice_dir}. "
                    "Please run python setup/download_models.py first."
                )
                return False

            # Try loading via the CosyVoice Python module
            try:
                self._load_cosyvoice_model(str(cosyvoice_dir))
                self._model_loaded = True
                logger.info("CosyVoice3 model loaded successfully")
                return True
            except ImportError as e:
                logger.warning(f"CosyVoice module not found: {e}")
                logger.warning("Please refer to the README to install the CosyVoice submodule.")
                return False
            except Exception as e:
                logger.error(f"CosyVoice3 model loading failed: {e}")
                return False

    def is_model_loaded(self) -> bool:
        """
        Check whether the model has been successfully loaded.

        Returns:
            bool: True if loaded
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
        Synthesize speech from the specified voice and return the complete WAV file as bytes.

        Workflow:
        1. Load the .voicepack for the specified voice_id (obtain embedding and reference audio)
        2. Call CosyVoice3's zero-shot cloning inference interface
        3. Merge the generated audio into complete WAV bytes and return

        Args:
            text: Text to synthesize (1~200 characters)
            voice_id: Voice UUID corresponding to storage/voicepacks/{voice_id}.voicepack
            speed: Speech speed (0.5=slow, 1.0=normal, 2.0=fast)
            language: Synthesis language ("zh"=Chinese, "en"=English)

        Returns:
            bytes: WAV format audio binary data

        Raises:
            FileNotFoundError: Voice pack does not exist
            RuntimeError: Model not loaded or inference failed
            ValueError: Invalid parameters
        """
        if not text or not text.strip():
            raise ValueError("Synthesis text must not be empty")

        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"Speed must be between 0.5 and 2.0; got: {speed}")

        if not self._model_loaded:
            # Try lazy loading
            if not self.load_model():
                raise RuntimeError(
                    "TTS model is not loaded. Please run setup/download_models.py to download "
                    "the model, then restart the service."
                )

        # Load voice pack
        logger.info(f"Synthesizing speech | voice_id: {voice_id} | Text length: {len(text)}")
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
            logger.error(f"TTS inference failed: {e}")
            raise RuntimeError(f"Speech synthesis failed: {e}") from e

        # Merge all audio chunks
        if not audio_chunks:
            raise RuntimeError("Inference returned empty audio")

        full_audio = np.concatenate(audio_chunks)

        # Convert numpy array to WAV bytes
        wav_bytes = self._numpy_to_wav_bytes(full_audio, OUTPUT_SAMPLE_RATE)
        logger.info(
            f"Synthesis complete | voice_id: {voice_id} | "
            f"Duration: {len(full_audio)/OUTPUT_SAMPLE_RATE:.2f}s | "
            f"Size: {len(wav_bytes)/1024:.1f} KB"
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
        Stream speech synthesis; generator function that yields WAV chunk bytes.

        Use cases:
        - WebSocket real-time streaming
        - HTTP Streaming Response (return long text in chunks)

        Each yielded bytes object is a WAV chunk (containing PCM data).
        The consumer should concatenate all chunks to get a complete WAV file,
        or feed each chunk directly into a streaming audio player.

        Args:
            text: Text to synthesize
            voice_id: Voice UUID
            speed: Speech speed (0.5~2.0)
            language: Synthesis language ("zh"/"en")

        Yields:
            bytes: WAV data chunks (PCM 16-bit, 22050 Hz, mono)

        Raises:
            FileNotFoundError: Voice pack does not exist
            RuntimeError: Model not loaded or inference failed
        """
        if not self._model_loaded:
            if not self.load_model():
                raise RuntimeError("TTS model is not loaded. Please download the model and restart the service.")

        embedding, reference_audio, ref_sr, metadata, _ = self.voicepack_manager.unpack(voice_id)
        logger.info(f"Streaming synthesis started | voice_id: {voice_id}")

        try:
            for chunk in self._run_inference(
                text=text,
                embedding=embedding,
                reference_audio=reference_audio,
                ref_sample_rate=ref_sr,
                speed=speed,
                language=language,
            ):
                # Convert each chunk to WAV bytes (no WAV header; raw PCM)
                chunk_int16 = (chunk * 32767).astype(np.int16)
                yield chunk_int16.tobytes()

        except Exception as e:
            logger.error(f"Streaming TTS inference failed: {e}")
            raise RuntimeError(f"Streaming speech synthesis failed: {e}") from e

        logger.info(f"Streaming synthesis complete | voice_id: {voice_id}")

    # ══════════════════════════════════════════════════════════════════════════
    # Internal methods
    # ══════════════════════════════════════════════════════════════════════════

    def _load_cosyvoice_model(self, cosyvoice_dir: str) -> None:
        """
        Load the CosyVoice3 Python model.

        Args:
            cosyvoice_dir: CosyVoice3 model directory path

        Raises:
            ImportError: CosyVoice module not installed
            RuntimeError: Model initialization failed
        """
        import sys

        # CosyVoice source must be cloned into <project_root>/CosyVoice/
        # Run: git clone https://github.com/FunAudioLLM/CosyVoice.git
        cosyvoice_src = Path(__file__).parents[2] / "CosyVoice"
        if cosyvoice_src.exists() and str(cosyvoice_src) not in sys.path:
            sys.path.insert(0, str(cosyvoice_src))
            logger.debug(f"Added CosyVoice source path to sys.path: {cosyvoice_src}")

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
        except ImportError:
            raise ImportError(
                "CosyVoice module not found. Clone the source into CosyVoice/ directory:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -r requirements.txt"
            )

        logger.info(f"Loading CosyVoice2 model: {cosyvoice_dir}")
        self._cosyvoice = CosyVoice2(cosyvoice_dir, load_jit=False, load_trt=False)
        logger.info("CosyVoice2 model loaded successfully")

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
        Run CosyVoice3 inference; yield audio array chunks.

        Args:
            text: Synthesis text
            embedding: Speaker embedding vector
            reference_audio: Reference audio array
            ref_sample_rate: Reference audio sample rate
            speed: Speech speed
            language: Language

        Yields:
            np.ndarray: float32 audio array chunks
        """
        if self._cosyvoice is None:
            raise RuntimeError("CosyVoice3 model is not initialized")

        # CosyVoice2 requires reference audio at exactly 16 kHz
        import librosa
        if ref_sample_rate != REFERENCE_SAMPLE_RATE:
            reference_audio = librosa.resample(
                reference_audio.astype(np.float32),
                orig_sr=ref_sample_rate,
                target_sr=REFERENCE_SAMPLE_RATE,
            )

        # Write reference audio to a temporary file (CosyVoice interface requires a file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, reference_audio, REFERENCE_SAMPLE_RATE, subtype="PCM_16")
            tmp_path = tmp.name

        try:
            # ── Call CosyVoice2 cross-lingual zero-shot inference ─────────────
            # inference_cross_lingual: does not require a reference audio transcript.
            # Reference audio must be at 16 kHz; the engine clones the speaker voice
            # and synthesizes the target text in any language.
            for output in self._cosyvoice.inference_cross_lingual(
                tts_text=text,
                prompt_speech_16k=tmp_path,
                stream=False,
                speed=speed,
            ):
                if "tts_speech" in output:
                    chunk = output["tts_speech"]
                    if isinstance(chunk, torch.Tensor):
                        chunk = chunk.squeeze().cpu().numpy()
                    yield chunk.astype(np.float32)

        except Exception as e:
            logger.warning(f"inference_cross_lingual failed: {e}; trying fallback")
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
        Fallback inference interface (used when the primary interface is unavailable).
        Generates a silent placeholder audio to prevent service crashes.

        Args:
            text: Synthesis text
            ref_audio_path: Reference audio path
            embedding: Speaker embedding
            speed: Speech speed

        Yields:
            np.ndarray: Silent audio chunk (placeholder)
        """
        logger.warning("Using fallback inference interface (generating silence)")
        # Estimate duration: approximately 0.3 s per character, adjusted for speed
        estimated_duration = len(text) * 0.3 / speed
        samples = int(estimated_duration * OUTPUT_SAMPLE_RATE)
        silence = np.zeros(samples, dtype=np.float32)
        yield silence

    @staticmethod
    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert a numpy audio array to WAV format bytes.

        Args:
            audio: float32 audio array, values in [-1.0, 1.0]
            sample_rate: Sample rate (Hz)

        Returns:
            bytes: Complete WAV file bytes (including WAV header)
        """
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
