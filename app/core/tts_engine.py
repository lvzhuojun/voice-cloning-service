"""
GPT-SoVITS TTS inference engine backed by TTS_infer_pack.TTS.TTS.

Each voice keeps an independent cached TTS instance because the GPT and SoVITS
weight paths differ per voice. Model loading is expensive, so instances are
created lazily and reused until evicted.
"""

from __future__ import annotations

import gc
import logging
import sys
import threading
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TTSEngine:
    """Singleton TTS engine with per-voice GPT-SoVITS caches."""

    _instance: Optional["TTSEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        gptsovits_dir: str = "GPT-SoVITS",
        pretrained_dir: str = "storage/pretrained_models/GPT-SoVITS",
        models_dir: str = "storage/models",
        device: Optional[str] = None,
    ) -> None:
        self.gptsovits_dir = Path(gptsovits_dir).resolve()
        self.gptsovits_inner = self.gptsovits_dir / "GPT_SoVITS"
        self.pretrained_dir = Path(pretrained_dir).resolve()
        self.models_dir = Path(models_dir).resolve()
        self.device = device

        self._voice_cache: Dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self._inference_lock = threading.Lock()

        logger.info(
            "TTSEngine initialized | GPT-SoVITS=%s | pretrained=%s | models=%s",
            self.gptsovits_dir,
            self.pretrained_dir,
            self.models_dir,
        )

    @classmethod
    def get_instance(
        cls,
        gptsovits_dir: str = "GPT-SoVITS",
        pretrained_dir: str = "storage/pretrained_models/GPT-SoVITS",
        models_dir: str = "storage/models",
        device: Optional[str] = None,
    ) -> "TTSEngine":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        gptsovits_dir=gptsovits_dir,
                        pretrained_dir=pretrained_dir,
                        models_dir=models_dir,
                        device=device,
                    )
        return cls._instance

    def is_ready(self) -> bool:
        return self.gptsovits_dir.exists() and self.gptsovits_inner.exists()

    def is_model_loaded(self) -> bool:
        return self.is_ready()

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _add_gptsovits_to_path(self) -> None:
        for path in (self.gptsovits_dir, self.gptsovits_inner):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _check_runtime_ready(self) -> None:
        if not self.gptsovits_dir.exists():
            raise RuntimeError(
                f"GPT-SoVITS directory not found: {self.gptsovits_dir}\n"
                "Run: setup\\clone_gptsovits.bat"
            )
        if not self.gptsovits_inner.exists():
            raise RuntimeError(f"GPT_SoVITS package directory not found: {self.gptsovits_inner}")

        required_paths = [
            self.pretrained_dir / "chinese-hubert-base",
            self.pretrained_dir / "chinese-roberta-wwm-ext-large",
        ]
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError(
                "Required pretrained models are missing:\n"
                + "\n".join(missing)
                + "\nRun: conda run -n voice-cloning python setup/download_models.py"
            )

    def _get_voice_paths(self, voice_id: str) -> dict:
        voice_dir = self.models_dir / voice_id
        paths = {
            "voice_dir": voice_dir,
            "gpt": voice_dir / f"{voice_id}_gpt.ckpt",
            "sovits": voice_dir / f"{voice_id}_sovits.pth",
            "reference": voice_dir / "reference.wav",
            "metadata": voice_dir / "metadata.json",
        }

        for name in ("gpt", "sovits", "reference"):
            if not paths[name].exists():
                raise FileNotFoundError(f"{name} file not found for voice {voice_id}: {paths[name]}")

        return paths

    def _get_voice_language(self, voice_id: str, override: Optional[str]) -> str:
        if override:
            return override

        try:
            from app.core.voice_manager import VoiceManager

            metadata = VoiceManager(storage_dir=str(self.models_dir.parent)).get_metadata(voice_id)
            language = str(metadata.get("language", "zh")).strip().lower()
            return language if language in {"zh", "en", "ja", "ko", "yue", "auto", "auto_yue"} else "zh"
        except Exception:
            return "zh"

    def _build_tts_instance(self, voice_id: str) -> dict:
        self._check_runtime_ready()
        self._add_gptsovits_to_path()
        paths = self._get_voice_paths(voice_id)

        device = self._resolve_device()
        is_half = device.startswith("cuda")

        try:
            from TTS_infer_pack.TTS import TTS, TTS_Config
        except Exception as exc:
            raise RuntimeError(f"Failed to import GPT-SoVITS TTS runtime: {exc}") from exc

        config = TTS_Config(
            {
                "custom": {
                    "device": device,
                    "is_half": is_half,
                    "version": "v2",
                    "t2s_weights_path": str(paths["gpt"]),
                    "vits_weights_path": str(paths["sovits"]),
                    "cnhuhbert_base_path": str(self.pretrained_dir / "chinese-hubert-base"),
                    "bert_base_path": str(self.pretrained_dir / "chinese-roberta-wwm-ext-large"),
                }
            }
        )

        logger.info("Loading GPT-SoVITS TTS instance for voice %s on %s", voice_id, device)
        tts = TTS(config)
        return {
            "tts": tts,
            "device": device,
            "gpt_path": paths["gpt"],
            "sovits_path": paths["sovits"],
            "reference_path": paths["reference"],
        }

    def _get_or_create_voice_cache(self, voice_id: str) -> dict:
        with self._cache_lock:
            cached = self._voice_cache.get(voice_id)
            paths = self._get_voice_paths(voice_id)
            if cached is not None:
                if cached["gpt_path"] == paths["gpt"] and cached["sovits_path"] == paths["sovits"]:
                    return cached
                self._dispose_cache_entry(cached)

            cached = self._build_tts_instance(voice_id)
            self._voice_cache[voice_id] = cached
            return cached

    def synthesize(
        self,
        text: str,
        voice_id: str,
        speed: float = 1.0,
        language: Optional[str] = None,
    ) -> bytes:
        if not text or not text.strip():
            raise ValueError("Synthesis text must not be empty")
        if not (0.5 <= speed <= 2.0):
            raise ValueError(f"Speed must be between 0.5 and 2.0; got: {speed}")

        voice_cache = self._get_or_create_voice_cache(voice_id)
        text_lang = self._get_voice_language(voice_id, language)

        inputs = {
            "text": text.strip(),
            "text_lang": text_lang,
            "ref_audio_path": str(voice_cache["reference_path"]),
            "prompt_text": "",
            "prompt_lang": text_lang,
            "speed_factor": speed,
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "batch_size": 1,
            "text_split_method": "cut1",
            "parallel_infer": False,
            "split_bucket": False,
            "return_fragment": False,
            "streaming_mode": False,
        }

        with self._inference_lock:
            try:
                result = next(voice_cache["tts"].run(inputs))
            except StopIteration as exc:
                raise RuntimeError(f"TTS returned no audio for voice {voice_id}") from exc
            except Exception as exc:
                raise RuntimeError(f"Speech synthesis failed for voice {voice_id}: {exc}") from exc

        sample_rate, audio = result
        return self._numpy_to_wav_bytes(audio, int(sample_rate))

    @staticmethod
    def _numpy_to_wav_bytes(audio, sample_rate: int) -> bytes:
        import io
        import numpy as np
        import soundfile as sf

        arr = np.asarray(audio)
        if arr.ndim > 1:
            arr = arr.squeeze()
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / max(np.iinfo(arr.dtype).max, 1)
        else:
            arr = arr.astype(np.float32)
        arr = np.clip(arr, -1.0, 1.0)

        buffer = io.BytesIO()
        sf.write(buffer, arr, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    def _dispose_cache_entry(self, cached: dict) -> None:
        cached.pop("tts", None)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def evict_cache(self, voice_id: str) -> None:
        with self._cache_lock:
            cached = self._voice_cache.pop(voice_id, None)
        if cached is not None:
            self._dispose_cache_entry(cached)
            logger.info("Evicted TTS cache for voice %s", voice_id)

    def clear_cache(self) -> None:
        with self._cache_lock:
            cached_values = list(self._voice_cache.values())
            self._voice_cache.clear()
        for cached in cached_values:
            self._dispose_cache_entry(cached)
        logger.info("Cleared all TTS caches")
