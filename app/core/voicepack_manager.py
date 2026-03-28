"""
VoicePack packaging management module
Handles creation, unpacking, validation, and management of .voicepack files.

A .voicepack is a standard ZIP file with a fixed internal structure (see VOICEPACK_FORMAT.md):
    {voice_id}/
    ├── speaker_embedding.pt     # torch.Tensor, speaker embedding vector
    ├── reference_audio.wav      # Optimal reference audio segment
    ├── model_config.json        # Model configuration
    └── metadata.json            # Metadata
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

# ── .voicepack internal structure specification ───────────────────────────────
REQUIRED_FILES = frozenset({
    "speaker_embedding.pt",
    "reference_audio.wav",
    "metadata.json",
    "model_config.json",
})

MODEL_TYPE = "cosyvoice3"
MODEL_VERSION = "Fun-CosyVoice3-0.5B-2512"
REFERENCE_AUDIO_SAMPLE_RATE = 22050  # Reference audio sample rate required by CosyVoice3


class VoicePackManager:
    """
    Voice pack manager encapsulating the full lifecycle of .voicepack files.

    Operations:
    - pack():      Pack embedding, reference audio, and config into a .voicepack
    - unpack():    Unpack and validate a .voicepack file
    - validate():  Validate format only, without unpacking content
    - list_all():  List metadata for all existing voice packs
    - delete():    Delete a specified voice pack
    - get_info():  Read metadata (without unpacking the embedding)

    Usage:
        manager = VoicePackManager(storage_dir="storage")
        voice_id = manager.pack(embedding, ref_audio, sr, metadata_kwargs)
        info = manager.get_info(voice_id)
    """

    def __init__(self, storage_dir: str = "storage") -> None:
        """
        Initialize VoicePackManager.

        Args:
            storage_dir: Storage root directory; voice packs are saved in {storage_dir}/voicepacks/

        Raises:
            OSError: Cannot create storage directory
        """
        self.storage_dir = Path(storage_dir)
        self.voicepacks_dir = self.storage_dir / "voicepacks"

        # Ensure storage directory exists
        try:
            self.voicepacks_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create voice pack storage directory ({self.voicepacks_dir}): {e}") from e

        logger.info(f"VoicePackManager initialized | Storage dir: {self.voicepacks_dir.absolute()}")

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
        Pack the speaker embedding and reference audio into a .voicepack file.

        Packed contents:
        - speaker_embedding.pt: torch.Tensor with shape (embedding_dim,)
        - reference_audio.wav: 16-bit PCM WAV, 22050 Hz, mono
        - metadata.json: Voice metadata
        - model_config.json: Model configuration

        Args:
            embedding: Speaker embedding vector with shape (embedding_dim,)
            reference_audio: Reference audio array (float32)
            reference_sample_rate: Reference audio sample rate (Hz)
            voice_name: User-defined voice name
            language: Primary language ("zh"/"en"), default "zh"
            sample_count: Number of reference audio segments used for training
            total_duration_seconds: Total duration of all reference audio (seconds)
            quality_score: Overall audio quality score (0.0~1.0)
            voice_id: Specified voice_id (UUID v4); auto-generated if not provided

        Returns:
            str: Generated voice_id (UUID v4 string)

        Raises:
            ValueError: Embedding shape is invalid
            IOError: File write failed
        """
        # Parameter validation
        if embedding.ndim != 1:
            raise ValueError(f"embedding must be a 1D tensor; current shape: {embedding.shape}")

        embedding_dim = embedding.shape[0]

        # Generate or use the specified voice_id
        if voice_id is None:
            voice_id = str(uuid.uuid4())

        output_path = self.voicepacks_dir / f"{voice_id}.voicepack"

        logger.info(f"Packing voice pack | voice_id: {voice_id} | Name: {voice_name}")

        try:
            with zipfile.ZipFile(str(output_path), "w", zipfile.ZIP_DEFLATED) as zf:
                # ── Write speaker embedding ───────────────────────────────────
                emb_buf = io.BytesIO()
                torch.save(embedding.cpu().float(), emb_buf)
                zf.writestr(
                    f"{voice_id}/speaker_embedding.pt",
                    emb_buf.getvalue(),
                )

                # ── Write reference audio ─────────────────────────────────────
                # Ensure sample rate meets CosyVoice3 requirements
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

                # ── Write metadata.json ───────────────────────────────────────
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

                # ── Write model_config.json ───────────────────────────────────
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
            raise IOError(f"ZIP packing failed ({output_path}): {e}") from e
        except Exception as e:
            # Delete the incomplete file if packing fails
            if output_path.exists():
                output_path.unlink()
            raise IOError(f"Voice pack packing failed: {e}") from e

        file_size = output_path.stat().st_size
        logger.info(
            f"Voice pack packed successfully | voice_id: {voice_id} | "
            f"Size: {file_size / 1024:.1f} KB | Path: {output_path}"
        )
        return voice_id

    def unpack(
        self,
        voice_id: str,
    ) -> Tuple[torch.Tensor, np.ndarray, int, Dict, Dict]:
        """
        Unpack the specified voice pack and return all its contents.

        Args:
            voice_id: Voice UUID v4 string

        Returns:
            Tuple containing:
                - torch.Tensor: Speaker embedding vector (embedding_dim,)
                - np.ndarray: Reference audio array (float32)
                - int: Reference audio sample rate (Hz)
                - Dict: Metadata dictionary
                - Dict: Model config dictionary

        Raises:
            FileNotFoundError: Voice pack file does not exist
            ValueError: File format is invalid
            IOError: Unpacking failed
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            raise FileNotFoundError(f"Voice pack does not exist: {voice_id}")

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                # Validate internal structure
                self._validate_zipfile(zf, voice_id)

                # ── Read metadata ─────────────────────────────────────────────
                with zf.open(f"{voice_id}/metadata.json") as f:
                    metadata = json.load(f)

                # ── Read model_config ─────────────────────────────────────────
                with zf.open(f"{voice_id}/model_config.json") as f:
                    model_config = json.load(f)

                # ── Read speaker embedding ────────────────────────────────────
                with zf.open(f"{voice_id}/speaker_embedding.pt") as f:
                    emb_buf = io.BytesIO(f.read())
                    embedding = torch.load(emb_buf, map_location="cpu")

                # ── Read reference audio ──────────────────────────────────────
                with zf.open(f"{voice_id}/reference_audio.wav") as f:
                    audio_buf = io.BytesIO(f.read())
                    reference_audio, ref_sr = sf.read(audio_buf, dtype="float32")

        except zipfile.BadZipFile as e:
            raise ValueError(f"Voice pack file is corrupted ({voice_id}): {e}") from e
        except Exception as e:
            raise IOError(f"Voice pack unpacking failed ({voice_id}): {e}") from e

        logger.info(f"Voice pack unpacked successfully | voice_id: {voice_id}")
        return embedding, reference_audio, ref_sr, metadata, model_config

    def validate(self, voice_id: str) -> Tuple[bool, List[str]]:
        """
        Validate whether the voice pack file format conforms to the specification
        (without fully unpacking contents).

        Args:
            voice_id: Voice UUID v4 string

        Returns:
            Tuple[bool, List[str]]: (whether valid, list of error messages)
        """
        voicepack_path = self._get_voicepack_path(voice_id)
        errors = []

        if not voicepack_path.exists():
            return False, [f"Voice pack file does not exist: {voice_id}.voicepack"]

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                # Check ZIP file integrity
                bad_file = zf.testzip()
                if bad_file:
                    errors.append(f"Corrupted file inside ZIP: {bad_file}")
                    return False, errors

                # Check internal file structure
                names = zf.namelist()
                if not names:
                    errors.append("ZIP file is empty")
                    return False, errors

                # Verify that the root directory name matches voice_id
                root_dir = names[0].split("/")[0]
                if root_dir != voice_id:
                    errors.append(f"ZIP internal root dir ({root_dir}) does not match voice_id ({voice_id})")

                # Check required files
                inner_files = {n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1]}
                missing = REQUIRED_FILES - inner_files
                if missing:
                    errors.append(f"Missing required files: {missing}")

                # Validate metadata.json format
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
                            errors.append(f"metadata.json is missing fields: {missing_keys}")
                        if metadata.get("voice_id") != voice_id:
                            errors.append("voice_id in metadata.json does not match the filename")
                    except (json.JSONDecodeError, Exception) as e:
                        errors.append(f"Failed to parse metadata.json: {e}")

        except zipfile.BadZipFile as e:
            return False, [f"File is not a valid ZIP format: {e}"]
        except Exception as e:
            return False, [f"Error during validation: {e}"]

        is_valid = len(errors) == 0
        if is_valid:
            logger.debug(f"Voice pack format validation passed: {voice_id}")
        else:
            logger.warning(f"Voice pack format validation failed: {voice_id} | Errors: {errors}")

        return is_valid, errors

    def list_all(self) -> List[VoiceInfo]:
        """
        Scan the voicepacks directory and list metadata for all existing voice packs.

        Returns:
            List[VoiceInfo]: Voice info list sorted by creation time (newest first)

        Note:
            Corrupted voice pack files are skipped with a warning log entry.
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
                logger.warning(f"Failed to read voice pack metadata; skipping ({voice_id}): {e}")

        # Sort by creation time descending (newest first)
        voiceinfos.sort(key=lambda x: x.created_at, reverse=True)
        logger.info(f"Found {len(voiceinfos)} voice pack(s)")
        return voiceinfos

    def delete(self, voice_id: str) -> bool:
        """
        Delete the specified voice pack file.

        Args:
            voice_id: Voice UUID v4 string

        Returns:
            bool: True if deletion succeeded, False if the file does not exist

        Raises:
            OSError: File deletion failed (permissions issue, etc.)
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            logger.warning(f"Voice pack to delete does not exist: {voice_id}")
            return False

        try:
            voicepack_path.unlink()
            logger.info(f"Voice pack deleted: {voice_id}")
            return True
        except OSError as e:
            raise OSError(f"Failed to delete voice pack ({voice_id}): {e}") from e

    def get_info(self, voice_id: str) -> VoiceInfo:
        """
        Read the metadata of a voice pack without unpacking the embedding or audio (faster).

        Args:
            voice_id: Voice UUID v4 string

        Returns:
            VoiceInfo: Voice information object

        Raises:
            FileNotFoundError: Voice pack does not exist
            ValueError: metadata.json format error
        """
        voicepack_path = self._get_voicepack_path(voice_id)

        if not voicepack_path.exists():
            raise FileNotFoundError(f"Voice pack does not exist: {voice_id}")

        try:
            with zipfile.ZipFile(str(voicepack_path), "r") as zf:
                with zf.open(f"{voice_id}/metadata.json") as f:
                    metadata = json.load(f)
        except zipfile.BadZipFile as e:
            raise ValueError(f"Voice pack file is corrupted: {voice_id}") from e
        except Exception as e:
            raise ValueError(f"Failed to read metadata ({voice_id}): {e}") from e

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
        Get the absolute path string of the voice pack file (for external use).

        Args:
            voice_id: Voice UUID v4 string

        Returns:
            str: Absolute path of the .voicepack file
        """
        return str(self._get_voicepack_path(voice_id).absolute())

    # ══════════════════════════════════════════════════════════════════════════
    # Private helper methods
    # ══════════════════════════════════════════════════════════════════════════

    def _get_voicepack_path(self, voice_id: str) -> Path:
        """
        Construct the voice pack file path.

        Args:
            voice_id: Voice UUID

        Returns:
            Path: File path object
        """
        return self.voicepacks_dir / f"{voice_id}.voicepack"

    def _validate_zipfile(self, zf: zipfile.ZipFile, voice_id: str) -> None:
        """
        Validate the internal structure of an already-opened ZipFile.

        Args:
            zf: Opened ZipFile object
            voice_id: Expected voice_id

        Raises:
            ValueError: Structure is invalid
        """
        names = zf.namelist()
        if not names:
            raise ValueError("Voice pack ZIP file is empty")

        root_dir = names[0].split("/")[0]
        if root_dir != voice_id:
            raise ValueError(
                f"ZIP internal root dir ({root_dir}) does not match voice_id ({voice_id})"
            )

        inner_files = {n.split("/", 1)[1] for n in names if "/" in n and n.split("/", 1)[1]}
        missing = REQUIRED_FILES - inner_files
        if missing:
            raise ValueError(f"Voice pack is missing required files: {missing}")
