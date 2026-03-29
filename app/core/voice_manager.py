"""
VoiceManager - manages trained GPT-SoVITS voice model storage.

Each voice is stored in storage/models/{voice_id}/ and contains:
  - {voice_id}_gpt.ckpt       : GPT (s1) checkpoint
  - {voice_id}_sovits.pth     : SoVITS (s2) generator weights
  - metadata.json             : voice metadata
  - reference.wav             : short reference audio for inference
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import VoiceInfo


class VoiceManager:
    """
    Manages voice model storage in storage/models/{voice_id}/.

    All paths are relative to the project working directory unless absolute.
    """

    def __init__(self, storage_dir: str = "storage") -> None:
        """
        Initialize VoiceManager.

        Args:
            storage_dir: Root storage directory (default: "storage")
        """
        self.storage_dir = Path(storage_dir)
        self.models_dir = self.storage_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Directory helpers
    # -------------------------------------------------------------------------

    def create_voice_dir(self, voice_id: str) -> Path:
        """
        Create the directory for a voice model.

        Args:
            voice_id: UUID string for the voice

        Returns:
            Path: the created directory
        """
        voice_dir = self.models_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        return voice_dir

    def get_voice_dir(self, voice_id: str) -> Path:
        """Return the directory path for a voice (does not check existence)."""
        return self.models_dir / voice_id

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    def save_metadata(self, voice_id: str, metadata: dict) -> None:
        """
        Write metadata.json for a voice.

        Args:
            voice_id: Voice UUID string
            metadata: Dictionary matching the metadata.json schema
        """
        voice_dir = self.get_voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)
        meta_path = voice_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def get_metadata(self, voice_id: str) -> dict:
        """
        Read metadata.json for a voice.

        Args:
            voice_id: Voice UUID string

        Returns:
            dict: metadata dictionary

        Raises:
            FileNotFoundError: if metadata.json does not exist
        """
        meta_path = self.models_dir / voice_id / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Voice metadata not found for voice_id={voice_id}. "
                f"Expected at: {meta_path}"
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # List / query
    # -------------------------------------------------------------------------

    def list_all(self) -> List[dict]:
        """
        Scan storage/models/ and return metadata for all voices.

        Returns:
            List[dict]: list of metadata dicts sorted by created_at descending
        """
        results: List[dict] = []
        if not self.models_dir.exists():
            return results

        for entry in self.models_dir.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                results.append(meta)
            except Exception:
                # Skip corrupt metadata files
                continue

        # Sort by created_at descending (newest first)
        results.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return results

    def voice_exists(self, voice_id: str) -> bool:
        """
        Check whether a voice exists (has a metadata.json).

        Args:
            voice_id: Voice UUID string

        Returns:
            bool
        """
        return (self.models_dir / voice_id / "metadata.json").exists()

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------

    def delete(self, voice_id: str) -> bool:
        """
        Delete a voice and all its files.

        Args:
            voice_id: Voice UUID string

        Returns:
            bool: True if deleted, False if the voice did not exist
        """
        voice_dir = self.models_dir / voice_id
        if not voice_dir.exists():
            return False
        shutil.rmtree(voice_dir)
        return True

    # -------------------------------------------------------------------------
    # Model paths
    # -------------------------------------------------------------------------

    def get_model_paths(self, voice_id: str) -> Dict[str, Path]:
        """
        Return expected paths for a voice's model files.

        Args:
            voice_id: Voice UUID string

        Returns:
            dict with keys: "gpt", "sovits", "dir", "reference", "metadata"
        """
        voice_dir = self.models_dir / voice_id
        return {
            "dir": voice_dir,
            "gpt": voice_dir / f"{voice_id}_gpt.ckpt",
            "sovits": voice_dir / f"{voice_id}_sovits.pth",
            "reference": voice_dir / "reference.wav",
            "metadata": voice_dir / "metadata.json",
        }

    # -------------------------------------------------------------------------
    # Build schema object
    # -------------------------------------------------------------------------

    def get_info(self, voice_id: str) -> VoiceInfo:
        """
        Get VoiceInfo schema object for a voice.

        Args:
            voice_id: Voice UUID string

        Returns:
            VoiceInfo

        Raises:
            FileNotFoundError: if voice does not exist
        """
        meta = self.get_metadata(voice_id)
        return VoiceInfo(**meta)

    # -------------------------------------------------------------------------
    # Build metadata dict (helper for trainer)
    # -------------------------------------------------------------------------

    @staticmethod
    def build_metadata(
        voice_id: str,
        voice_name: str,
        language: str = "zh",
        audio_duration_seconds: float = 0.0,
        training_epochs_gpt: int = 0,
        training_epochs_sovits: int = 0,
    ) -> dict:
        """
        Build a metadata dict in the canonical format.

        Args:
            voice_id: UUID string
            voice_name: User-provided name
            language: Language code
            audio_duration_seconds: Total training audio duration
            training_epochs_gpt: GPT training epochs completed
            training_epochs_sovits: SoVITS training epochs completed

        Returns:
            dict matching metadata.json schema
        """
        return {
            "voice_id": voice_id,
            "voice_name": voice_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": language,
            "audio_duration_seconds": audio_duration_seconds,
            "training_epochs_gpt": training_epochs_gpt,
            "training_epochs_sovits": training_epochs_sovits,
            "gpt_model_file": f"{voice_id}_gpt.ckpt",
            "sovits_model_file": f"{voice_id}_sovits.pth",
            "base_model_version": "GPT-SoVITS v2",
        }
