"""
File operation utility functions
Provides safe file upload handling, temporary directory management, and other common utilities.
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional


ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}


def is_allowed_audio_file(filename: str) -> bool:
    """
    Check whether a filename has an allowed audio format extension.

    Args:
        filename: File name (including extension)

    Returns:
        bool: True if the format is allowed
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS


def create_temp_dir(prefix: str = "voice_") -> str:
    """
    Create a unique temporary directory.

    Args:
        prefix: Directory name prefix

    Returns:
        str: Absolute path of the temporary directory
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_dir(dir_path: str) -> None:
    """
    Safely delete a directory and all its contents (logs a warning on error).

    Args:
        dir_path: Path of the directory to delete
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory ({dir_path}): {e}")


def get_audio_files_in_dir(directory: str) -> List[str]:
    """
    Scan a directory and return a list of paths for all audio files in allowed formats.

    Args:
        directory: Directory path to scan

    Returns:
        List[str]: Absolute paths of audio files (sorted by filename)

    Raises:
        FileNotFoundError: Directory does not exist
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    audio_files = [
        str(p.absolute())
        for p in sorted(dir_path.iterdir())
        if p.is_file() and is_allowed_audio_file(p.name)
    ]
    return audio_files


def safe_filename(name: str) -> str:
    """
    Convert a user-supplied string to a safe filename (removes special characters).

    Args:
        name: Original name

    Returns:
        str: Safe filename (only letters, digits, Chinese characters, hyphens, underscores)
    """
    import re
    # Keep Chinese characters, letters, digits, underscores, hyphens
    safe = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', name)
    return safe[:64]  # Truncate to a reasonable length
