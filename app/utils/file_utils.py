"""
文件操作工具函数
提供安全的文件上传、临时目录管理等通用功能。
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
    检查文件名是否为允许的音频格式。

    Args:
        filename: 文件名（含扩展名）

    Returns:
        bool: 允许的格式返回 True
    """
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_AUDIO_EXTENSIONS


def create_temp_dir(prefix: str = "voice_") -> str:
    """
    创建唯一的临时目录。

    Args:
        prefix: 目录名前缀

    Returns:
        str: 临时目录的绝对路径
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_dir(dir_path: str) -> None:
    """
    安全删除目录及其所有内容（出错时仅记录警告）。

    Args:
        dir_path: 要删除的目录路径
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        logger.warning(f"清理临时目录失败 ({dir_path}): {e}")


def get_audio_files_in_dir(directory: str) -> List[str]:
    """
    扫描目录，返回所有允许格式的音频文件路径列表。

    Args:
        directory: 要扫描的目录路径

    Returns:
        List[str]: 音频文件绝对路径列表（按文件名排序）

    Raises:
        FileNotFoundError: 目录不存在
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"路径不是目录: {directory}")

    audio_files = [
        str(p.absolute())
        for p in sorted(dir_path.iterdir())
        if p.is_file() and is_allowed_audio_file(p.name)
    ]
    return audio_files


def safe_filename(name: str) -> str:
    """
    将用户输入的字符串转换为安全的文件名（去除特殊字符）。

    Args:
        name: 原始名称

    Returns:
        str: 安全的文件名（仅保留字母、数字、中文、连字符、下划线）
    """
    import re
    # 保留中文字符、字母、数字、下划线、连字符
    safe = re.sub(r'[^\w\u4e00-\u9fff\-]', '_', name)
    return safe[:64]  # 截断到合理长度
