"""
统一日志配置模块
提供结构化、格式统一的日志记录器，供整个项目使用。
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    配置全局日志格式和级别。

    日志格式：[时间] [级别] [模块名] 消息内容
    同时输出到：控制台（stdout）+ 可选的日志文件

    Args:
        level: 日志级别字符串（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 日志文件路径（可选，None 则只输出到控制台）
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 根 logger 配置
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 清除已有 handlers（避免重复配置时日志重复输出）
    root_logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # 文件 handler（可选）
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # 降低第三方库的日志噪音
    for noisy_logger in ["uvicorn.access", "httpx", "httpcore"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger 实例。

    Args:
        name: logger 名称，通常使用 __name__

    Returns:
        logging.Logger: logger 实例
    """
    return logging.getLogger(name)
