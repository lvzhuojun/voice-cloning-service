"""
Unified logging configuration module
Provides a structured, consistently formatted logger for use throughout the project.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Configure global log format and level.

    Log format: [timestamp] [level] [module name] message
    Output destinations: console (stdout) + optional log file

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        log_file: Log file path (optional; if None, output goes to console only)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers (avoid duplicate log output on re-configuration)
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Reduce log noise from third-party libraries
    for noisy_logger in ["uvicorn.access", "httpx", "httpcore"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name; typically use __name__

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
