"""Logging configuration for MarketRegimeML."""

import logging
import sys
from pathlib import Path
from typing import Optional

from marketregimeml.config_loader import config_loader


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        format_string: Custom format string for log messages
    """
    # Try to load from config
    try:
        logging_config = config_loader.get_logging_config()
        if level is None:
            level = logging_config.get("level", "INFO")
        if log_file is None:
            log_file = logging_config.get("file")
        if format_string is None:
            format_string = logging_config.get(
                "format",
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
    except (ValueError, FileNotFoundError):
        # Use defaults if config not available
        if level is None:
            level = "INFO"
        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(format_string))
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
