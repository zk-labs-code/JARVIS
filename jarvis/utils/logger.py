"""Logging utility for JARVIS."""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "jarvis",
    log_dir: str = "logs",
    level: str = "INFO",
) -> logging.Logger:
    """Set up and return a configured logger.

    Args:
        name: Logger name.
        log_dir: Directory to store log files.
        level: Logging level string.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"jarvis_{datetime.now().strftime('%Y%m%d')}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_format = logging.Formatter(
        "%(asctime)s | %(name)-12s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "jarvis") -> logging.Logger:
    """Get an existing logger by name.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


# Module-level convenience
_root_logger: logging.Logger | None = None


def init_logging(log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    """Initialize the root JARVIS logger.

    Args:
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        The root JARVIS logger.
    """
    global _root_logger
    _root_logger = setup_logger("jarvis", log_dir, level)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "httpx", "httpcore", "matplotlib"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return _root_logger


def log_separator(logger: logging.Logger, title: str = "") -> None:
    """Log a visual separator line.

    Args:
        logger: Logger instance.
        title: Optional title for the separator.
    """
    if title:
        logger.info(f"{'=' * 20} {title} {'=' * 20}")
    else:
        logger.info("=" * 60)


def log_system_info(logger: logging.Logger) -> None:
    """Log basic system information.

    Args:
        logger: Logger instance.
    """
    import platform

    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Working Dir: {os.getcwd()}")
