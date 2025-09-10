"""Logging utilities for AngleCam."""

import logging
from pathlib import Path
import hydra
from typing import Union

# Configure default logging format
FORMATTER = logging.Formatter(
    "%(asctime)s - Module(%(module)s):Line(%(lineno)d) %(levelname)s - %(message)s"
)


def ensure_path_exists(path: Union[Path, str]) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logger(
    log_file: Path, name: str = "main", level: int = logging.INFO
) -> logging.Logger:
    """Set up and configure a logger with file handler."""
    ensure_path_exists(log_file.parent)

    # Configure file handler
    handler = logging.FileHandler(log_file, mode="a")
    handler.setFormatter(FORMATTER)

    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def clear_hydra_cache() -> None:
    """Clear the Hydra config cache."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def log_separator(logger: logging.Logger) -> None:
    """Add separator lines to the log."""
    separator = "-" * 50
    for _ in range(2):
        logger.info(separator)
