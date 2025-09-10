#!/usr/bin/env python3
"""
Prediction script for AngleCam.
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys
import json

PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from anglecam.main import AngleCam

log = logging.getLogger(__name__)


def resolve_input_path(config: DictConfig, project_dir: Path) -> Path:
    """
    Resolve input path handling both relative and absolute paths.

    Args:
        config: Hydra configuration
        project_dir: Project root directory

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If the resolved path doesn't exist
        ValueError: If the path is neither file nor directory
    """
    # Get the image_path from config (can be overridden via command line)
    image_path_str = config.inference.image_path
    image_path = Path(image_path_str)

    # Handle absolute vs relative paths
    if image_path.is_absolute():
        # User provided absolute path - use as-is
        resolved_path = image_path
        log.info(f"Using absolute path: {resolved_path}")
    else:
        # Relative path - resolve relative to project directory
        resolved_path = project_dir / image_path
        log.info(f"Using relative path: {image_path} -> {resolved_path}")

    # Validate path exists
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_path}")

    # Validate it's either a file or directory
    if not (resolved_path.is_file() or resolved_path.is_dir()):
        raise ValueError(f"Input path must be a file or directory: {resolved_path}")

    return resolved_path


def resolve_output_path(config: DictConfig, project_dir: Path) -> Path:
    """Resolve output path handling both relative and absolute paths."""
    output_path_str = config.inference.output_dir
    output_path = Path(output_path_str)

    if output_path.is_absolute():
        resolved_path = output_path
    else:
        resolved_path = project_dir / output_path

    resolved_path.mkdir(parents=True, exist_ok=True)

    return resolved_path


@hydra.main(version_base=None, config_path="../config", config_name="main")
def predict(config: DictConfig) -> None:
    """
    Main prediction function.
    """
    log.info("Starting AngleCam prediction")

    model_path = PROJECT_DIR / config.inference.pretrained_model_path

    # Load model from checkpoint
    log.info(f"Loading model from: {model_path}")
    model = AngleCam.from_checkpoint(model_path, config)

    try:
        input_dir = resolve_input_path(config, PROJECT_DIR)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Path resolution failed: {e}")
        sys.exit(1)

    log.info(f"Running predictions on: {input_dir}")
    results = model.predict(str(input_dir))

    # Save the results
    try:
        output_path = resolve_output_path(config, PROJECT_DIR)
    except (FileNotFoundError, ValueError) as e:
        log.error(f"Path resolution failed: {e}")
        sys.exit(1)

    with open(output_path / "anglecam_inference.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    predict()
