#!/usr/bin/env python3
"""
Prediction script for AngleCam.
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from anglecam import AngleCam

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../anglecam/config", config_name="inference"
)
def predict(config: DictConfig) -> None:
    """
    Main prediction function.
    """
    log.info("Starting AngleCam prediction")

    # Validate required config
    if not hasattr(config, "model_path") or not hasattr(config, "input_path"):
        raise ValueError("Configuration must specify 'model_path' and 'input_path'")

    # Load model from checkpoint
    log.info(f"Loading model from: {config.model_path}")
    model = AngleCam.from_checkpoint(config.model_path)

    # Run predictions
    log.info(f"Running predictions on: {config.input_path}")
    results = model.predict(config.input_path)

    # Handle results
    if isinstance(results, list):
        log.info(f"Processed {len(results)} images")
    else:
        log.info(f"Predicted angle: {results['predicted_angle']:.2f}°")


if __name__ == "__main__":
    predict()
