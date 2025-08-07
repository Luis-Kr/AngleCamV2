#!/usr/bin/env python3
"""
Training script for AngleCam models.
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys

# Add the project root to Python path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from anglecam.main import AngleCam

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def train(config: DictConfig) -> None:
    """
    Main training function.

    Args:
        config: Hydra configuration object automatically injected by Hydra
    """
    log.info("Starting AngleCam training")

    # Create AngleCam instance
    model = AngleCam(config)

    # Train model
    log.info(f"Starting training with model: {config.model.name}")
    results = model.train()

    # Log results
    best_val_loss = results.get("best_metrics", {}).get("val_loss", "N/A")
    log.info(f"Training completed. Best validation loss: {best_val_loss}")

    # Save model
    save_path = Path.cwd() / "model_final.pth"
    model.save(str(save_path))
    log.info(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train()
