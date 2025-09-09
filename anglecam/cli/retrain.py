#!/usr/bin/env python3
"""
Retraining script for AngleCam models.
"""

import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys

# Add the project root to Python path for imports
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_DIR))

from anglecam.main import AngleCam

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../config", config_name="main")
def retrain(config: DictConfig) -> None:
    """
    Main retraining function.
    
    The script expects these command line arguments:
    - checkpoint_path: Path to the checkpoint to load
    - train_csv: (optional) New training CSV file
    - val_csv: (optional) New validation CSV file
    - freeze_backbone: (optional) Whether to freeze backbone (default: true)
    - reset_head: (optional) Whether to reset head weights (default: false)
    """
    # Get retraining parameters from command line
    checkpoint_path = PROJECT_DIR / config.inference.pretrained_model_path
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided via command line: checkpoint_path=/path/to/checkpoint.pth")
    
    train_csv = config.data.get("train_csv", None)
    val_csv = config.data.get("val_csv", None)
    freeze_backbone = config.model.get("freeze_backbone", True)
    reset_head = config.model.get("reset_head", False)
    
    logger.info("Starting AngleCam retraining")
    logger.info(f"Checkpoint: {checkpoint_path}")
    if train_csv:
        logger.info(f"New training data: {train_csv}")
    if val_csv:
        logger.info(f"New validation data: {val_csv}")
        
    # Load model from checkpoint (this loads the config from checkpoint)
    model = AngleCam.from_checkpoint(str(checkpoint_path))
    
    # Start retraining
    results = model.retrain(
        checkpoint_path=checkpoint_path,
        train_csv=train_csv,
        val_csv=val_csv,
        freeze_backbone=freeze_backbone,
        reset_head=reset_head,
        save_path=PROJECT_DIR / "model_retrained.pth"
    )
    
    # Log results
    best_val_loss = results.get("best_metrics", {}).get("val_loss", "N/A")
    logger.info(f"Retraining completed. Best validation loss: {best_val_loss}")
    
    # Save retrained model
    logger.info(f"Retrained model saved to: {str(PROJECT_DIR / 'model_retrained.pth')}")

if __name__ == "__main__":
    retrain()