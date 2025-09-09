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

@hydra.main(version_base=None, config_path="../config", config_name="main")
def predict(config: DictConfig) -> None:
    """
    Main prediction function.
    """
    log.info("Starting AngleCam prediction")
    
    output_dir = PROJECT_DIR / "data" / "model" / "final_output" / "best_model_output"
    model_path = output_dir / "checkpoints" / "best_model_val_loss.pth"

    # Load model from checkpoint
    log.info(f"Loading model from: {model_path}")
    model = AngleCam.from_checkpoint(model_path)

    # Run predictions
    #input_dir = PROJECT_DIR / config.input_path
    #input_dir = "/mnt/gsdata/projects/other/anglecam_arbofun/data_2023_arbofun_brinno_timeseries"
    input_dir = "/mnt/gsdata/projects/other/anglecam_arbofun/data_2023_arbofun_brinno_timeseries_missing/339_Thu_occ"
    
    log.info(f"Running predictions on: {input_dir}")
    results = model.predict(str(input_dir))

    # Handle results
    if isinstance(results, list):
        log.info(f"Processed {len(results)} images")
    else:
        log.info(f"Predicted angle: {results['predicted_angle']:.2f}°")

    # Save the results to a JSON file
    output_path = PROJECT_DIR / "data" / "inference" / "arbofun" / "339_Thu_occ_anglecam_inference.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    predict()
