# anglecam/inference/predictor.py

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from PIL import Image

from .._data.transforms import create_transform_pipeline


class AngleCamPredictor:
    """Inference pipeline for AngleCam models."""

    def __init__(self, model, config, device: str, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger

        # Setup transforms
        self.transform = create_transform_pipeline(config, mode="test")

        # Put model in eval mode
        self.model.eval()

    def predict(
        self,
        input_data: Union[str, List[str], Path],
        reference_mean: Optional[Union[float, List[float], Dict[str, float]]] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict leaf angles from image(s).

        Args:
            input_data: Single image path, list of paths, or directory
            reference_mean: Optional reference mean angle(s) for comparison.
                          - Single float: applied to single image
                          - List of floats: applied to list of images (must match length)
                          - Dict[str, float]: maps image paths to reference values
                          - None: no reference comparison

        Returns:
            Prediction results with optional reference comparison
        """
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)

            if input_path.is_file():
                # Single image
                ref_val = (
                    reference_mean
                    if isinstance(reference_mean, (float, int, type(None)))
                    else None
                )
                return self._predict_single_image(input_path, ref_val)

            elif input_path.is_dir():
                # Directory of images
                image_paths = list(input_path.glob("*.png")) + list(
                    input_path.glob("*.jpg")
                )
                return [
                    self._predict_single_image(
                        img_path,
                        self._get_reference_for_path(img_path, reference_mean, idx),
                    )
                    for idx, img_path in enumerate(image_paths)
                ]
            else:
                raise ValueError(f"Invalid input path: {input_path}")

        elif isinstance(input_data, list):
            # List of image paths
            return [
                self._predict_single_image(
                    Path(img_path),
                    self._get_reference_for_path(Path(img_path), reference_mean, idx),
                )
                for idx, img_path in enumerate(input_data)
            ]

        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _get_reference_for_path(
        self,
        img_path: Path,
        reference_mean: Optional[Union[float, List[float], Dict[str, float]]],
        idx: int,
    ) -> Optional[float]:
        """Extract reference value for a specific image path."""
        if reference_mean is None:
            return None
        elif isinstance(reference_mean, (float, int)):
            return float(reference_mean)
        elif isinstance(reference_mean, list):
            return float(reference_mean[idx]) if idx < len(reference_mean) else None
        elif isinstance(reference_mean, dict):
            return reference_mean.get(str(img_path)) or reference_mean.get(
                img_path.name
            )
        else:
            return None

    def _predict_single_image(
        self, image_path: Path, reference_mean: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predict angles for a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Apply transforms
            if hasattr(self.transform, "__call__"):
                # Albumentations transform
                transformed = self.transform(image=np.array(image))
                image_tensor = transformed["image"].unsqueeze(0).to(self.device)
            else:
                # Torchvision transforms
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = outputs.squeeze().cpu().numpy()

            # Convert to angle statistics
            angle_stats = self._probabilities_to_stats(probabilities)

            # Build result dictionary
            result = {
                "image_path": str(image_path),
                "predicted_mean_leaf_angle": angle_stats["mean_angle"],
                "angle_distribution": probabilities.tolist(),
            }

            # Add reference comparison if provided
            if reference_mean is not None:
                result["reference_mean_leaf_angle"] = float(reference_mean)
                result["prediction_error"] = angle_stats["mean_angle"] - float(
                    reference_mean
                )
                result["absolute_error"] = abs(
                    angle_stats["mean_angle"] - float(reference_mean)
                )

            return result

        except Exception as e:
            self.logger.error(f"Error predicting {image_path}: {str(e)}")
            error_result = {"image_path": str(image_path), "error": str(e)}
            if reference_mean is not None:
                error_result["reference_mean_leaf_angle"] = float(reference_mean)
            return error_result

    def _probabilities_to_stats(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Convert probability distribution to angle statistics."""
        # Create angle bins
        num_bins = len(probabilities)
        angles = np.linspace(0, 90, num_bins)

        # Calculate statistics
        mean_angle = np.sum(probabilities * angles)
        variance = np.sum(probabilities * (angles - mean_angle) ** 2)
        std_angle = np.sqrt(variance)

        # Calculate angle range (95% confidence interval)
        cumulative = np.cumsum(probabilities)
        lower_idx = (
            np.where(cumulative >= 0.025)[0][0]
            if len(np.where(cumulative >= 0.025)[0]) > 0
            else 0
        )
        upper_idx = (
            np.where(cumulative >= 0.975)[0][0]
            if len(np.where(cumulative >= 0.975)[0]) > 0
            else num_bins - 1
        )

        return {
            "mean_angle": float(mean_angle),
            "variance": float(variance),
            "std_angle": float(std_angle),
        }
