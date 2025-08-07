import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
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
        self, input_data: Union[str, List[str], Path]
    ) -> Union[Dict, List[Dict]]:
        """
        Predict leaf angles from image(s).

        Args:
            input_data: Single image path, list of paths, or directory

        Returns:
            Prediction results
        """
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)

            if input_path.is_file():
                # Single image
                return self._predict_single_image(input_path)
            elif input_path.is_dir():
                # Directory of images
                image_paths = list(input_path.glob("*.png")) + list(
                    input_path.glob("*.jpg")
                )
                return [
                    self._predict_single_image(img_path) for img_path in image_paths
                ]
            else:
                raise ValueError(f"Invalid input path: {input_path}")

        elif isinstance(input_data, list):
            # List of image paths
            return [
                self._predict_single_image(Path(img_path)) for img_path in input_data
            ]

        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _predict_single_image(self, image_path: Path) -> Dict[str, Any]:
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

            return {
                "image_path": str(image_path),
                "predicted_mean_leaf_angle": angle_stats["mean_angle"],
                "predicted_variance_leaf_angle": angle_stats["variance"],
                "predicted_std_leaf_angle": angle_stats["std_angle"],
                "angle_distribution": probabilities.tolist(),
                # "confidence": angle_stats["confidence"],
                # "angle_range": angle_stats["angle_range"],
                # "statistics": angle_stats,
            }

        except Exception as e:
            self.logger.error(f"Error predicting {image_path}: {str(e)}")
            return {"image_path": str(image_path), "error": str(e)}

    def _probabilities_to_stats(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Convert probability distribution to angle statistics."""
        # Create angle bins
        num_bins = len(probabilities)
        angles = np.linspace(0, 90, num_bins)

        # Calculate statistics
        mean_angle = np.sum(probabilities * angles)
        variance = np.sum(probabilities * (angles - mean_angle) ** 2)
        std_angle = np.sqrt(variance)

        # Calculate confidence as inverse of entropy
        # entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        # max_entropy = np.log(num_bins)
        # confidence = 1 - (entropy / max_entropy)

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
            # "confidence": float(confidence),
            # "angle_range": (float(angles[lower_idx]), float(angles[upper_idx])),
            # "mode_angle": float(angles[np.argmax(probabilities)]),
        }
