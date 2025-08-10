import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union, List, Optional, Any, Dict

from .models.base import DINOv2_AngleCam
from .training.trainer import AngleCamTrainer
from .inference.predictor import AngleCamPredictor
from .utils.logging import setup_logger
from .utils.reproducibility import setup_reproducibility


class AngleCam:
    """
    Main AngleCam interface for leaf angle distribution estimation.

    Provides unified API for training and prediction.
    """

    def __init__(self, config: DictConfig):
        """Initialize AngleCam with Hydra configuration."""
        self.config = config
        self.model = None
        self.trainer = None
        self.predictor = None
        self.logger = None

        # Setup core components
        self._setup_logging()
        self._setup_reproducibility()
        self._setup_device()

    def _setup_logging(self) -> None:
        """Setup logging system."""

        log_file = Path(self.config.output_dir) / "anglecam.log"
        self.logger = setup_logger(log_file, name="anglecam")
        self.logger.info("AngleCam initialized with Hydra configuration")
        self.logger.info(f"Output directory: {self.config.output_dir}")

    def _setup_reproducibility(self) -> None:
        """Setup reproducibility settings."""
        seed = self.config.seed
        setup_reproducibility(seed)
        self.logger.info(f"Reproducibility setup with seed: {seed}")

    def _setup_device(self) -> None:
        """Setup compute device."""
        device_config = self.config.device
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
        self.logger.info(f"Using device: {self.device}")

    def _get_model(self) -> DINOv2_AngleCam:
        """Get or create model instance."""
        if self.model is None:
            self.model = DINOv2_AngleCam(self.config).to(self.device)
            self.logger.info(f"Created model: {self.config.model.name}")
        return self.model

    def train(self, **kwargs) -> Dict[str, Any]:
        """
        Train the AngleCam model.

        Args:
            **kwargs: Training parameters that override config

        Returns:
            Training results dictionary
        """
        self.logger.info("Starting training process")

        # Merge kwargs with config for overrides
        if kwargs:
            override_config = OmegaConf.create(kwargs)
            training_config = OmegaConf.merge(self.config, override_config)
        else:
            training_config = self.config

        # Create trainer if not exists
        if self.trainer is None:
            self.trainer = AngleCamTrainer(
                config=training_config, device=self.device, logger=self.logger
            )

        # Get model
        model = self._get_model()

        # Run training
        results = self.trainer.train(model, **kwargs)

        # Update our model reference with the trained model
        self.model = results["model"]

        self.logger.info("Training completed successfully")
        return results

    def predict(
        self,
        input_data: Union[str, List[str], Path],
        reference_mean: Optional[Union[float, List[float], Dict[str, float]]] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict leaf angles from image(s).

        Args:
            input_data: Single image path, list of paths, or directory

        Returns:
            Prediction results with angle distributions and statistics
        """
        self.logger.info(f"Starting prediction on: {input_data}")

        # Ensure we have a model
        if self.model is None:
            raise ValueError(
                "No trained model available. Train a model first or load from checkpoint."
            )

        # Create predictor if not exists
        if self.predictor is None:
            self.predictor = AngleCamPredictor(
                model=self.model,
                config=self.config,
                device=self.device,
                logger=self.logger,
            )

        # Run prediction
        results = self.predictor.predict(input_data, reference_mean=reference_mean)

        self.logger.info("Prediction completed successfully")
        return results

    def save(self, save_path: str) -> None:
        """Save trained model and configuration."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and config
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": OmegaConf.to_yaml(self.config),  # Save as YAML string
                "model_class": self.model.__class__.__name__,
            },
            save_path,
        )

        self.logger.info(f"Model saved to: {save_path}")

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, config_overrides: Optional[Dict] = None
    ) -> "AngleCam":
        """Load trained AngleCam from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Load config from checkpoint
        if isinstance(checkpoint["config"], str):
            config = OmegaConf.create(checkpoint["config"])
        else:
            config = checkpoint["config"]

        # Apply any config overrides
        if config_overrides:
            override_config = OmegaConf.create(config_overrides)
            config = OmegaConf.merge(config, override_config)

        # Create instance
        instance = cls(config)

        # Load model
        instance.model = DINOv2_AngleCam(config)
        instance.model.load_state_dict(checkpoint["model_state_dict"])

        return instance
