import logging
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Union, List, Optional, Any, Dict
from datetime import datetime
import sys, os

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from anglecam.models.base import DINOv2_AngleCam
from anglecam.training.trainer import AngleCamTrainer
from anglecam.inference.predictor import AngleCamPredictor
from anglecam.utils.logging import setup_logger
from anglecam.utils.reproducibility import setup_reproducibility


class AngleCam:
    """
    Main AngleCam interface for leaf inclination angle distribution estimation.

    Provides unified API for training and prediction.
    """

    def __init__(self, config: DictConfig, mode: str = "training"):
        """Initialize AngleCam."""
        self.config = config
        self.model = None
        self.trainer = None
        self.predictor = None
        self.logger = None
        self.mode = mode
        
        self._setup_project_directories()

        # Setup core components
        self._setup_logging()
        self._setup_reproducibility()
        self._setup_device()
        #self._validate_config()

    def _setup_logging(self) -> None:
        """Setup logging."""
        if self.mode == "inference":
            output_dir = self.config.inference.output_dir
            log_context = "inference"
        else:
            output_dir = self.config.training.output_dir
            log_context = "training"

        log_file = Path(output_dir) / "anglecam.log"
        self.logger = setup_logger(log_file, name="anglecam")
        self.logger.info("::: AngleCam initialized :::")
        self.logger.info(f"Log directory ({log_context}): {output_dir}")

    def _setup_reproducibility(self) -> None:
        """Setup reproducibility settings."""
        seed = self.config.seed
        setup_reproducibility(seed)
        self.logger.info(f"Reproducibility setup with seed: {seed}")
        
    def _setup_project_directories(self) -> None:
        """Create necessary data directories for AngleCam."""
        self.logger.info("Setting up project directories...")
        
        directories = [
            "data",
            "data/checkpoint",
            "data/01_Training_Validation_Data",
            "data/01_Training_Validation_Data/image_data",
            "data/01_Training_Validation_Data/splits",
            "data/outputs",
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _setup_device(self) -> None:
        """Setup compute device."""
        device_config = self.config.device
        if device_config == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_config
        self.logger.info(f"Using device: {self.device}")

    def _validate_config(self) -> None:
        """Validate critical configuration parameters."""
        required_paths = [
            self.config.data.train_csv,
            self.config.data.val_csv,
            self.config.data.data_dir,
        ]

        for path in required_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Required path not found: {path}")

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

    def retrain(
        self,
        checkpoint_path: str,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        freeze_backbone: bool = True,
        reset_head: bool = False,
        save_path: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Retrain model from checkpoint with optional new data.

        Args:
            checkpoint_path: Path to checkpoint file to load
            train_csv: Optional new training CSV (uses config default if None)
            val_csv: Optional new validation CSV (uses config default if None)
            freeze_backbone: Whether to freeze backbone during retraining
            reset_head: Whether to reset head weights
            save_path: Path to save the retrained model
            **kwargs: Additional training parameters

        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting retraining from checkpoint: {checkpoint_path}")

        # Update data paths if new ones provided
        if train_csv is not None:
            self.config.data.train_csv = train_csv
            self.logger.info(f"Using new training data: {train_csv}")

        if val_csv is not None:
            self.config.data.val_csv = val_csv
            self.logger.info(f"Using new validation data: {val_csv}")

        # Apply retraining strategy to model
        self._apply_retraining_strategy(freeze_backbone, reset_head)

        # Update output directory to avoid overwriting original results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Handle different config structures for output_dir
        if hasattr(self.config, "training") and hasattr(
            self.config.training, "output_dir"
        ):
            original_output = self.config.training.output_dir
        elif hasattr(self.config, "output_dir"):
            original_output = self.config.output_dir
        else:
            # Fallback default
            original_output = "data/model/output"
            self.logger.warning("No output_dir found in config, using default")

        retrain_output = f"{original_output}_retrain_{timestamp}"
        self.config.training.output_dir = retrain_output
        self.logger.info(f"Retraining output directory: {retrain_output}")

        # Create new trainer (this will use the updated config)
        self.trainer = AngleCamTrainer(
            config=self.config, device=self.device, logger=self.logger
        )

        # Run training with the pretrained model
        results = self.trainer.train(self.model, **kwargs)
        self.model = results["model"]

        # Save model
        self.save(save_path)

        self.logger.info("Retraining completed successfully")
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
        cls,
        checkpoint_path: str,
        current_config: DictConfig,
        config_overrides: Optional[Dict] = None,
    ) -> "AngleCam":
        """Load trained AngleCam from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Use current config as base
        config = current_config

        # Apply any config overrides
        if config_overrides:
            override_config = OmegaConf.create(config_overrides)
            config = OmegaConf.merge(config, override_config)

        # Create instance
        instance = cls(config, mode="inference")

        # Load model
        instance.model = DINOv2_AngleCam(config)
        instance.model.load_state_dict(checkpoint["model_state_dict"])
        instance.model = instance.model.to(instance.device)

        return instance

    def _apply_retraining_strategy(
        self, freeze_backbone: bool, reset_head: bool
    ) -> None:
        """Apply retraining strategy to the model."""
        if freeze_backbone:
            # Freeze all backbone parameters
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            self.logger.info("Backbone frozen for retraining")
        else:
            # Use the trainable blocks from config
            for param in self.model.backbone.parameters():
                param.requires_grad = False

            # Unfreeze specified transformer blocks
            for name, param in self.model.backbone.named_parameters():
                for block_idx in self.config.model.trainable_transformer_blocks:
                    if f"blocks.{block_idx}" in name:
                        param.requires_grad = True
                        break
            self.logger.info("Backbone partially unfrozen based on config")

        if reset_head:
            # Reinitialize head layers
            self.model.head = self.model._create_head()
            self.model.head = self.model.head.to(self.device)
            self.logger.info("Head layers reinitialized")

        # Always ensure head is trainable
        for param in self.model.head.parameters():
            param.requires_grad = True