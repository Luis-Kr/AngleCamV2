import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from .._data.dataset import get_data_loaders
from .._data.transforms import create_transform_pipeline
from ..utils.metrics import calculate_metrics
from ..utils.checkpointing import ModelCheckpointer


class AngleCamTrainer:
    """Training pipeline for AngleCam models."""

    def __init__(self, config: DictConfig, device: str, logger: logging.Logger):
        self.config = config
        self.device = device
        self.logger = logger

        # Training components
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.checkpointer = None

        # Training state
        self.training_history = []
        self.best_metrics = {"val_loss": float("inf")}

    def _setup_training_components(self, model: nn.Module) -> None:
        """Setup optimizer, criterion, scheduler, etc."""
        # Optimizer
        if self.config.training.optimizer.name == "AdamW":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay,
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {self.config.training.optimizer.name}. Only AdamW supported."
            )

        # Loss criterion
        loss_config = self.config.training.loss
        if loss_config.name == "HuberLoss":
            self.criterion = nn.HuberLoss()
        elif loss_config.name == "MSELoss":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss: {loss_config.name}. Only HuberLoss and MSELoss supported"
            )

        # Scheduler
        if hasattr(self.config, "scheduler"):
            scheduler_config = self.config.training.scheduler
            if scheduler_config.name == "ReduceLROnPlateau":
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode=scheduler_config.mode,
                    factor=scheduler_config.factor,
                    patience=scheduler_config.patience,
                    min_lr=scheduler_config.min_lr,
                    threshold=scheduler_config.threshold,
                    threshold_mode=scheduler_config.threshold_mode,
                )
            else:
                raise ValueError(
                    f"Unsupported scheduler: {scheduler_config.name}. Only ReduceLROnPlateau supported."
                )

        # Checkpointer
        output_dir = Path(self.config.training.output_dir) / "checkpoints"
        self.checkpointer = ModelCheckpointer(output_dir, self.logger, self.config)

    def _train_epoch(
        self, model: nn.Module, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, (images, targets, _, _) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = model(images)
            targets = self._normalize_distribution(targets)

            # Convert probability distribution to angle predictions
            predicted_mean_angles = self._probabilities_to_mean_angles(outputs)
            target_mean_angles = self._probabilities_to_mean_angles(targets)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.training.gradient_clipping.enabled:
                clip_grad_norm_(
                    model.parameters(),
                    max_norm=self.config.training.gradient_clipping.max_norm,
                    norm_type=self.config.training.gradient_clipping.norm_type,
                )

            self.optimizer.step()

            # Track metrics
            running_loss += loss.item()
            all_predictions.extend(predicted_mean_angles.detach().cpu().numpy())
            all_targets.extend(target_mean_angles.detach().cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)

        return {
            "train_loss": epoch_loss,
        }

    def _validate_epoch(
        self, model: nn.Module, val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _, _ in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = model(images)
                targets = self._normalize_distribution(targets)

                # Convert to angles
                predicted_mean_angles = self._probabilities_to_mean_angles(outputs)
                target_mean_angles = self._probabilities_to_mean_angles(targets)

                # Calculate loss
                loss = self.criterion(outputs, targets)
                running_loss += loss.item()

                all_predictions.extend(predicted_mean_angles.cpu().numpy())
                all_targets.extend(target_mean_angles.cpu().numpy())

        # Calculate metrics
        epoch_loss = running_loss / len(val_loader)

        return {
            "val_loss": epoch_loss,
        }

    def _normalize_distribution(
        self, distribution_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Normalize a distribution tensor to ensure it sums to 1."""
        return distribution_tensor / torch.sum(distribution_tensor, dim=1, keepdim=True)

    def _probabilities_to_mean_angles(
        self, probabilities: torch.Tensor
    ) -> torch.Tensor:
        """Convert probability distribution to angle predictions."""
        # Create angle bins
        angles = torch.linspace(
            0, 90, probabilities.shape[1], device=probabilities.device
        )
        
        # # Make sure the probabilities sum to 1
        # probabilities = probabilities / torch.sum(probabilities, dim=1, keepdim=True)

        return torch.sum(probabilities * angles, dim=1)

    def train(self, model: nn.Module, **kwargs) -> Dict[str, Any]:
        """
        Execute complete training pipeline.

        Args:
            model: Model to train
            **kwargs: Training parameter overrides

        Returns:
            Training results dictionary
        """
        start_time = time.time()

        # Setup training components
        self._setup_training_components(model)

        # Create data loaders
        self.logger.info("Loading datasets...")
        train_transform = create_transform_pipeline(self.config, mode="train")
        val_transform = create_transform_pipeline(self.config, mode="val")

        train_loader, val_loader = (
            get_data_loaders(
                config=self.config,
                train_transform=train_transform,
                val_transform=val_transform,
            )
        )

        # Training parameters
        num_epochs = self.config.training.epochs

        self.logger.info(f"Starting training for {num_epochs} epochs")

        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train epoch
            train_metrics = self._train_epoch(model, train_loader, epoch)

            # Validate epoch
            val_metrics = self._validate_epoch(model, val_loader)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics["val_loss"])

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - epoch_start,
                **train_metrics,
                **val_metrics,
            }

            self.training_history.append(epoch_metrics)

            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.7f}, "
                f"Val Loss: {val_metrics['val_loss']:.7f}, "
            )

            # Save best model
            if val_metrics["val_loss"] < self.best_metrics["val_loss"]:
                self.best_metrics.update(val_metrics)
                self.checkpointer.save_best_model(
                    model, self.optimizer, epoch, val_metrics["val_loss"], "val_loss"
                )

            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpointing.save_every == 0:
                self.checkpointer.save_checkpoint(model, self.optimizer, epoch)

        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")

        # Save training history
        self._save_training_history()

        return {
            "model": model,
            "training_history": self.training_history,
            "best_metrics": self.best_metrics,
            "total_time": total_time,
        }

    def _save_training_history(self) -> None:
        """Save training history to file."""
        output_path = Path(self.config.training.output_dir) / "training_history.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Training history saved to: {output_path}")
