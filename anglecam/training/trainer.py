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
        self.best_test_metrics = {"test_r2_harmonic": 0.0}

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
        output_dir = Path(self.config.output_dir) / "checkpoints"
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
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))

        return {
            "train_loss": epoch_loss,
            "train_mae": metrics["mae"],
            "train_rmse": metrics["rmse"],
            "train_r2": metrics["r2"],
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
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))

        return {
            "val_loss": epoch_loss,
            "val_mae": metrics["mae"],
            "val_rmse": metrics["rmse"],
            "val_r2": metrics["r2"],
        }

    def _test_epoch(
        self, model: nn.Module, test_loader: DataLoader
    ) -> Dict[str, float]:
        """Testing for one epoch."""
        model.eval()
        all_predictions = []
        all_targets = []
        all_predictions_dict = []

        with torch.no_grad():
            for images, targets, _, _ in tqdm(test_loader, desc="Testing"):
                images = images.to(self.device)

                # Convert targets to proper tensor type and move to device
                if isinstance(targets, (list, tuple)):
                    targets = torch.tensor(targets, dtype=torch.float32).to(self.device)
                else:
                    targets = targets.float().to(self.device)

                outputs = model(images)

                # Convert to angles
                predicted_mean_angles = self._probabilities_to_mean_angles(outputs)

                all_predictions.extend(predicted_mean_angles.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())

        # Calculate metrics
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))

        batch_dict = {
            "predicted_mean_angles": all_predictions,
            "targets": all_targets,
        }

        # Save all predictions to JSON file once
        output_file = Path(self.config.output_dir) / "predictions.json"
        with open(output_file, "w") as f:
            json.dump(batch_dict, f, indent=2)

        return {
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "test_r2": metrics["r2"],
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

        train_loader, val_loader, test_loader_calathea, test_loader_maranta = (
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

            # Test epoch
            test_metrics_calathea = self._test_epoch(model, test_loader_calathea)
            test_metrics_maranta = self._test_epoch(model, test_loader_maranta)

            # Calculate harmonic mean of test R2
            test_r2_calathea = test_metrics_calathea["test_r2"]
            test_r2_maranta = test_metrics_maranta["test_r2"]
            test_r2_harmonic = 2 / (1 / test_r2_calathea + 1 / test_r2_maranta)

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
                "test_r2_harmonic": test_r2_harmonic,
                **test_metrics_calathea,
                **test_metrics_maranta,
            }

            self.training_history.append(epoch_metrics)

            # Log progress
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.7f}, "
                f"Train R2: {train_metrics['train_r2']:.4f} || "
                f"Val Loss: {val_metrics['val_loss']:.7f}, "
                f"Val R2: {val_metrics['val_r2']:.4f}, "
                f"Val MAE: {val_metrics['val_mae']:.4f}, "
                f"Val RMSE: {val_metrics['val_rmse']:.4f}, || "
                f"Test R2 Harmonic: {epoch_metrics['test_r2_harmonic']:.4f}, "
                f"Test Calathea R2: {test_metrics_calathea['test_r2']:.4f}, "
                f"Test Maranta R2: {test_metrics_maranta['test_r2']:.4f}, "
                f"Test Calathea RMSE: {test_metrics_calathea['test_rmse']:.4f}, "
                f"Test Maranta RMSE: {test_metrics_maranta['test_rmse']:.4f}, "
            )

            # Save best model
            if val_metrics["val_loss"] < self.best_metrics["val_loss"]:
                self.best_metrics.update(val_metrics)
                self.checkpointer.save_best_model(
                    model, self.optimizer, epoch, val_metrics["val_loss"], "val_loss"
                )

            # Save best model based on test R2 harmonic
            if test_r2_harmonic > self.best_test_metrics["test_r2_harmonic"]:
                self.best_test_metrics.update({"test_r2_harmonic": test_r2_harmonic})
                self.checkpointer.save_best_model(
                    model, self.optimizer, epoch, test_r2_harmonic, "test_r2_harmonic"
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
        output_path = Path(self.config.output_dir) / "training_history.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info(f"Training history saved to: {output_path}")
