import torch
from pathlib import Path
import logging
from omegaconf import DictConfig, OmegaConf


class ModelCheckpointer:
    """Handle model checkpointing and saving."""

    def __init__(
        self, output_dir: Path, logger: logging.Logger, config: DictConfig = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.config = config

    def save_checkpoint(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int
    ) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # Add config if available
        if self.config is not None:
            checkpoint_dict["config"] = OmegaConf.to_yaml(self.config)
            checkpoint_dict["model_class"] = model.__class__.__name__

        torch.save(checkpoint_dict, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_best_model(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metric_value: float,
        metric_name: str,
    ) -> None:
        """Save best model based on metric."""
        best_path = self.output_dir / f"best_model_{metric_name}.pth"

        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": metric_value,
            "metric_name": metric_name,
        }

        # Add config if available
        if self.config is not None:
            checkpoint_dict["config"] = OmegaConf.to_yaml(self.config)
            checkpoint_dict["model_class"] = model.__class__.__name__

        torch.save(checkpoint_dict, best_path)
        self.logger.info(
            f"Best model saved: {best_path} ({metric_name}: {metric_value:.4f})"
        )
