import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List


class DINOv2_AngleCam(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # Load the backbone
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", self.cfg.model.name, pretrained=True
        )

        # Freeze the backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the transformer blocks specified in the config
        for name, param in self.backbone.named_parameters():
            for block_idx in self.cfg.model.trainable_transformer_blocks:
                if f"blocks.{block_idx}" in name:
                    param.requires_grad = True
                    break

        # Create the head
        self.head = self._create_head()

    def _create_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Dropout(self.cfg.model.head.dropout),
            nn.Linear(
                self.cfg.model.backbone.hidden_dim, self.cfg.model.head.hidden_dims[0]
            ),
            nn.Dropout(self.cfg.model.head.dropout),
            nn.GELU(),
            nn.Linear(
                self.cfg.model.head.hidden_dims[0], self.cfg.model.output.num_bins
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through the backbone
        x = self.backbone(x)
        # Forward pass through the head
        x = self.head(x)
        x = F.softmax(x, dim=1)
        return x

    def _get_trainable_parameters(self) -> List[torch.Tensor]:
        return [p for p in self.parameters() if p.requires_grad]
