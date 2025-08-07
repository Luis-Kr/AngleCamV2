"""
Image transformation utilities for AngleCam leaf angle estimation.

"""

from typing import Tuple, Union
from pathlib import Path

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


class GetTransforms:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def get_transforms_training(self) -> A.Compose:
        """Create complete training transform pipeline."""
        transform_list = []

        # Add geometric transforms
        transform_list.extend(
            [
                A.RandomResizedCrop(
                    size=[
                        self.cfg.model.backbone.resize_size,
                        self.cfg.model.backbone.resize_size,
                    ],
                    scale=self.cfg.model.augmentation.resized_crop_scale,
                    ratio=[0.75, 1.33],
                    interpolation=cv2.INTER_LINEAR,
                )
            ]
        )

        # Add horizontal flip if enabled
        horizontal_flip_prob = self.cfg.model.augmentation.horizontal_flip
        if horizontal_flip_prob > 0:
            transform_list.append(A.HorizontalFlip(p=horizontal_flip_prob))

        # Add photometric transforms
        brightness_limit = self.cfg.model.augmentation.brightness
        contrast_limit = self.cfg.model.augmentation.contrast
        if brightness_limit > 0 or contrast_limit > 0:
            transform_list.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=0.8,
                )
            )

        # Add noise if enabled
        gaussian_noise_prob = self.cfg.model.augmentation.gaussian_noise
        if gaussian_noise_prob > 0:
            transform_list.append(
                A.GaussNoise(
                    std_range=[0.1, 0.2], mean_range=[0, 0], p=gaussian_noise_prob
                )
            )

        # Add normalization if enabled
        if self.cfg.model.backbone.normalize:
            transform_list.append(self._get_normalization())

        # Add tensor conversion
        transform_list.append(ToTensorV2())

        return A.Compose(transform_list)

    def get_transforms_validation(self) -> A.Compose:
        transform_list = [
            A.Resize(
                height=self.cfg.model.backbone.resize_size,
                width=self.cfg.model.backbone.resize_size,
                interpolation=cv2.INTER_LINEAR,
            ),
            A.CenterCrop(
                height=self.cfg.model.backbone.crop_size,
                width=self.cfg.model.backbone.crop_size,
            ),
        ]

        if self.cfg.model.backbone.normalize:
            transform_list.append(self._get_normalization())

        transform_list.append(ToTensorV2())

        return A.Compose(transform_list)

    def get_transforms_testing(self) -> A.Compose:
        return self.get_transforms_validation()

    @staticmethod
    def _get_normalization() -> A.Normalize:
        return A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet statistics
        )


def create_transform_pipeline(
    config: DictConfig, mode: str = "train"
) -> A.Compose: 
    transform_factory = GetTransforms(config)

    if mode == "train":
        return transform_factory.get_transforms_training()
    elif mode in ["val", "validation"]:
        return transform_factory.get_transforms_validation()
    elif mode == "test":
        return transform_factory.get_transforms_testing()
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train', 'val', or 'test'.")
