"""
Image transformation utilities for AngleCam leaf angle estimation.

"""

import numpy as np

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


def normalize_to_float32(image, **kwargs):
    """Convert image to float32 and normalize to [0,1] range."""
    return image.astype(np.float32) / 255.0


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

        # # Randomly convert to grayscale
        # grayscale_prob = self.cfg.model.augmentation.grayscale
        # if grayscale_prob > 0:
        #     transform_list.append(A.ToGray(num_output_channels=3, p=grayscale_prob))

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
                    p=0.5,
                )
            )

        saturation_limit = self.cfg.model.augmentation.saturation
        if saturation_limit > 0:
            transform_list.append(
                A.HueSaturationValue(
                    hue_shift_limit=0,
                    sat_shift_limit=saturation_limit,
                    val_shift_limit=[-5, 5],
                    p=0.1,
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
        else:
            transform_list.append(A.Lambda(image=normalize_to_float32))

        # transform_list.append(A.Lambda(image=lambda im, **k: (print('RANGE', im.dtype, float(im.min()), float(im.max())) or im)))

        transform_list.append(ToTensorV2())

        return A.Compose(transform_list)

    def _debug_print(self, image, stage):
        """Debug function to print image info"""
        print(
            f"DEBUG {stage}: shape={image.shape}, dtype={image.dtype}, min={image.min():.3f}, max={image.max():.3f}"
        )
        return image

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
            # A.Lambda(image=lambda x, **kwargs: x.astype(np.float32) / 255.0),
        ]

        if self.cfg.model.backbone.normalize:
            transform_list.append(self._get_normalization())
        else:
            transform_list.append(A.Lambda(image=normalize_to_float32))

        # transform_list.append(A.Lambda(image=normalize_to_float32))

        # transform_list.append(A.Lambda(image=lambda im, **k: (print('RANGE', im.dtype, float(im.min()), float(im.max())) or im)))

        transform_list.append(ToTensorV2())

        return A.Compose(transform_list)

    def _probe(self, img, tag):
        mn, mx = float(img.min()), float(img.max())
        ch = img.shape[2] if img.ndim == 3 else 1
        print(f"[{tag}] shape={img.shape} range=({mn:.3f},{mx:.3f})")
        return img

    @staticmethod
    def _get_normalization() -> A.Normalize:
        return A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]  # ImageNet statistics
        )


def create_transform_pipeline(config: DictConfig, mode: str = "train") -> A.Compose:
    transform_factory = GetTransforms(config)

    if mode == "train":
        return transform_factory.get_transforms_training()
    elif mode in ["val", "validation"]:
        return transform_factory.get_transforms_validation()
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'val'.")
