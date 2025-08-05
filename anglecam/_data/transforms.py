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
    
    def get_transforms_training(self) -> Tuple[A.Compose, A.Compose, A.Compose]:
        geometric_tf = self._create_geometric_transforms()
        photometric_tf = self._create_photometric_transforms()
        rescaling_tf = self._create_rescaling_transforms()
        
        return geometric_tf, photometric_tf, rescaling_tf
    
    def get_transforms_validation(self) -> A.Compose:
        transform_list = [
            A.Resize(
                height=self.cfg.backbone.resize_size,
                width=self.cfg.backbone.resize_size,
                interpolation=cv2.INTER_LINEAR
            ),
            A.CenterCrop(
                height=self.cfg.backbone.crop_size, 
                width=self.cfg.backbone.crop_size
            )
        ]

        if self.cfg.backbone.normalize:
            transform_list.append(self._get_normalization())
        
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)
    
    def get_transforms_testing(self) -> A.Compose:
        return self.get_transforms_validation()
    
    def _create_geometric_transforms(self) -> A.Compose:
        geometric_transforms = [
            A.RandomResizedCrop(
                size=[self.cfg.backbone.resize_size, self.cfg.backbone.resize_size],
                scale=self.cfg.augmentation.resized_crop_scale,
                ratio=[0.75, 1.33], 
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST
            )
        ]
        
        horizontal_flip_prob = self.cfg.augmentation.horizontal_flip
        if horizontal_flip_prob > 0:
            geometric_transforms.append(A.HorizontalFlip(p=horizontal_flip_prob))
        
        return A.Compose(geometric_transforms)
    
    def _create_photometric_transforms(self) -> A.Compose:
        photometric_transforms = []
        
        brightness_limit = self.cfg.augmentation.brightness
        contrast_limit = self.cfg.augmentation.contrast
        if brightness_limit > 0 or contrast_limit > 0:
            photometric_transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=brightness_limit,
                    contrast_limit=contrast_limit,
                    p=0.8 
                )
            )
        
        gaussian_noise_prob = self.cfg.augmentation.gaussian_noise
        if gaussian_noise_prob > 0:
            photometric_transforms.append(
                A.GaussNoise(
                    std_range=[0.1, 0.2],
                    mean_range=[0, 0],
                    p=gaussian_noise_prob
                )
            )
        
        return A.Compose(photometric_transforms)
    
    def _create_rescaling_transforms(self) -> A.Compose:
        transform_list = []
        
        if self.normalize:
            transform_list.append(self._get_normalization())
        
        transform_list.append(ToTensorV2())
        
        return A.Compose(transform_list)
    
    @staticmethod
    def _get_normalization() -> A.Normalize:
        return A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        )


def create_transform_pipeline(
    config: DictConfig,
    mode: str = 'train'
) -> Union[Tuple[A.Compose, A.Compose, A.Compose], A.Compose]:
    transform_factory = GetTransforms(config)
    
    if mode == 'train':
        return transform_factory.get_transforms_training()
    elif mode in ['val', 'validation']:
        return transform_factory.get_transforms_validation()
    elif mode == 'test':
        return transform_factory.get_transforms_testing()
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train', 'val', or 'test'.")


# Convenience aliases for backward compatibility
AngleCamTransforms = GetTransforms