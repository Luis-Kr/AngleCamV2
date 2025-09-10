import time
import random
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig
import os
import cv2

# Get project root directory (relative to this file)
root_dir = Path(__file__).parent.parent.parent


class BaseAngleCamDataset(Dataset):
    """
    Base dataset class providing common functionality for AngleCam datasets.

    Handles image loading, label processing, and provides consistent interfaces
    for traininga and validation workflows.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the base dataset.

        Args:
            data_dir: Directory containing the image and label files
            dataframe: DataFrame with filename and metadata information
            transform: Image transformation pipeline
        """
        self.data_dir = Path(data_dir)
        self.dataframe = dataframe.copy()
        self.transform = transform

        # Validate data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Validate required columns in dataframe
        required_columns = ["filename"]
        missing_columns = [
            col for col in required_columns if col not in self.dataframe.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in dataframe: {missing_columns}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        return len(self.dataframe)

    def _load_image(self, idx: int) -> Image.Image:
        """Load and validate image file"""
        filename = self.dataframe.iloc[idx]["filename"]
        image_path = self.data_dir / filename

        try:
            image = Image.open(image_path)
            # Ensure consistent three channel format for all images
            three_channel_image = image.convert("RGB")

            if three_channel_image.mode != "RGB":
                raise ValueError(
                    f"Failed to convert to three channel mode. Current mode: {three_channel_image.mode}"
                )

            return three_channel_image

        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")

    def _load_labels_safely(self, labels_path: Path) -> pd.DataFrame:
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Try common delimiters
                for delimiter in [" ", ",", "\t"]:
                    try:
                        labels_df = pd.read_csv(
                            labels_path, header=None, sep=delimiter, dtype=float
                        )
                        if not labels_df.empty and labels_df.shape[1] > 1:
                            return labels_df
                    except (
                        pd.errors.EmptyDataError,
                        ValueError,
                        pd.errors.ParserError,
                    ):
                        continue

                # If no delimiter worked
                raise ValueError("Could not parse file with any standard delimiter")

            except (IOError, OSError) as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to read {labels_path} after {max_retries} attempts: {str(e)}"
                    )
                time.sleep(retry_delay)
            except ValueError as e:
                raise RuntimeError(f"Invalid data format in {labels_path}: {str(e)}")

        raise RuntimeError(f"Unexpected error reading {labels_path}")

    def _get_label_filename(self, image_filename: str) -> str:
        """Convert image filename to corresponding label filename"""
        base_name = image_filename.rsplit(".", 1)[0]
        return f"{base_name}_sim.csv"


class AngleCamTrainingDataset(BaseAngleCamDataset):
    """Training dataset with data augmentation."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
        min_simulation_rows: int = 2,
        grayscale_prob: float = 0.5,
        distance_dimming_prob: float = 0.5,
        d0: float = 1.0,
        min_intensity: float = 0.1,
    ) -> None:
        """
        Initialize training dataset.

        Args:
            data_dir: Directory containing training data
            dataframe: DataFrame with training sample information
            transform: Image transformation pipeline
            logger: Logger instance for reporting issues
            min_simulation_rows: Minimum number of simulation rows required per sample
        """
        super().__init__(data_dir, dataframe, transform)

        self.min_simulation_rows = min_simulation_rows
        self.valid_indices = list(range(len(dataframe)))
        self.logger = logger

        # RGB/Grayscale control parameters
        self.grayscale_prob = grayscale_prob
        self.distance_dimming_prob = distance_dimming_prob
        self.d0 = d0
        self.min_intensity = min_intensity

        # Validate that label files exist and have sufficient data
        self._validate_label_files()

    def _load_depth_map(self, image_filename: str) -> np.ndarray:
        """Load depth map corresponding to the image."""
        # Convert image filename to depth map filename
        base_name = image_filename.rsplit(".", 1)[0]
        depth_filename = f"{base_name}_dpm.npy"
        depth_path = self.data_dir / depth_filename

        try:
            depth_map = np.load(depth_path)
            return depth_map
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"Depth map not found: {depth_path}")
            # Return dummy depth map (all ones) if not found
            return np.ones((224, 224), dtype=np.float32)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error loading depth map {depth_path}: {str(e)}")
            return np.ones((224, 224), dtype=np.float32)

    def _apply_distance_dimming(
        self, image: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        """Apply distance-based dimming to simulate IR camera behavior."""
        # Ensure depth map has same spatial dimensions as image
        if depth_map.shape[:2] != image.shape[:2]:
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))

        # Smooth the depth map to reduce speckle noise
        depth_map = cv2.medianBlur(depth_map, 3)
        depth_map = cv2.GaussianBlur(depth_map, (3, 3), 0.3)

        # Scale grayscale image to 0-255 range
        image = (image * 255).astype(np.uint8)

        # Apply distance-based scaling: I_IR = I_gray * (d0/d)^2
        distance_factor = (self.d0 / (depth_map + 2.0)) ** 2

        # Apply scaling to all channels
        pseudo_IR = image.copy().astype(np.float32)
        pseudo_IR *= distance_factor

        # Scale back to np.uint8 to smooth out subtle artifacts
        pseudo_IR = pseudo_IR.astype(np.uint8)

        # Scale back to 0-1 range
        pseudo_IR = (pseudo_IR - pseudo_IR.min()) / (
            pseudo_IR.max() - pseudo_IR.min()
        ).astype(np.float32)

        # Expand dims to 3 channels
        pseudo_IR = np.expand_dims(pseudo_IR, axis=2)
        pseudo_IR = np.repeat(pseudo_IR, 3, axis=2)

        return pseudo_IR

    def _process_image_modality(self, image: Image.Image, filename: str) -> np.ndarray:
        """Process image to either RGB or grayscale with optional distance dimming."""
        # Convert to numpy array (0-1 range)
        image_array = np.array(image).astype(np.float32) / 255.0

        # Decide on modality based on probability
        use_grayscale = random.random() < self.grayscale_prob

        if use_grayscale:
            # Convert to grayscale (maintain 3 channels)
            gray_image = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
            gray_image_expanded_dims = np.expand_dims(gray_image, axis=2)
            image_array = np.repeat(gray_image_expanded_dims, 3, axis=2)

            # Apply distance dimming with probability
            if random.random() < self.distance_dimming_prob:
                depth_map = self._load_depth_map(filename)
                image_array = self._apply_distance_dimming(gray_image, depth_map)

        return image_array

    def _validate_label_files(self) -> None:
        """Validate that all required label files exist and have sufficient data."""
        invalid_indices = []

        for idx in range(len(self.dataframe)):
            try:
                filename = self.dataframe.iloc[idx]["filename"]
                labels_path = self.data_dir / self._get_label_filename(filename)

                if not labels_path.exists():
                    invalid_indices.append(idx)
                    continue

                # Check if file has sufficient simulation rows
                labels_df = self._load_labels_safely(labels_path)
                if len(labels_df) < self.min_simulation_rows:
                    invalid_indices.append(idx)

            except Exception:
                invalid_indices.append(idx)

        # Remove invalid indices
        self.valid_indices = [i for i in self.valid_indices if i not in invalid_indices]

        if invalid_indices and self.logger:
            self.logger.warning(
                f"Removed {len(invalid_indices)} samples with insufficient or missing label data"
            )

    def __len__(self) -> int:
        """Return number of valid training samples."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """Get training sample with RGB/grayscale modality."""
        valid_idx = self.valid_indices[idx]

        try:
            # Load image
            image = self._load_image(valid_idx)
            filename = self.dataframe.iloc[valid_idx]["filename"]
            image_path = str(self.data_dir / filename)

            # Process image modality (RGB vs grayscale with distance dimming)
            processed_image = self._process_image_modality(image, filename)

            # Apply transforms if provided
            if self.transform:
                # Convert back to 0-255 range for albumentations
                image_uint8 = (processed_image * 255).astype(np.uint8)

                transformed = self.transform(image=image_uint8)
                image_tensor = transformed["image"]
            else:
                # Convert to tensor directly
                image_tensor = (
                    torch.from_numpy(processed_image).permute(2, 0, 1).float()
                )

            # Load labels
            labels = self._load_labels(valid_idx)

            return image_tensor, labels, image_path, valid_idx

        except Exception as e:
            raise RuntimeError(f"Error processing sample {idx}: {str(e)}")

    def _load_labels(self, idx: int) -> torch.Tensor:
        filename = self.dataframe.iloc[idx]["filename"]
        labels_path = self.data_dir / self._get_label_filename(filename)

        try:
            labels_df = self._load_labels_safely(labels_path)

            # Use random simulation from available rows
            max_row = max(self.min_simulation_rows - 1, len(labels_df) - 1)
            row_idx = random.randint(1, max_row) if max_row > 0 else 0

            row_data = labels_df.iloc[row_idx, :].astype(float)
            return torch.tensor(row_data.values, dtype=torch.float32)

        except Exception as e:
            raise RuntimeError(f"Error loading labels for index {idx}: {str(e)}")


class AngleCamValidationDataset(BaseAngleCamDataset):
    """Validation dataset using labeled LIADs."""

    def __init__(
        self,
        data_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ) -> None:
        """
        Initialize validation dataset.

        Args:
            data_dir: Directory containing validation data
            dataframe: DataFrame with validation sample information
            transform: Image transformation pipeline
        """
        super().__init__(data_dir, dataframe, transform)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """
        Get validation sample with ground truth labels.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label_tensor, image_path, index)
        """
        try:
            # Load and transform image
            image = self._load_image(idx)

            # Apply transforms if provided
            if self.transform:
                if hasattr(self.transform, "__call__"):
                    # Albumentations transform
                    transformed = self.transform(image=np.array(image))
                    image_tensor = transformed["image"]
                else:
                    # Torchvision transforms
                    image_tensor = self.transform(image)
            else:
                # Default to tensor conversion
                image_tensor = transforms.ToTensor()(image)

            # Load ground truth labels (row 0 is ground truth)
            filename = self.dataframe.iloc[idx]["filename"]
            labels_path = self.data_dir / self._get_label_filename(filename)

            labels_df = self._load_labels_safely(labels_path)
            labels_tensor = torch.tensor(
                labels_df.iloc[0, :].values, dtype=torch.float32
            )

            image_path = str(self.data_dir / filename)

            return image_tensor, labels_tensor, image_path, idx

        except Exception as e:
            raise RuntimeError(f"Error processing validation sample {idx}: {str(e)}")


def load_dataframes(
    config: DictConfig,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation from config-specified paths.

    Args:
        config: Hydra configuration containing data paths
        logger: Optional logger for reporting

    Returns:
        Tuple of (train_df, val_df)
    """
    # Use relative paths from project root
    train_path = root_dir / config.data.train_csv
    val_path = root_dir / config.data.val_csv

    # Load dataframes
    train_df = pd.read_csv(train_path, sep=",")
    val_df = pd.read_csv(val_path, sep=",")

    # Reset indices for clean datasets
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if logger:
        logger.info(f"Loaded {len(train_df)} training samples from {train_path}")
        logger.info(f"Loaded {len(val_df)} validation samples from {val_path}")

    return train_df, val_df


def get_data_loaders(
    config: DictConfig,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation using Hydra config.

    Args:
        config: Hydra configuration object
        train_transform: Training data augmentation pipeline
        val_transform: Validation data preprocessing pipeline
        logger: Optional logger for reporting

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load dataframes from config
    train_df, val_df = load_dataframes(config, logger)

    # Get data directory from config (relative to project root)
    data_dir = root_dir / config.data.data_dir

    # Get training parameters from config
    batch_size = config.training.batch_size
    num_workers = config.training.dataloader.num_workers
    pin_memory = config.training.dataloader.pin_memory

    # Check if system has enough workers available
    if num_workers > 0:
        num_workers = min(num_workers, os.cpu_count() - 1)

    # Create datasets
    train_dataset = AngleCamTrainingDataset(
        data_dir=data_dir,
        dataframe=train_df,
        transform=train_transform,
        logger=logger,
        grayscale_prob=config.model.augmentation.get("grayscale_prob", 0.5),
        distance_dimming_prob=config.model.augmentation.get(
            "distance_dimming_prob", 0.5
        ),
        d0=config.model.augmentation.get("d0", 1.0),
        min_intensity=config.model.augmentation.get("min_intensity", 0.1),
    )

    val_dataset = AngleCamValidationDataset(
        data_dir=data_dir, dataframe=val_df, transform=val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(config.seed),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    if logger:
        logger.info(f"Created data loaders:")
        logger.info(
            f"  - Training: {len(train_loader)} batches (batch_size={batch_size})"
        )
        logger.info(f"  - Validation: {len(val_loader)} batches (batch_size=1)")
        logger.info(f"  - Data directory: {data_dir}")

    return train_loader, val_loader


def worker_init_fn(worker_id):
    """Initialize worker with deterministic seed."""
    # Set seed for this worker
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Convenience function for backward compatibility
def create_data_loaders(config: DictConfig, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """Backward compatibility function that wraps get_data_loaders."""
    return get_data_loaders(config, **kwargs)
