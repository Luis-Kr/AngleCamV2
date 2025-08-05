import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
from omegaconf import DictConfig

# Get base_dir of the file
root_dir = Path(__file__).parent.parent.parent


class BaseAngleCamDataset(Dataset):
    """
    Base dataset class providing common functionality for AngleCam datasets.
    
    Handles image loading, label processing, and provides consistent interfaces
    for training, validation, and testing workflows.
    """
    
    def __init__(
        self, 
        data_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None
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
        required_columns = ['filename']
        missing_columns = [col for col in required_columns if col not in self.dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataframe: {missing_columns}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset """
        return len(self.dataframe)
    
    def _load_image(self, idx: int) -> Image.Image:
        """Load and validate image file """
        filename = self.dataframe.iloc[idx]['filename']
        image_path = self.data_dir / filename
        
        try:
            image = Image.open(image_path)
            # Ensure consistent three channel format for all images
            three_channel_image = image.convert('RGB')
            
            if three_channel_image.mode != 'RGB':
                raise ValueError(f"Failed to convert to three channel mode. Current mode: {three_channel_image.mode}")
                
            return three_channel_image
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {str(e)}")
    
    def _load_labels_safely(self, labels_path: Path) -> pd.DataFrame:
        max_retries = 3
        retry_delay = 1.0 
        
        for attempt in range(max_retries):
            try:
                # Try common delimiters
                for delimiter in [' ', ',', '\t']:
                    try:
                        labels_df = pd.read_csv(
                            labels_path, 
                            header=None, 
                            sep=delimiter, 
                            dtype=float
                        )
                        if not labels_df.empty and labels_df.shape[1] > 1:
                            return labels_df
                    except (pd.errors.EmptyDataError, ValueError, pd.errors.ParserError):
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
        """Convert image filename to corresponding label filename """
        base_name = image_filename.rsplit('.', 1)[0]
        return f"{base_name}_sim.csv"


class AngleCamTrainingDataset(BaseAngleCamDataset):
    """Training dataset with data augmentation."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        logger: logging.Logger = None,
        min_simulation_rows: int = 2
    ) -> None:
        """
        Initialize training dataset.
        
        Args:
            data_dir: Directory containing training data
            dataframe: DataFrame with training sample information
            transform: Image transformation pipeline
            curriculum_enabled: Whether to use curriculum learning
            min_simulation_rows: Minimum number of simulation rows required per sample
        """
        super().__init__(data_dir, dataframe, transform)
        
        self.min_simulation_rows = min_simulation_rows
        self.valid_indices = list(range(len(dataframe)))
        self.logger = logger
        
        # Validate that label files exist and have sufficient data
        self._validate_label_files()
    
    def _validate_label_files(self) -> None:
        """Validate that all required label files exist and have sufficient data."""
        invalid_indices = []
        
        for idx in range(len(self.dataframe)):
            try:
                filename = self.dataframe.iloc[idx]['filename']
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
        
        if invalid_indices:
            self.logger.warning(
                f"Removed {len(invalid_indices)} samples with insufficient or missing label data"
            )
    
    def __len__(self) -> int:
        """Return number of valid training samples."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """
        Get training sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor, image_path, original_index)
        """
        valid_idx = self.valid_indices[idx]
        
        try:
            # Load and transform image
            image = self._load_image(valid_idx)
            image_tensor = transforms.ToTensor()(image)
            
            # Load labels with curriculum strategy
            labels = self._load_labels(valid_idx)
            
            image_path = str(self.data_dir / self.dataframe.iloc[valid_idx]['filename'])
            
            return image_tensor, labels, image_path, valid_idx
            
        except Exception as e:
            raise RuntimeError(f"Error processing sample {idx} (valid_idx={valid_idx}): {str(e)}")
    
    def _load_labels(self, idx: int) -> torch.Tensor:
        """
        Load labels using curriculum learning strategy.
        
        Early in training, uses only the most accurate simulation (row 0).
        As training progresses, gradually includes noisier simulations.
        
        Args:
            idx: Sample index
            
        Returns:
            Label tensor for the selected simulation
        """
        filename = self.dataframe.iloc[idx]['filename']
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
    """Validation dataset using ground truth labels."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int]:
        """
        Get validation sample with ground truth labels."""
        try:
            # Load and transform image
            image = self._load_image(idx)
            image_tensor = transforms.ToTensor()(image)
            
            # Load ground truth labels
            filename = self.dataframe.iloc[idx]['filename']
            labels_path = self.data_dir / self._get_label_filename(filename)
            
            labels_df = self._load_labels_safely(labels_path)
            labels_tensor = torch.tensor(labels_df.iloc[0, :].values, dtype=torch.float32)
            
            image_path = str(self.data_dir / filename)
            
            return image_tensor, labels_tensor, image_path, idx
            
        except Exception as e:
            raise RuntimeError(f"Error processing validation sample {idx}: {str(e)}")


class AngleCamTestDataset(BaseAngleCamDataset):
    """Test dataset for inference."""
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """Get test sample for inference."""
        try:
            # Load and transform image
            image = self._load_image(idx)
            image_tensor = transforms.ToTensor()(image)
            
            image_path = str(self.data_dir / self.dataframe.iloc[idx]['filename'])
            
            return image_tensor, image_path, idx
            
        except Exception as e:
            raise RuntimeError(f"Error processing test sample {idx}: {str(e)}")


def load_dataframes(
    cfg: DictConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load main data
    train_df = pd.read_csv(root_dir / cfg.input.train_csv, sep=',')
    val_df = pd.read_csv(root_dir / cfg.input.val_csv, sep=',')
    test_df = pd.read_csv(root_dir / cfg.input.test_csv, sep=',')
    
    # Reset indices for clean datasets
    train_df = train_df.reset_index(drop=True)
    if val_df is not None:
        val_df = val_df.reset_index(drop=True)
    if test_df is not None:
        test_df = test_df.reset_index(drop=True)
        
    logger.info(f"Loaded {len(train_df)} training samples")
    logger.info(f"Loaded {len(val_df)} validation samples")
    logger.info(f"Loaded {len(test_df)} test samples")
    
    return train_df, val_df, test_df


def get_data_loaders(
    config: DictConfig = None,
    train_data: Union[str, Path, pd.DataFrame] = None,
    val_data: Optional[Union[str, Path, pd.DataFrame]] = None,
    test_data: Optional[Union[str, Path, pd.DataFrame]] = None,
    worker_init_fn: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create PyTorch DataLoaders for training, validation, and testing.
    
    Args:
        train_data: Training data (CSV path or DataFrame)
        val_data: Validation data (CSV path or DataFrame, optional)
        test_data: Test data (CSV path or DataFrame, optional)
        config: Configuration object with data loading parameters
        worker_init_fn: Function for initializing DataLoader workers
        train_transform: Training data augmentation pipeline
        val_transform: Validation/test data preprocessing pipeline
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set default configuration values
    if config is None:
        config = DictConfig({
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'data_dir': './data'
        })
    
    batch_size = config.get('batch_size', 16)
    num_workers = config.get('num_workers', 4)
    pin_memory = config.get('pin_memory', True)
    data_dir = config.get('data_dir', './data')
    
    # Load dataframes if paths are provided
    if isinstance(train_data, (str, Path)):
        train_df = pd.read_csv(train_data)
    else:
        train_df = train_data
    
    if val_data is not None:
        if isinstance(val_data, (str, Path)):
            val_df = pd.read_csv(val_data)
        else:
            val_df = val_data
    else:
        val_df = None
    
    if test_data is not None:
        if isinstance(test_data, (str, Path)):
            test_df = pd.read_csv(test_data)
        else:
            test_df = test_data
    else:
        test_df = None
    
    # Create datasets
    train_dataset = AngleCamTrainingDataset(
        data_dir=data_dir,
        dataframe=train_df,
        transform=train_transform
    )
    
    val_dataset = None
    if val_df is not None:
        val_dataset = AngleCamValidationDataset(
            data_dir=data_dir,
            dataframe=val_df,
            transform=val_transform
        )
    
    test_dataset = None
    if test_df is not None:
        test_dataset = AngleCamTestDataset(
            data_dir=data_dir,
            dataframe=test_df,
            transform=val_transform
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Use batch size 1 for validation for consistent metrics
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=num_workers > 0
        )
    
    return train_loader, val_loader, test_loader