# anglecam/inference/predictor.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
import time

from .._data.transforms import create_transform_pipeline

class ImageDataset(Dataset):
    """Dataset for batch image loading."""
    
    def __init__(self, image_paths: List[Path], transform):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        if hasattr(self.transform, "__call__"):
            # Albumentations
            transformed = self.transform(image=np.array(image))
            image_tensor = transformed["image"]
        else:
            # Torchvision
            image_tensor = self.transform(image)
            
        return image_tensor, str(image_path)

class AngleCamPredictor:
    """Inference pipeline for AngleCam models."""

    def __init__(self, model, config, device: str, logger):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logger

        # Setup transforms
        self.transform = create_transform_pipeline(config, mode="val")

        # Put model in eval mode
        self.model.eval()

    def predict(
        self,
        input_data: Union[str, List[str], Path],
        reference_mean: Optional[Union[float, List[float], Dict[str, float]]] = None,
    ) -> Union[Dict, List[Dict]]:
        """
        Predict leaf angles from image(s).

        Args:
            input_data: Single image path, list of paths, or directory
            reference_mean: Optional reference mean angle(s) for comparison.
                          - Single float: applied to single image
                          - List of floats: applied to list of images (must match length)
                          - Dict[str, float]: maps image paths to reference values
                          - None: no reference comparison

        Returns:
            Prediction results with optional reference comparison
        """
        if isinstance(input_data, (str, Path)):
            input_path = Path(input_data)

            if input_path.is_file():
                # Single image
                ref_val = (
                    reference_mean
                    if isinstance(reference_mean, (float, int, type(None)))
                    else None
                )
                return self._predict_single_image(input_path, ref_val)

            elif input_path.is_dir():
                # Directory of images
                image_paths = list(input_path.glob("*.png")) + list(
                    input_path.glob("*.jpg")
                )
                print(f"Predicting {len(image_paths)} images")
                return [
                    self._predict_batch(
                        image_paths,
                        reference_means=[self._get_reference_for_path(img_path, reference_mean, idx) for idx, img_path in enumerate(image_paths)],
                        batch_size=16,
                        num_workers=20
                    ) 
                ]
            else:
                raise ValueError(f"Invalid input path: {input_path}")

        elif isinstance(input_data, list):
            # List of image paths
            return [
                self._predict_single_image(
                    Path(img_path),
                    self._get_reference_for_path(Path(img_path), reference_mean, idx),
                )
                for idx, img_path in enumerate(input_data)
            ]

        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

    def _get_reference_for_path(
        self,
        img_path: Path,
        reference_mean: Optional[Union[float, List[float], Dict[str, float]]],
        idx: int,
    ) -> Optional[float]:
        """Extract reference value for a specific image path."""
        if reference_mean is None:
            return None
        elif isinstance(reference_mean, (float, int)):
            return float(reference_mean)
        elif isinstance(reference_mean, list):
            return float(reference_mean[idx]) if idx < len(reference_mean) else None
        elif isinstance(reference_mean, dict):
            return reference_mean.get(str(img_path)) or reference_mean.get(
                img_path.name
            )
        else:
            return None

    def _predict_single_image(
        self, image_path: Path, reference_mean: Optional[float] = None
    ) -> Dict[str, Any]:
        """Predict angles for a single image."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Apply transforms
            if hasattr(self.transform, "__call__"):
                # Albumentations transform
                transformed = self.transform(image=np.array(image))
                image_tensor = transformed["image"].unsqueeze(0).to(self.device)
            else:
                # Torchvision transforms
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
  
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = outputs.squeeze().cpu().numpy()
                

            # Convert to angle statistics
            angle_stats = self._probabilities_to_stats(probabilities)

            # Build result dictionary
            result = {
                "image_path": str(image_path),
                "predicted_mean_leaf_angle": angle_stats["mean_angle"],
                "angle_distribution": probabilities.tolist(),
            }

            # Add reference comparison if provided
            if reference_mean is not None:
                result["reference_mean_leaf_angle"] = float(reference_mean)
                result["prediction_error"] = angle_stats["mean_angle"] - float(
                    reference_mean
                )
                result["absolute_error"] = abs(
                    angle_stats["mean_angle"] - float(reference_mean)
                )

            return result

        except Exception as e:
            self.logger.error(f"Error predicting {image_path}: {str(e)}")
            error_result = {"image_path": str(image_path), "error": str(e)}
            if reference_mean is not None:
                error_result["reference_mean_leaf_angle"] = float(reference_mean)
            return error_result
        
    def _predict_batch(
        self, 
        image_paths: List[Path], 
        reference_means: List[Optional[float]] = None,
        batch_size: int = 16,
        num_workers: int = 8
    ) -> List[Dict[str, Any]]:
        """Efficient batch prediction with progress tracking."""
        
        # Create dataset and dataloader
        dataset = ImageDataset(image_paths, self.transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.startswith('cuda') else False,
            prefetch_factor=2,
            persistent_workers=True if num_workers > 0 else False
        )
        
        results = []
        total_images = len(image_paths)
        
        # Setup progress bar - track IMAGES, not batches
        pbar = tqdm(
            total=total_images,
            desc="Processing images",
            unit="img",
            unit_scale=True,
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Track timing for accurate rate calculation
        start_time = time.time()
        processed_images = 0
        
        with torch.no_grad():
            for batch_idx, (batch_tensors, batch_paths) in enumerate(dataloader):
                batch_start = time.time()
                
                # Move batch to device
                batch_tensors = batch_tensors.to(self.device, non_blocking=True)
                
                # Batch inference
                batch_outputs = self.model(batch_tensors)
                batch_probabilities = batch_outputs.cpu().numpy()
                
                # Process each image in batch
                batch_results = []
                for i, (probabilities, image_path) in enumerate(zip(batch_probabilities, batch_paths)):
                    global_idx = batch_idx * dataloader.batch_size + i
                    reference_mean = (
                        reference_means[global_idx] 
                        if reference_means and global_idx < len(reference_means) 
                        else None
                    )
                    
                    # Convert to angle statistics
                    angle_stats = self._probabilities_to_stats(probabilities)
                    
                    # Build result
                    result = {
                        "image_path": image_path,
                        "predicted_mean_leaf_angle": angle_stats["mean_angle"],
                        "angle_distribution": probabilities.tolist(),
                    }
                    
                    if reference_mean is not None:
                        result["reference_mean_leaf_angle"] = float(reference_mean)
                        result["prediction_error"] = angle_stats["mean_angle"] - float(reference_mean)
                        result["absolute_error"] = abs(angle_stats["mean_angle"] - float(reference_mean))
                    
                    batch_results.append(result)
                
                results.extend(batch_results)
                
                # Update progress bar with actual images processed
                current_batch_size = len(batch_paths)
                processed_images += current_batch_size
                
                # Calculate and update rate
                elapsed_time = time.time() - start_time
                current_rate = processed_images / elapsed_time if elapsed_time > 0 else 0
                
                # Update progress bar
                pbar.update(current_batch_size)
                pbar.set_postfix({
                    'rate': f'{current_rate:.1f} img/s',
                    'batch_time': f'{time.time() - batch_start:.2f}s',
                    'gpu': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU'
        })
        
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        final_rate = total_images / total_time
        self.logger.info(f"Processed {total_images} images in {total_time:.2f}s ({final_rate:.1f} img/s)")
        
        return results

    def _probabilities_to_stats(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Convert probability distribution to angle statistics."""
        # Create angle bins
        num_bins = len(probabilities)
        angles = np.linspace(0, 90, num_bins)

        # Calculate statistics
        mean_angle = np.sum(probabilities * angles)
        variance = np.sum(probabilities * (angles - mean_angle) ** 2)
        std_angle = np.sqrt(variance)

        # Calculate angle range (95% confidence interval)
        cumulative = np.cumsum(probabilities)
        lower_idx = (
            np.where(cumulative >= 0.025)[0][0]
            if len(np.where(cumulative >= 0.025)[0]) > 0
            else 0
        )
        upper_idx = (
            np.where(cumulative >= 0.975)[0][0]
            if len(np.where(cumulative >= 0.975)[0]) > 0
            else num_bins - 1
        )

        return {
            "mean_angle": float(mean_angle),
            "variance": float(variance),
            "std_angle": float(std_angle),
        }
