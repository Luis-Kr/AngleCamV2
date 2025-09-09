#!/usr/bin/env python3
"""
AngleCam Prediction Script - V1
"""

import tensorflow as tf
import os
import keras
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
import datetime
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Force CPU-only execution to avoid CuDNN version mismatch (slows down the script)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"Memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        logger.warning(f"GPU configuration error: {e}")

# Set working directory
PROJECT_DIR = Path(__file__).parent.parent.parent

# Configuration
CSV_FILE = PROJECT_DIR / "data" / "model" / "input" / "validation.csv"
IMAGE_DIR = PROJECT_DIR / "data" / "images" / "training_validation"
MODEL_PATH = (
    PROJECT_DIR
    / "data"
    / "model"
    / "anglecam_v01"
    / "AngleCam_efficientnet_V2L_14-03-2023.hdf5"
)
OUTPUT_DIR = PROJECT_DIR / "data" / "paper" / "results" / "anglecam_v01"
BATCH_SIZE = 32
IMAGE_SIZE = (600, 600)


def load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return None

        # Resize to target size
        image = cv2.resize(image, IMAGE_SIZE)

        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image

    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")
        return None


def create_data_generator(image_paths: list, batch_size: int = 32):
    """
    Create a generator that yields batches of preprocessed images.

    Args:
        image_paths: List of image file paths
        batch_size: Number of images per batch

    Yields:
        Batches of preprocessed images
    """
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_filenames = []

        for image_path in batch_paths:
            processed_image = load_and_preprocess_image(image_path)
            if processed_image is not None:
                batch_images.append(processed_image)
                batch_filenames.append(os.path.basename(image_path))

        if batch_images:
            yield np.array(batch_images), batch_filenames


def load_model(model_path: str):
    """
    Load the AngleCam V1 model.

    Args:
        model_path: Path to the model file

    Returns:
        Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def predict_angles(model, csv_file: str, image_dir: str, batch_size: int = 32):
    """
    Run predictions on images listed in the CSV file.

    Args:
        model: Loaded Keras model
        csv_file: Path to CSV file with image filenames
        image_dir: Directory containing the images
        batch_size: Number of images to process per batch

    Returns:
        DataFrame with predictions and metadata
    """
    # Load CSV file
    logger.info(f"Loading image list from: {csv_file}")
    df = pd.read_csv(csv_file)

    # Create full image paths
    image_paths = []
    valid_filenames = []

    for filename in df["filename"]:
        image_path = os.path.join(image_dir, filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            valid_filenames.append(filename)
        else:
            logger.warning(f"Image not found: {image_path}")

    logger.info(f"Found {len(image_paths)} valid images out of {len(df)} total")

    # Process images in batches
    all_predictions = []
    all_filenames = []

    data_generator = create_data_generator(image_paths, batch_size)

    for batch_idx, (batch_images, batch_filenames) in enumerate(data_generator):
        logger.info(f"Processing batch {batch_idx + 1}")

        # Make predictions
        predictions = model.predict(batch_images, verbose=0)

        # Scale predictions (divide by 10 as in original script)
        predictions = predictions / 10

        all_predictions.extend(predictions)
        all_filenames.extend(batch_filenames)

    # Create results DataFrame
    results_df = pd.DataFrame(all_predictions, index=all_filenames)

    # Add metadata from original CSV
    metadata_df = df[df["filename"].isin(valid_filenames)].set_index("filename")
    results_df = results_df.join(metadata_df, how="left")

    return results_df


def calculate_average_angles(predictions_df: pd.DataFrame) -> pd.Series:
    """
    Calculate average leaf angles from predictions.

    Args:
        predictions_df: DataFrame with angle distribution predictions

    Returns:
        Series with average angles
    """
    # Get only the prediction columns (first 43 columns)
    pred_cols = predictions_df.iloc[:, :43]

    # Calculate weighted average
    angles = np.linspace(0, 90, 43)  # 0 to 90 degrees in 43 steps
    avg_angles = pred_cols.apply(lambda row: np.sum(row * angles), axis=1)

    return avg_angles


def save_results(
    predictions_df: pd.DataFrame, avg_angles: pd.Series, output_dir: str = "."
):
    """
    Save prediction results and create visualizations.

    Args:
        predictions_df: DataFrame with predictions
        avg_angles: Series with average angles
        output_dir: Directory to save results
    """
    # Save predictions
    predictions_file = os.path.join(output_dir, "AngleCam_predictions.csv")
    predictions_df.to_csv(predictions_file)
    logger.info(f"Predictions saved to: {predictions_file}")

    # Create summary DataFrame
    summary_data = {
        "filename": predictions_df.index,
        "avg_angle": avg_angles,
        "species": predictions_df.get("species", "Unknown"),
        "mean_angle_ground_truth": predictions_df.get("mean_angle", np.nan),
    }

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, "leaf_angle_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Summary saved to: {summary_file}")

    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(avg_angles)), avg_angles, alpha=0.6)
    plt.xlabel("Image Index")
    plt.ylabel("Average Leaf Angle [deg]")
    plt.title("AngleCam Predictions - Average Leaf Angles")
    plt.grid(True, alpha=0.3)

    plot_file = os.path.join(output_dir, "leaf_angle_predictions.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved to: {plot_file}")

    # Print statistics
    logger.info(f"Prediction Statistics:")
    logger.info(f"  Number of images processed: {len(avg_angles)}")
    logger.info(f"  Mean predicted angle: {avg_angles.mean():.2f}°")
    logger.info(f"  Std predicted angle: {avg_angles.std():.2f}°")
    logger.info(f"  Min predicted angle: {avg_angles.min():.2f}°")
    logger.info(f"  Max predicted angle: {avg_angles.max():.2f}°")


def main():
    """Main execution function."""
    try:
        # Load model
        logger.info("Loading AngleCam model...")
        model = load_model(MODEL_PATH)

        # Run predictions
        logger.info("Starting predictions...")
        predictions_df = predict_angles(model, CSV_FILE, IMAGE_DIR, BATCH_SIZE)

        # Calculate average angles
        logger.info("Calculating average angles...")
        avg_angles = calculate_average_angles(predictions_df)

        # Save results
        logger.info("Saving results...")
        save_results(predictions_df, avg_angles, OUTPUT_DIR)

        logger.info("Prediction pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
