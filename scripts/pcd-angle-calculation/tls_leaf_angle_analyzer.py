import open3d as o3d
import numpy as np
from pathlib import Path
import laspy
import json
import logging
from tqdm import tqdm
from scipy.spatial import cKDTree
from numba import njit, prange
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
from scipy import stats

class TqdmLoggingHandler(logging.Handler):
    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# Set up logging with custom handler
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(TqdmLoggingHandler())
formatter = logging.Formatter('%(asctime)s - %(message)s')
logger.handlers[0].setFormatter(formatter)

# Remove any existing handlers to avoid duplicate output
for handler in logger.handlers[1:]:
    logger.removeHandler(handler)

@dataclass
class LeafAngleConfig:
    """Configuration parameters for leaf angle analysis."""
    # Voxel downsampling
    voxel_size: float = 0.01  # Size of voxels for downsampling (in meters)
    
    # Normal estimation parameters
    neighbor_search_radius: float = 0.0075  # Radius for neighbor search (in meters)
    min_neighbors: int = 6  # Minimum number of neighbors for normal estimation
    
    # Angle calculation
    angle_bin_size: int = 2  # Size of angle bins in degrees
    max_angle: int = 90  # Maximum angle to consider
    
    def __post_init__(self):
        """Generate angle bins after initialization."""
        self.angle_bins = np.arange(0, self.max_angle + 1, self.angle_bin_size)

@njit(parallel=True)
def adjust_normals(normals: np.ndarray, scanner_LOS: np.ndarray) -> np.ndarray:
    """Adjust normals to face away from scanner."""
    # Ensure inputs are contiguous
    normals = np.ascontiguousarray(normals)
    scanner_LOS = np.ascontiguousarray(scanner_LOS)
    for i in prange(len(normals)):
        if np.dot(normals[i], scanner_LOS[i]) < 0:
            normals[i] = -normals[i]
    return normals

def calculate_angle_statistics(angles: np.ndarray) -> Dict:
    """Calculate statistical parameters for leaf angles."""
    # Basic statistics
    mean = float(np.mean(angles))
    std = float(np.std(angles))
    variance = float(np.var(angles))
    
    # Calculate 95% confidence interval
    confidence_level = 0.95
    degrees_of_freedom = len(angles) - 1
    t_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    margin_of_error = t_value * (std / np.sqrt(len(angles)))
    ci_lower = float(mean - margin_of_error)
    ci_upper = float(mean + margin_of_error)
    
    # Skewness and Kurtosis
    skewness = float(stats.skew(angles))
    kurtosis = float(stats.kurtosis(angles))
    
    # Percentiles
    percentiles = np.percentile(angles, [25, 50, 75])
    q1, median, q3 = float(percentiles[0]), float(percentiles[1]), float(percentiles[2])
    iqr = q3 - q1
    
    # Mode (using histogram to find most frequent bin)
    hist, bin_edges = np.histogram(angles, bins=50)
    mode_bin_index = np.argmax(hist)
    mode = float((bin_edges[mode_bin_index] + bin_edges[mode_bin_index + 1]) / 2)
    
    # Range
    angle_range = float(np.max(angles) - np.min(angles))
    
    # Coefficient of Variation
    cv = (std / mean) * 100 if mean != 0 else 0
    
    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'confidence_interval': {
            'level': confidence_level,
            'lower': ci_lower,
            'upper': ci_upper,
            'margin_of_error': float(margin_of_error)
        },
        'skewness': skewness,
        'kurtosis': kurtosis,
        'percentiles': {
            'q1': q1,
            'median': median,
            'q3': q3,
            'iqr': iqr
        },
        'mode': mode,
        'range': angle_range,
        'cv_percent': float(cv)
    }

class LeafAngleAnalyzer:
    def __init__(self, input_dir: Path, output_dir: Optional[Path] = None, config: Optional[LeafAngleConfig] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "processed_angles"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zenith = np.array([0, 0, 1])
        self.config = config or LeafAngleConfig()  # Use default config if none provided
        self.test_mode = False

    def compute_scanner_LOS(self, points: np.ndarray) -> np.ndarray:
        """Compute line of sight vectors from scanner to points."""
        scanner_LOS = points[:, :3] / np.linalg.norm(points[:, :3], axis=1, keepdims=True)
        return -scanner_LOS

    def compute_angle_distribution(self, angles: np.ndarray) -> np.ndarray:
        """Compute histogram of angles in 2-degree bins."""
        hist, _ = np.histogram(angles, bins=self.config.angle_bins)
        return hist / len(angles)  # normalize to get distribution

    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud"):
        """Visualize point cloud with normals."""
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        
        # Set colors for better visualization
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray for points
        
        # Visualization options
        opt = o3d.visualization.Visualizer()
        opt.create_window(window_name=window_name)
        opt.add_geometry(pcd)
        opt.add_geometry(coord_frame)
        
        # Run visualization
        opt.run()
        opt.destroy_window()

    def process_file(self, file_path: Path) -> Dict:
        """Process a single point cloud file."""
        try:
            logging.info(f"Processing {file_path.name}")
            
            # Load point cloud
            las_data = laspy.read(file_path)
            points = np.vstack((las_data.x, las_data.y, las_data.z)).T
            
            # Center points and compute LOS
            centroid = np.mean(points, axis=0)
            centered_points = points - centroid
            scanner_LOS = self.compute_scanner_LOS(points)
            
            # Create Open3D point cloud and compute normals
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(centered_points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.neighbor_search_radius,
                    max_nn=100
                )
            )
            
            # Get computed normals
            normals = np.asarray(pcd.normals)
            
            # Adjust normals orientation
            normals = adjust_normals(normals, scanner_LOS)
            
            # Get downsampled points and normals
            original_points = np.asarray(pcd.points) + centroid  # Add centroid back
            
            # Remove points with invalid normals (zero vectors)
            valid_normals = np.any(normals != 0, axis=1)
            original_points = original_points[valid_normals]
            normals = normals[valid_normals]
            
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(original_points)
            
            # Downsample the point cloud
            downsampled_pcd = original_pcd.voxel_down_sample(self.config.voxel_size)
            downsampled_points = np.asarray(downsampled_pcd.points)
            
            # Find closest original points to filtered downsampled points
            tree = cKDTree(points)
            _, kept_indices = tree.query(downsampled_points, k=1)
            kept_indices = np.unique(kept_indices)  # Remove any duplicates
            
            # Update final points and normals
            downsampled_points = original_points[kept_indices]
            downsampled_normals = normals[kept_indices]
            
            # Create final LAS file for downsampled points
            final_las = laspy.create(point_format=las_data.header.point_format,
                                     file_version=las_data.header.version)
            
            # Copy header properties
            final_las.header.scales = las_data.header.scales
            final_las.header.offsets = las_data.header.offsets
            
            # Create new points array with correct dimensions
            #point_count = len(kept_indices)
            #final_las.points = laspy.PackedPointRecord.zeros(point_count, final_las.header.point_format)
            
            # Copy all attributes for kept points
            for dim in las_data.point_format.dimension_names:
                if hasattr(las_data, dim):
                    setattr(final_las, dim, las_data[dim][kept_indices])
                        
            # Calculate angles and verticality
            verticality = 1 - np.abs(np.dot(downsampled_normals, self.zenith))
            angles = np.arccos(np.abs(np.dot(downsampled_normals, self.zenith)))
            angles = np.rad2deg(angles)
            
            # Add new attributes (normals, angles, verticality)
            final_las.add_extra_dim(laspy.ExtraBytesParams(name='normal_x', type=np.float32))
            final_las.add_extra_dim(laspy.ExtraBytesParams(name='normal_y', type=np.float32))
            final_las.add_extra_dim(laspy.ExtraBytesParams(name='normal_z', type=np.float32))
            final_las.add_extra_dim(laspy.ExtraBytesParams(name='leaf_angle', type=np.float32))
            final_las.add_extra_dim(laspy.ExtraBytesParams(name='verticality', type=np.float32))
            
            # Assign computed values
            final_las.normal_x = downsampled_normals[:, 0].astype(np.float32)
            final_las.normal_y = downsampled_normals[:, 1].astype(np.float32)
            final_las.normal_z = downsampled_normals[:, 2].astype(np.float32)
            final_las.leaf_angle = angles.astype(np.float32)
            final_las.verticality = verticality.astype(np.float32)
            
            # Save modified LAS file
            output_file = self.output_dir / file_path.name
            final_las.write(output_file)
            
            logging.info(f"Subsampled from {len(las_data.points)} to {len(final_las.points)} points")
            
            return {
                'filename': file_path.name,
                'original_points': len(las_data.points),
                'subsampled_points': len(final_las.points),
                'average_angle': float(np.mean(angles)),
                'average_verticality': float(np.mean(verticality)),
                'angle_distribution': self.compute_angle_distribution(angles).tolist(),
                'angle_statistics': calculate_angle_statistics(angles)
            }
            
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {str(e)}")
            return None

    def process_all_files(self):
        """Process all LAS files in the input directory and save results."""
        results = []
        
        las_files = list(self.input_dir.glob('*.las'))
        total_files = len(las_files)
        
        logging.info(f"Found {total_files} LAS files in {self.input_dir}")
        
        if total_files == 0:
            logging.warning(f"No LAS files found in {self.input_dir}")
            return
            
        if self.test_mode:
            # Process only first file in test mode
            file_path = las_files[0]
            logging.info(f"Test mode: Processing only {file_path.name}")
            
            # Load and process the point cloud
            las_data = laspy.read(file_path)
            points = np.vstack((las_data.x, las_data.y, las_data.z)).T
            
            # Create and process point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.neighbor_search_radius,
                    max_nn=30
                )
            )
            
            # Visualize original point cloud with normals
            logging.info("Showing original point cloud with normals. Close window to continue processing.")
            if os.environ.get('DISPLAY'):
                self.visualize_point_cloud(pcd, "Original Point Cloud with Normals")
            
            # Process the file normally
            result = self.process_file(file_path)
            if result:
                results.append(result)
        else:
            # Normal processing for all files
            with tqdm(total=total_files, desc="Processing files", unit="file", 
                     dynamic_ncols=True, position=0, leave=True) as pbar:
                for file_path in las_files:
                    result = self.process_file(file_path)
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        # Save results to JSON
        output_json = self.output_dir / 'leaf_angle_analysis.json'
        with open(output_json, 'w') as f:
            json.dump({
                'angle_bins': self.config.angle_bins.tolist(),
                'results': results
            }, f, indent=2)
        
        logging.info(f"Analysis complete. Results saved to {output_json}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Leaf Angle Analyzer")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing LAS files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save processed files.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with visualization")
    args = parser.parse_args()

    # Add input validation and logging
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")
    if args.test:
        logging.info("Running in test mode with visualization")

    config = LeafAngleConfig()
    analyzer = LeafAngleAnalyzer(input_dir, output_dir, config)
    analyzer.test_mode = args.test
    analyzer.process_all_files()

if __name__ == "__main__":
    main()
