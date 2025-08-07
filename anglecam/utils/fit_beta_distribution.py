import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging  # Add this - use Python's built-in logging

import hydra
from omegaconf import DictConfig

class BetaDistributionFitter:
    def __init__(self, 
                 sd_fraction=0.2,
                 sim_per_pic=50,
                 angle_resolution=43,
                 scaling_factor=10,
                 ncp=0):
        self.sd_fraction = sd_fraction
        self.sim_per_pic = sim_per_pic
        self.angle_resolution = angle_resolution
        self.scaling_factor = scaling_factor
        self.ncp = ncp
        self.bins = np.linspace(0, 90, angle_resolution)  

    def fit_mle(self, data):
        """Fit beta distribution using Maximum Likelihood Estimation"""
        # Scale angles from 0-90 to 0-1
        scaled_data = np.clip(data / 90.0, 0.00001, 0.99999)
        
        # Fit beta distribution
        a, b, loc, scale = stats.beta.fit(scaled_data, floc=0, fscale=1)
        
        # Calculate standard errors through Fisher Information Matrix
        x = scaled_data
        def neg_log_likelihood(params):
            return -np.sum(stats.beta.logpdf(x, params[0], params[1]))
        
        result = minimize(neg_log_likelihood, [a, b], method='Nelder-Mead')
        hess_inv = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(2)
        std_errors = np.sqrt(np.diag(hess_inv))
        
        return {'estimate': (a, b), 'sd': std_errors}

    def generate_distributions(self, fit_result):
        """Generate reference and simulated distributions"""
        x = np.linspace(0, 1, self.angle_resolution)
        
        # Incorporate self.ncp by shifting the x values.
        # Convert ncp from degrees to fraction on [0,1]
        x_adj = np.clip(x - self.ncp/90.0, 0.00001, 0.99999)
        
        # Generate reference distribution using the shifted x values
        ref_dist = stats.beta.pdf(x_adj, fit_result['estimate'][0], fit_result['estimate'][1])
        
        # Handle invalid values and prevent division by zero
        ref_dist = np.nan_to_num(ref_dist, nan=0.0, posinf=0.0, neginf=0.0)
        sum_ref = np.sum(ref_dist)
        ref_dist = ref_dist / sum_ref if sum_ref > 0 else ref_dist
        ref_dist = ref_dist * self.scaling_factor
        
        # Generate simulated distributions
        sims = []
        for _ in range(self.sim_per_pic):
            # Sample parameters with error
            a = fit_result['estimate'][0] + np.random.normal(0, fit_result['sd'][0]) * self.sd_fraction
            b = fit_result['estimate'][1] + np.random.normal(0, fit_result['sd'][1]) * self.sd_fraction
            
            try:
                # Use the same shifted x values for simulation
                dist = stats.beta.pdf(x_adj, a, b)
                
                # Handle invalid values and prevent division by zero
                dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
                sum_dist = np.sum(dist)
                if sum_dist > 0:
                    dist = dist / sum_dist * self.scaling_factor
                    sims.append(dist)
            except:
                continue
                
        sims = np.array(sims)
        
        # Only proceed if we have valid simulations
        if len(sims) > 0:
            # Combine reference and simulations, remove first and last points
            all_dist = np.vstack([ref_dist[1:-1], sims[:, 1:-1]])
            
            # Scale back to 0-0.9 range
            return all_dist * 90 * 0.01
        else:
            # Return empty array if no valid simulations
            return np.array([])

    @staticmethod
    def mean_angle(pred: np.ndarray) -> float:
        """Calculate mean angle from prediction array."""
        angle_res = pred.shape[0]
        angles = np.linspace(0, 90, angle_res)
        return np.sum(pred/10 * angles)

    def plot_distribution(self, angles, fit_result, output_path):
        """Plot histogram and fitted beta distribution"""
        plt.figure(figsize=(10, 6))
        
        # Create custom colormap for simulations
        colors = plt.cm.rainbow(np.linspace(0, 1, self.sim_per_pic))
        
        # Plot histogram of actual data
        counts, bins, _ = plt.hist(angles, bins=self.bins, density=True, 
                                 alpha=0.6, label='Data', color='gray')
        
        # Plot fitted beta distribution (reference)
        x = np.linspace(0, 90, 200)
        # Shift x values by self.ncp (converted to fraction)
        x_adj = np.clip((x - self.ncp)/90.0, 0.00001, 0.99999)
        a, b = fit_result['estimate']
        y = stats.beta.pdf(x_adj, a, b) / 90.0
        
        # Calculate mean angle for fitted distribution
        y_for_mean = stats.beta.pdf(np.linspace(0, 1, self.angle_resolution) - self.ncp/90.0, a, b)
        y_for_mean = np.nan_to_num(y_for_mean, nan=0.0, posinf=0.0, neginf=0.0)
        sum_y = np.sum(y_for_mean)
        if sum_y > 0:
            y_for_mean = y_for_mean / sum_y
        mean_angle = self.mean_angle(y_for_mean * 10)  # Scale by 10 to match expected input scale
        
        plt.plot(x, y, 'k-', lw=2, label='Fitted Beta')
        
        # Plot simulated distributions with unique colors
        for i in range(self.sim_per_pic):
            a_sim = fit_result['estimate'][0] + np.random.normal(0, fit_result['sd'][0]) * self.sd_fraction
            b_sim = fit_result['estimate'][1] + np.random.normal(0, fit_result['sd'][1]) * self.sd_fraction
            try:
                # Use shifted x values for simulation
                y_sim = stats.beta.pdf(x_adj, a_sim, b_sim) / 90.0
                plt.plot(x, y_sim, '-', lw=0.5, alpha=0.3, color=colors[i])
            except:
                continue
        
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Density')
        plt.title(f'Angle Distribution with Beta Fit and Simulations\nMean Angle: {mean_angle:.1f}°')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = str(Path(output_path).with_suffix('')) + '_plot.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()

    def process_file(self, file_path):
        """Process a single CSV file"""
        # Read data
        df = pd.read_csv(file_path, sep=',')    
        angles = df['angle'].values
        
        # Fit distribution and generate variants
        fit_result = self.fit_mle(angles)
        distributions = self.generate_distributions(fit_result)
        
        # Plot histogram and fitted distribution
        self.plot_distribution(angles, fit_result, file_path)
        
        # Save results
        output_path = str(Path(file_path).with_suffix('')) + '_sim.csv'
        np.savetxt(output_path, distributions, delimiter=' ')
        
    def process_directory(self, directory):
        """Process all CSV files in directory"""
        csv_files = [f for f in os.listdir(directory) 
                    if f.endswith('.csv') and 'sim' not in f and 'optuna' not in f]
        
        for file in csv_files:
            self.process_file(os.path.join(directory, file))

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    fitter = BetaDistributionFitter(
        sd_fraction=cfg.fit_dist.sd_fraction,
        sim_per_pic=cfg.fit_dist.sim_per_pic,
        angle_resolution=cfg.fit_dist.angle_resolution,
        scaling_factor=cfg.fit_dist.scaling_factor,
        ncp=cfg.fit_dist.ncp
    )
    fitter.process_directory(cfg.fit_dist.input_path)

if __name__ == "__main__":
    main()