import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Union, List, Optional, Any

class AngleCam:
    """
    Main AngleCam interface for leaf angle estimation.
    
    Provides unified API for training and prediction.
    """
    
    def __init__(self, config: DictConfig):
        """Initialize AngleCam with Hydra configuration."""
        self.config = config
        self.model = None
        self.trainer = None
        self.predictor = None
        self._setup_logging()
        self._setup_reproducibility()
    
    @classmethod
    def from_config(cls, config_path: str) -> "AngleCam":
        """Create AngleCam instance from Hydra config file."""
        with hydra.initialize(config_path="../config"):
            config = hydra.compose(config_name=config_path)
        return cls(config)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, config_overrides: Optional[dict] = None) -> "AngleCam":
        """Load trained AngleCam from checkpoint."""
        # Implementation here
        pass
    
    def train(self, data_path: str, **kwargs) -> dict:
        """
        Train the AngleCam model on provided data.
        
        Args:
            data_path: Path to training data CSV file
            **kwargs: Override config parameters
            
        Returns:
            Training results dictionary
        """
        # Implementation here
        pass
    
    def predict(self, input_data: Union[str, List[str], Path]) -> Union[dict, List[dict]]:
        """
        Predict leaf angles from image(s).
        
        Args:
            input_data: Single image path, list of paths, or directory
            
        Returns:
            Prediction results with angle distributions and statistics
        """
        # Implementation here
        pass
    
    def save(self, save_path: str) -> None:
        """Save trained model and configuration."""
        # Implementation here
        pass