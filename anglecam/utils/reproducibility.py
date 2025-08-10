import random
import numpy as np
import torch


def setup_reproducibility(seed: int = 42) -> None:
    """Setup reproducible training environment."""
    import os
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Force deterministic behavior (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Additional CUDA determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        #torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For transformers/huggingface models
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
