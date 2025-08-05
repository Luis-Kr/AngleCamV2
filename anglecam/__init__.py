"""
AngleCam - Deep Learning Framework for Leaf Angle Distribution Estimation

"""

__version__ = "2.0.0"

# Main API
from .main import AngleCam

# Sub-modules for advanced users
from . import models
from . import _data
from . import training
from . import inference
from . import utils

__all__ = [
    "AngleCam",
    "models",
    "_data", 
    "training",
    "inference",
    "utils",
]