from setuptools import setup, find_packages
import os


def create_data_directories():
    """Create necessary data directories for AngleCam."""
    directories = [
        "data",
        "data/checkpoint",
        "data/01_Training_Validation_Data",
        "data/01_Training_Validation_Data/image_data",
        "data/01_Training_Validation_Data/splits",
        "data/outputs",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file to ensure directory is tracked
        gitkeep_path = os.path.join(directory, ".gitkeep")
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, "w") as f:
                f.write("")


# Create directories when package is installed
create_data_directories()

setup(
    name="anglecam",
    version="2.0.0",
    packages=find_packages(),
    python_requires=">=3.9",
)
